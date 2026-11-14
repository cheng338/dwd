"""Bounded exact-coordinate original-feature MM step.

For supplied binary64 h, positive binary64 s and Z=[1,X], return coordinatewise
nearest-even binary64 of the exact solution to (Z.T Z+s*diag(0,I))*new=Z.T*(Z*old-h).
No error tolerance, eigen clipping, ridge change, SVD or rank truncation decides
acceptance. Raw-bit decoding and exact rounding-cell checks bind stored output.
This contract does not certify nonlinear h evaluation, public score rounding,
convergence or arbitrary problem-size availability.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction as F
import hashlib
import math
import struct

import numpy as np


@dataclass(frozen=True)
class ExactCoordinatePolicy:
    max_entries: int = 65536
    max_features: int = 16
    max_integer_bits: int = 32768
    max_exact_work: int = 10000000
    max_workspace_bytes: int = 2**30


class ExactCoordinateLimitError(ArithmeticError):
    def __init__(self, reason, stage, usage):
        super().__init__(reason + ' at ' + stage)
        self.reason, self.stage, self.usage = reason, stage, dict(usage)


class _Budget:
    def __init__(self, policy):
        self.policy = policy
        self.work = self.bits = self.workspace = 0
        self.stage = 'preparation'

    def usage(self):
        return dict(exact_work=self.work, peak_integer_bits=self.bits,
                    modeled_workspace_bytes=self.workspace,
                    accounting='Source-defined rational operation reservations, not machine instructions or time. Bit preflights cover managed exact arithmetic; conversion/serialization temporaries are separate. Storage is a source model, not hard RSS; caller-retained receipts are excluded.')

    def fail(self, reason):
        raise ExactCoordinateLimitError(reason, self.stage, self.usage())

    def tick(self, count=1):
        if self.work + count > self.policy.max_exact_work:
            self.fail('exact_work_exhausted')
        self.work += count

    def width(self, bits):
        if bits > self.policy.max_integer_bits:
            self.fail('integer_bits_exhausted')
        self.bits = max(self.bits,bits)

    def value(self, value):
        self.tick()
        value = value if isinstance(value,F) else F(value)
        self.width(max(abs(value.numerator).bit_length(),value.denominator.bit_length()))
        return value

    def add(self,a,b):
        self.tick()
        self.width(max(abs(a.numerator).bit_length()+b.denominator.bit_length(),
                       abs(b.numerator).bit_length()+a.denominator.bit_length())+1)
        self.width(a.denominator.bit_length()+b.denominator.bit_length())
        return a+b

    def sub(self,a,b):
        return self.add(a,-b)

    def mul(self,a,b):
        self.tick()
        self.width(max(abs(a.numerator).bit_length()+abs(b.numerator).bit_length(),
                       a.denominator.bit_length()+b.denominator.bit_length()))
        return a*b

    def div(self,a,b):
        if not b:
            raise ArithmeticError('Unexpected zero exact denominator.')
        self.tick()
        self.width(max(abs(a.numerator).bit_length()+b.denominator.bit_length(),
                       a.denominator.bit_length()+abs(b.numerator).bit_length()))
        return a/b

    def total(self,values):
        result=F(0)
        for value in values:
            result=self.add(result,value)
        return result

    def dot(self,a,b):
        return self.total(self.mul(x,y) for x,y in zip(a,b))

    def le(self,a,b):
        self.tick()
        self.width(max(abs(a.numerator).bit_length()+b.denominator.bit_length(),
                       abs(b.numerator).bit_length()+a.denominator.bit_length()))
        return a<=b


_SIGN=1<<63
_MAGNITUDE=_SIGN-1
_MAX_FINITE=0x7fefffffffffffff


def _word(value,name):
    if type(value) not in (float,np.float64):
        raise TypeError(name+' must be a binary64 scalar.')
    payload=value.tobytes() if type(value) is np.float64 else struct.pack('=d',value)
    return struct.unpack('=Q',payload)[0]


def _decode(word,budget):
    exponent=(word>>52)&0x7ff
    fraction=word&((1<<52)-1)
    if exponent==0x7ff:
        raise ValueError('All numerical operands must be finite binary64 values.')
    mantissa=fraction if exponent==0 else (1<<52)|fraction
    power=-1074 if exponent==0 else exponent-1023-52
    if not mantissa:
        return budget.value(F(0))
    budget.width(max(mantissa.bit_length()+max(0,power),1+max(0,-power)))
    numerator=mantissa<<power if power>=0 else mantissa
    denominator=1 if power>=0 else 1<<-power
    if word&_SIGN:
        numerator=-numerator
    return budget.value(F(numerator,denominator))


def _rounding_word(value,budget):
    """Propose a float, then prove its exact nearest-even cell using raw bits."""
    budget.stage='coordinate_rounding'
    if not value:
        return 0  # Mathematical zero has the documented +0 convention.
    try:
        proposed=float(value)
    except (OverflowError,ValueError) as exc:
        raise FloatingPointError('The exact coordinate has no verified finite binary64 rounding.') from exc
    word=struct.unpack('=Q',struct.pack('=d',proposed))[0]
    magnitude=word&_MAGNITUDE
    if magnitude>_MAX_FINITE:
        raise FloatingPointError('The exact coordinate has no verified finite binary64 rounding.')
    absolute=abs(value)
    rounded=_decode(magnitude,budget)
    previous=_decode(magnitude-1,budget) if magnitude else F(0)
    if magnitude==_MAX_FINITE:
        budget.width(1025)
        following=budget.value(F(1<<1024))
    else:
        following=_decode(magnitude+1,budget)
    lower=budget.div(budget.add(previous,rounded),F(2))
    upper=budget.div(budget.add(rounded,following),F(2))
    even=(magnitude&1)==0
    lower_ok=(not magnitude or (budget.le(lower,absolute) if even else not budget.le(absolute,lower)))
    upper_ok=(budget.le(absolute,upper) if even else not budget.le(upper,absolute))
    if not lower_ok or not upper_ok:
        raise FloatingPointError('The proposed binary64 coordinate failed exact nearest-even verification.')
    # Sign comes from the exact rational, including negative underflow to -0.
    return magnitude|(_SIGN if value<0 else 0)


def _array_payload(value,shape,name):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype(np.float64):
        raise TypeError(name+' must be a native float64 ndarray.')
    if value.shape!=shape:
        raise ValueError(name+' has the wrong shape.')
    return value.tobytes(order='C')


def _words(payload):
    return [word for (word,) in struct.iter_unpack('=Q',payload)]


class ExactCoordinateMM:
    """Exact LDL preparation and fixed-h next iterates, with shared finite caps."""
    def __init__(self,X,shift,*,policy=ExactCoordinatePolicy()):
        if type(policy) is not ExactCoordinatePolicy:
            raise TypeError('policy must be ExactCoordinatePolicy.')
        if any(type(v) is not int or v<0 for v in asdict(policy).values()) or policy.max_features>16:
            raise ValueError('Policy limits must be nonnegative integers and max_features<=16.')
        if not isinstance(X,np.ndarray) or X.dtype!=np.dtype(np.float64):
            raise TypeError('X must be a native float64 ndarray.')
        if X.ndim!=2 or min(X.shape)<1:
            raise ValueError('X must be nonempty and two-dimensional.')
        self.n,self.p=X.shape
        self.policy=policy
        self.budget=_Budget(policy)
        b=self.budget
        if X.size>policy.max_entries or self.p>policy.max_features:
            b.fail('shape_limit_exhausted')
        cell=128+2*(32+4*((policy.max_integer_bits+29)//30))
        workspace=(X.size*512+(8*self.n+8*self.p*self.p+32*self.p+64)*cell
                   +64*(X.size+self.p*self.p+self.n)+2**20)
        if workspace>policy.max_workspace_bytes:
            b.fail('workspace_limit_exhausted')
        b.workspace=workspace
        self.shift_word=_word(shift,'shift')
        self.s=_decode(self.shift_word,b)
        if self.s<=0:
            raise ValueError('The raw represented shift must be strictly positive.')
        payload=X.tobytes(order='C')
        self.X=np.frombuffer(payload,dtype=np.float64).reshape(self.n,self.p)
        self.input_sha256=hashlib.sha256(payload).hexdigest()
        exact=[_decode(word,b) for word in _words(payload)]
        self.exact_X=tuple(tuple(exact[i*self.p:(i+1)*self.p]) for i in range(self.n))
        self.nf=b.value(self.n)
        self.columns=tuple(b.total(row[j] for row in self.exact_X) for j in range(self.p))
        self.factorizations_completed=0
        self.actions_completed=0
        self._prepare_ldl()

    def _prepare_ldl(self):
        b=self.budget
        b.stage='exact_LDL_preparation'
        size=self.p+1
        P=[[F(0) for _ in range(size)] for _ in range(size)]
        P[0][0]=self.nf
        for j in range(self.p):
            P[0][j+1]=P[j+1][0]=self.columns[j]
            for k in range(j+1):
                value=b.total(b.mul(row[j],row[k]) for row in self.exact_X)
                if j==k:
                    value=b.add(value,self.s)
                P[j+1][k+1]=P[k+1][j+1]=value
        L=[[F(int(i==j)) for j in range(size)] for i in range(size)]
        D=[]
        for j in range(size):
            diagonal=b.sub(P[j][j],b.total(b.mul(b.mul(L[j][k],L[j][k]),D[k]) for k in range(j)))
            if diagonal<=0:
                raise ArithmeticError('Exact positive-shift LDL invariant failed.')
            D.append(diagonal)
            for i in range(j+1,size):
                numerator=b.sub(P[i][j],b.total(b.mul(b.mul(L[i][k],L[j][k]),D[k]) for k in range(j)))
                L[i][j]=b.div(numerator,diagonal)
        self.L=tuple(tuple(row) for row in L)
        self.D=tuple(D)
        self.factorizations_completed+=1

    def next_iterate(self,h,beta,offset):
        b=self.budget
        b.stage='exact_original_feature_RHS'
        h_payload=_array_payload(h,(self.n,),'h')
        beta_payload=_array_payload(beta,(self.p,),'beta')
        offset_word=_word(offset,'offset')
        hf=[_decode(word,b) for word in _words(h_payload)]
        bf=[_decode(word,b) for word in _words(beta_payload)]
        intercept=_decode(offset_word,b)
        target=[b.sub(b.add(intercept,b.dot(row,bf)),hi) for row,hi in zip(self.exact_X,hf)]
        rhs=[b.total(target)]+[b.total(b.mul(row[j],t) for row,t in zip(self.exact_X,target)) for j in range(self.p)]
        b.stage='exact_LDL_solve'
        y=[]
        for i in range(self.p+1):
            y.append(b.sub(rhs[i],b.dot(self.L[i][:i],y)))
        z=[b.div(x,d) for x,d in zip(y,self.D)]
        exact=[F(0)]*(self.p+1)
        for i in range(self.p,-1,-1):
            exact[i]=b.sub(z[i],b.total(b.mul(self.L[j][i],exact[j]) for j in range(i+1,self.p+1)))
        output_words=[_rounding_word(value,b) for value in exact]
        output_payload=b''.join(struct.pack('=Q',word) for word in output_words)
        result=np.frombuffer(output_payload,dtype=np.float64)
        if result.tobytes()!=output_payload or _words(result.tobytes())!=output_words:
            raise ArithmeticError('The actual returned coordinate storage changed.')
        b.stage='exact_rounding_error_record'
        differences=[b.sub(_decode(word,b),value) for word,value in zip(output_words,exact)]
        scores=[b.add(differences[0],b.dot(row,differences[1:])) for row in self.exact_X]
        errors=dict(beta_error_squared=str(b.dot(differences[1:],differences[1:])),
                    score_error_squared=str(b.dot(scores,scores)),
                    intercept_error_abs=str(abs(differences[0])))
        self.actions_completed+=1
        receipt=dict(schema='f11-exact-coordinate-next-v1',status='exact_coordinates_rounded',
                     target='original_feature_fixed_h',input_sha256=self.input_sha256,
                     shift_bits=format(self.shift_word,'016x'),
                     h_sha256=hashlib.sha256(h_payload).hexdigest(),
                     beta_sha256=hashlib.sha256(beta_payload).hexdigest(),
                     offset_bits=format(offset_word,'016x'),
                     exact_theta=list(map(str,exact)),
                     theta_bits=[format(word,'016x') for word in output_words],
                     theta_sha256=hashlib.sha256(output_payload).hexdigest(),
                     coordinate_rounding_verified=True,rounding='binary64_round_to_nearest_ties_to_even',
                     signed_zero='Exact zero maps to +0; nonzero underflow retains the exact sign.',
                     reference_errors=errors,
                     reference_error_scope='Error of coordinatewise nearest-even rounding, not a proven global score-error lower bound or a production error threshold.',
                     factorizations_completed=self.factorizations_completed,actions_completed=self.actions_completed,
                     resource_policy=asdict(self.policy),budget_usage=b.usage())
        return result,receipt
