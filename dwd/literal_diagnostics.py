"""Explicit, bounded diagnostics for a literal stored kernel objective.

This module never changes a fit, convergence decision, or numerical tolerance.
Its penalty operator is S=(K+K.T)/2 because a.T K a == a.T S a in exact
arithmetic. K itself is never replaced, and the score map remains K a + b.
All binary floating inputs are treated as their exact represented numbers.
Decimal transcendental enclosures assume correctly rounded ln/exp.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING, DecimalException
from fractions import Fraction as F
import math
import numpy as np
from ._dual_certificate import feasible_dual_slopes, _conjugate_mean_lower


class DiagnosticRefusal(ArithmeticError):
    """The requested diagnostic is outside its explicit arithmetic contract."""


class _Arithmetic:
    def __init__(self, max_bits=8192, max_operations=100000):
        self.max_bits=max_bits
        self.max_operations=max_operations
        self.operations=0

    def check(self,x):
        self.operations+=1
        if self.operations>self.max_operations:
            raise DiagnosticRefusal('operation budget exhausted')
        if max(abs(x.numerator).bit_length(),x.denominator.bit_length())>self.max_bits:
            raise DiagnosticRefusal('rational bit budget exhausted')
        return x

    def add(self,a,b):
        # Preflight both cross-products: cancellation cannot hide work size.
        if max(abs(a.numerator).bit_length()+b.denominator.bit_length(),abs(b.numerator).bit_length()+a.denominator.bit_length(),a.denominator.bit_length()+b.denominator.bit_length())>2*self.max_bits:
            raise DiagnosticRefusal('rational intermediate budget exhausted')
        return self.check(a+b)

    def mul(self,a,b):
        if max(abs(a.numerator).bit_length()+abs(b.numerator).bit_length(),a.denominator.bit_length()+b.denominator.bit_length())>2*self.max_bits:
            raise DiagnosticRefusal('rational intermediate budget exhausted')
        return self.check(a*b)

    def div(self,a,b):
        if not b:raise DiagnosticRefusal('zero exact divisor')
        return self.mul(a,1/b)

    def total(self,values):
        result=F(0)
        for value in values:result=self.add(result,value)
        return result


def _fraction(value):
    return F.from_float(float(value))


def _array(value,name,shape):
    # Refuse oversized/mismatched inputs before conversion or a finite scan.
    # Numeric arrays and ordinary nested sequences have inspectable dimensions.
    declared=getattr(value,'shape',None)
    if declared is not None:
        if tuple(declared)!=shape:raise ValueError(name+' has incompatible shape')
    elif isinstance(value,(list,tuple)):
        if len(value)!=shape[0]:raise ValueError(name+' has incompatible shape')
        if len(shape)==2 and any(not hasattr(row,'__len__') or len(row)!=shape[1] for row in value):
            raise ValueError(name+' has incompatible shape')
        rows=[value] if len(shape)==1 else value
        if any(not isinstance(item,(int,float,np.integer,np.floating)) for row in rows for item in row):
            raise ValueError(name+' sequence must contain real numeric scalars')
    else:raise ValueError(name+' requires an array or bounded ordinary sequence')
    raw=np.asarray(value)
    if raw.shape!=shape:raise ValueError(name+' has incompatible shape')
    if raw.dtype.kind not in 'iuf':
        raise ValueError(name+' must contain real numeric values')
    result=np.asarray(raw,dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError(name+' must be finite')
    return result


def _scalar(value,name,positive=False):
    if isinstance(value,(list,tuple,dict,str,bytes)):
        raise ValueError(name+' must be a finite real scalar')
    declared=getattr(value,'shape',None)
    if declared is not None and tuple(declared)!=():
        raise ValueError(name+' must be a finite real scalar')
    if np.ndim(value)!=0 or np.asarray(value).dtype.kind not in 'iuf':
        raise ValueError(name+' must be a finite real scalar')
    try:result=float(value)
    except (TypeError,ValueError,OverflowError) as e:raise ValueError(name+' must be a finite real scalar') from e
    if not math.isfinite(result) or (positive and result<=0):raise ValueError(name+' is outside its domain')
    return result


def _decimal(x,rounding,precision):
    with localcontext() as c:
        c.prec=precision;c.rounding=rounding
        return Decimal(x.numerator)/Decimal(x.denominator)


def _power_interval(base,exponent,ar,precision):
    """Outward enclosure for 0<base<=1 and positive rational exponent."""
    if not 0<base<=1 or exponent<=0:raise DiagnosticRefusal('power domain')
    if base==1:return F(1),F(1)
    if exponent.denominator==1 and exponent.numerator<=32:
        value=F(1)
        for _ in range(exponent.numerator):value=ar.mul(value,base)
        return value,value
    try:
        with localcontext() as c:
            c.prec=precision
            lo=_decimal(base,ROUND_FLOOR,precision)
            hi=_decimal(base,ROUND_CEILING,precision)
            loglo=F(lo.ln().next_minus())
            loghi=F(hi.ln().next_plus())
            arglo=ar.mul(exponent,loglo);arghi=ar.mul(exponent,loghi)
            dlo=_decimal(arglo,ROUND_FLOOR,precision).exp().next_minus()
            dhi=_decimal(arghi,ROUND_CEILING,precision).exp().next_plus()
            if not (dlo.is_finite() and dhi.is_finite()) or dlo<=0:
                raise DiagnosticRefusal('transcendental range or precision exhausted')
            return ar.check(F(dlo)),ar.check(F(dhi))
    except (DecimalException,OverflowError,ValueError) as e:
        raise DiagnosticRefusal('transcendental range or precision exhausted') from e


def _float_endpoint(x,upper):
    try:value=float(x)
    except OverflowError as e:raise DiagnosticRefusal('diagnostic float range exhausted') from e
    if not math.isfinite(value):raise DiagnosticRefusal('diagnostic float range exhausted')
    if (upper and _fraction(value)<x) or (not upper and _fraction(value)>x):
        value=float(np.nextafter(value,np.inf if upper else -np.inf))
    if not math.isfinite(value):raise DiagnosticRefusal('diagnostic float range exhausted')
    return value


def literal_objective_diagnostics(K,y,alpha,offset,lambd,q=1.,*,raw_slopes=None,max_n=16,max_bits=8192,max_operations=100000,precision=80):
    """Enclose objective and literal coefficient/bias gradient; certify dual.

    Requires a finite real n-by-n matrix, n in [2,16], signed labels containing
    both classes, finite coefficients/bias and positive lambda/q. The exact
    symmetric penalty must be PSD; singular range failure selects the distinct
    zero-slope witness. Limits are refusals, never eigenvalue truncations.
    Returned values are plain serialization-safe metadata and arrays of numbers.
    No kernel, fitted state, diagnostic already used by a solver, or input is
    mutated. This bounded explicit API does not run automatically during fit.
    """
    for value,name,lo,hi in ((max_n,'max_n',2,16),(max_bits,'max_bits',64,8192),(max_operations,'max_operations',1,100000),(precision,'precision',32,256)):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or not lo<=value<=hi:raise ValueError(name+' outside supported diagnostic budget')
    try:n=len(y)
    except TypeError as error:raise ValueError('labels require a bounded vector') from error
    if n<2 or n>max_n:raise DiagnosticRefusal('dimension budget or empty labels')
    y=_array(y,'y',(n,));K=_array(K,'K',(n,n));alpha=_array(alpha,'alpha',(n,))
    if not np.isin(y,[-1.,1.]).all() or len(np.unique(y))!=2:raise ValueError('both signed classes are required')
    offset=_scalar(offset,'offset');lambd=_scalar(lambd,'lambd',True);q=_scalar(q,'q',True)
    ar=_Arithmetic(int(max_bits),int(max_operations))
    A=[[ar.check(_fraction(x)) for x in row] for row in K]
    a=[ar.check(_fraction(x)) for x in alpha];b=ar.check(_fraction(offset));lam=ar.check(_fraction(lambd));qq=ar.check(_fraction(q));yy=[F(int(x)) for x in y]
    threshold=ar.div(qq,ar.add(qq,F(1)))
    S=[[ar.div(ar.add(A[i][j],A[j][i]),F(2)) for j in range(n)] for i in range(n)]
    # Exact PSD/range factorization; all pivots are retained, however small.
    L=[[F(i==j) for j in range(n)] for i in range(n)];D=[]
    for j in range(n):
        pivot=ar.add(S[j][j],-ar.total(ar.mul(ar.mul(L[j][k],L[j][k]),D[k]) for k in range(j)))
        if pivot<0:raise DiagnosticRefusal('exact indefinite penalty')
        D.append(pivot)
        for i in range(j+1,n):
            value=ar.add(S[i][j],-ar.total(ar.mul(ar.mul(L[i][k],L[j][k]),D[k]) for k in range(j)))
            if not pivot:
                if value:raise DiagnosticRefusal('exact indefinite zero-pivot penalty')
            else:L[i][j]=ar.div(value,pivot)
    scores=[ar.add(ar.total(ar.mul(A[i][j],a[j]) for j in range(n)),b) for i in range(n)]
    margins=[ar.mul(yy[i],scores[i]) for i in range(n)]
    loss=[];rho=[];branches=[]
    for m in margins:
        if m<=threshold:
            value=ar.add(F(1),-m);loss.append((value,value));rho.append((F(1),F(1)));branches.append('linear')
        else:
            ratio=ar.div(threshold,m);ll,lh=_power_interval(ratio,qq,ar,precision)
            loss.append((ar.div(ll,qq+1),ar.div(lh,qq+1)))
            rho.append(_power_interval(ratio,ar.add(qq,F(1)),ar,precision));branches.append('tail')
    penalty=ar.mul(lam,ar.total(ar.mul(ar.mul(a[i],A[i][j]),a[j]) for i in range(n) for j in range(n)))
    primal=(ar.add(ar.total(x[0] for x in loss)/n,penalty),ar.add(ar.total(x[1] for x in loss)/n,penalty))
    z=[(-yy[i]*r[1]/n,-yy[i]*r[0]/n) if yy[i]>0 else (-yy[i]*r[0]/n,-yy[i]*r[1]/n) for i,r in enumerate(rho)]
    gradient=[]
    for i in range(n):
        terms=[(ar.mul(A[j][i],z[j][0]),ar.mul(A[j][i],z[j][1])) if A[j][i]>=0 else (ar.mul(A[j][i],z[j][1]),ar.mul(A[j][i],z[j][0])) for j in range(n)]
        regularizer=ar.mul(lam,ar.total(ar.mul(ar.add(A[i][j],A[j][i]),a[j]) for j in range(n)))
        gradient.append((ar.add(ar.total(t[0] for t in terms),regularizer),ar.add(ar.total(t[1] for t in terms),regularizer)))
    bias=(ar.total(v[0] for v in z),ar.total(v[1] for v in z))
    if raw_slopes is None:raw=np.array([float(r[0]) for r in rho])
    else:
        raw=_array(raw_slopes,'raw_slopes',(n,))
    balanced=feasible_dual_slopes(raw,y)
    v=[ar.mul(yy[i],ar.check(_fraction(balanced[i]))) for i in range(n)]
    assert ar.total(v)==0
    d=[ar.total(ar.mul(A[j][i],v[j]) for j in range(n)) for i in range(n)]
    transformed=[]
    for i in range(n):transformed.append(ar.add(d[i],-ar.total(ar.mul(L[i][j],transformed[j]) for j in range(i))))
    if any(not pivot and value for pivot,value in zip(D,transformed)):
        dual=F(0);dual_kind='zero_slopes_range_failure'
    else:
        quadratic=ar.total(ar.div(ar.mul(value,value),pivot) for value,pivot in zip(transformed,D) if pivot)
        dual=ar.add(_conjugate_mean_lower(balanced,q),-ar.div(quadratic,ar.mul(F(4*n*n),lam)))
        dual_kind='literal_psd_range_certificate' if dual>=0 else 'zero_slopes_stronger'
        dual=max(F(0),dual)
    def endpoints(pair):return [_float_endpoint(pair[0],False),_float_endpoint(pair[1],True)]
    return {'schema':'literal-objective-diagnostic-v1','objective_interval':endpoints(primal),
      'coefficient_gradient_intervals':[endpoints(v) for v in gradient], 'bias_gradient_interval':endpoints(bias),
      'dual_lower_bound':_float_endpoint(dual,False), 'objective_gap_upper':_float_endpoint(ar.add(primal[1],-dual),True),
      'rank':sum(bool(d) for d in D),'dual_kind':dual_kind,'exact_class_balance':'0','margin_branches':branches,
      'operations':ar.operations,'operation_scope':'instrumented rational operations; excludes separately bounded slope and conjugate helper work',
      'caps':{'rows':max_n,'bits':max_bits,'operations':max_operations,'decimal_precision':precision},
      'mathematical_scope':'literal stored K score map; exact symmetric-part PSD penalty',
      'arithmetic_assumptions':'exact rational operations within budgets; correctly rounded Decimal ln/exp; finite binary64 input conversion',
      'changes_fit_or_acceptance':False}
