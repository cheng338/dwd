"""Validated symmetric eigendecomposition, independent of DWD objectives.

This module never clips a signed eigenvalue, changes the input matrix, adds a
ridge, or truncates rank. Computed failures try native EVD, EVR, then EVX. Supplied
eigenpairs are validated and either returned in their original order or rejected.
Large-matrix checks use O(n**2) work/storage with a bounded number of probes;
they are numerical consistency checks, not interval-arithmetic certificates.
"""
import numpy as np
from scipy.linalg import eigh, LinAlgError


class EigenDecompositionError(FloatingPointError):
    """Every configured computed decomposition failed numerical validation."""
    def __init__(self, message, info):
        super().__init__(message)
        self.info = info
        self.attempts = info['attempts']


def _matrix_statistics(A):
    if np.iscomplexobj(A):
        raise ValueError('The eigendecomposition matrix must be real symmetric.')
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not len(A):
        raise ValueError('The eigendecomposition matrix must be nonempty and square.')
    n = len(A)
    scale = 0.
    for start in range(0, n, 128):
        block = A[start:start+128]
        if not np.isfinite(block).all():
            raise ValueError('The eigendecomposition matrix must contain finite values.')
        scale = max(scale, float(np.max(np.abs(block))))
    scale = scale if scale else 1.
    diagonal = np.diag(A) / scale
    symmetry, norm_inf, frobenius_squared = 0., 0., 0.
    lower, upper = float('inf'), -float('inf')
    for start in range(0, n, 128):
        block = A[start:start+128] / scale
        row_abs = np.sum(np.abs(block), axis=1)
        d = diagonal[start:start+len(block)]
        symmetry = max(symmetry, float(np.max(np.abs(block-A[:, start:start+len(block)].T/scale))))
        norm_inf = max(norm_inf, float(row_abs.max()))
        frobenius_squared += float(np.einsum('ij,ij->', block, block))
        lower = min(lower, float(np.min(d-row_abs+np.abs(d))))
        upper = max(upper, float(np.max(d+row_abs-np.abs(d))))
    eps = np.finfo(float).eps
    if symmetry > 100*eps:
        raise ValueError('The eigendecomposition matrix must be symmetric.')
    return A, dict(n=n, matrix_scale=scale, symmetry_error_scaled=symmetry,
                   norm_inf_scaled=norm_inf, frobenius_squared_scaled=frobenius_squared,
                   trace_scaled=float(diagonal.sum()), absolute_diagonal_sum_scaled=float(np.abs(diagonal).sum()),
                   diagonal_min_scaled=float(diagonal.min()), diagonal_max_scaled=float(diagonal.max()),
                   gershgorin_lower_scaled=lower, gershgorin_upper_scaled=upper)


def _scaled_product(A, vectors, scale):
    """Avoid a second full dense matrix when scale normalization is needed."""
    result = np.empty((len(A), vectors.shape[1]), dtype=float)
    for start in range(0, len(A), 128):
        result[start:start+128] = (A[start:start+128]/scale) @ vectors
    return result


def _validate_values(values, statistics):
    n = statistics['n']
    issues, details = [], {}
    if np.iscomplexobj(values) or np.shape(values) != (n,):
        return ['eigenvalues must be a real vector of length n'], details
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return ['eigenvalues contain nonfinite values'], details
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        scaled = values/statistics['matrix_scale']
        squared_sum = float(scaled @ scaled)
    if not np.isfinite(scaled).all() or not np.isfinite(squared_sum):
        return ['eigenvalue magnitudes are inconsistent with the matrix scale'], details
    tolerance = 128*np.finfo(float).eps*max(1, n)
    trace_scale = max(1., float(np.abs(scaled).sum()), statistics['absolute_diagonal_sum_scaled'])
    trace_error = abs(float(scaled.sum())-statistics['trace_scaled'])/trace_scale
    frobenius_error = abs(squared_sum-statistics['frobenius_squared_scaled'])/max(1.,statistics['frobenius_squared_scaled'])
    bound_tolerance = tolerance*max(1.,statistics['norm_inf_scaled'])
    if trace_error > tolerance:
        issues.append('eigenvalue trace does not match matrix trace')
    if frobenius_error > tolerance:
        issues.append('squared eigenvalues do not match matrix Frobenius norm')
    if (scaled.min() < statistics['gershgorin_lower_scaled']-bound_tolerance or
            scaled.max() > statistics['gershgorin_upper_scaled']+bound_tolerance):
        issues.append('eigenvalues exceed matrix spectral bounds')
    if (scaled.min() > statistics['diagonal_min_scaled']+bound_tolerance or
            scaled.max() < statistics['diagonal_max_scaled']-bound_tolerance):
        issues.append('eigenvalue range does not enclose diagonal Rayleigh quotients')
    details.update(trace_relative_error=float(trace_error), frobenius_squared_relative_error=float(frobenius_error),
                   spectrum_invariant_tolerance=float(tolerance),
                   min_eigenvalue=float(values.min()),max_eigenvalue=float(values.max()))
    return issues, details


def _validate_pairs(A, values, vectors, statistics):
    issues, details = _validate_values(values, statistics)
    n = len(A)
    if np.iscomplexobj(vectors) or np.shape(vectors) != (n,n):
        return issues+['eigenvectors must be a real n by n matrix'], details
    vectors = np.asarray(vectors, dtype=float)
    if not np.isfinite(vectors).all():
        return issues+['eigenvectors contain nonfinite values'], details
    if np.shape(values) != (n,) or np.iscomplexobj(values) or not np.isfinite(values).all():
        return issues, details
    # All columns are checked. A few selected eigenpairs missed the observed
    # EVR failure, whose defective columns were not at spectral extremes.
    squared_norms = np.zeros(n)
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0,n,128):
            block = vectors[start:start+128]
            squared_norms += np.einsum('ij,ij->j',block,block)
    if not np.isfinite(squared_norms).all():
        return issues+['eigenvector column norms are nonfinite'], details
    eps = np.finfo(float).eps
    orth_tolerance = 64*eps*max(1,n)
    norm_error = np.abs(squared_norms-1.)
    details.update(column_squared_norm_max_error=float(norm_error.max()),
                   column_norm_tolerance=float(orth_tolerance),
                   norm_defective_columns=int(np.count_nonzero(norm_error>orth_tolerance)))
    if norm_error.max() > orth_tolerance:
        issues.append('eigenvector columns are not normalized')
    # Extreme malformed vectors need not enter products that could overflow.
    if squared_norms.max() > 4. or squared_norms.min() < .25:
        return issues, details
    if n <= 128:
        probes = np.eye(n)
        selected = list(range(n))
        kind = 'full_small_matrix'
    else:
        strongest = int(np.argmax(np.abs(values)))
        suspect = np.argsort(norm_error)[-3:]
        selected = sorted(set([strongest,int(np.argmin(values)),n//2])|set(map(int,suspect)))
        rng = np.random.default_rng(104729+n)
        random_probes = (2*rng.integers(0,2,size=(n,4))-1)/np.sqrt(n)
        selected_probes = np.zeros((n,len(selected)))
        selected_probes[selected,np.arange(len(selected))] = 1.
        probes = np.column_stack((random_probes,selected_probes))
        kind = 'all_column_norms_and_deterministic_actions'
    mapped = vectors@probes
    orth_error = vectors.T@mapped-probes
    scaled_values = np.asarray(values,float)/statistics['matrix_scale']
    eigen_error = _scaled_product(A,mapped,statistics['matrix_scale'])-vectors@(scaled_values[:,None]*probes)
    orth_max = float(np.max(np.linalg.norm(orth_error,axis=0)))
    eigen_max = float(np.max(np.linalg.norm(eigen_error,axis=0)))
    eigen_tolerance = 64*eps*np.sqrt(max(1,n))*max(1.,statistics['norm_inf_scaled'])
    if not np.isfinite(orth_max) or orth_max > orth_tolerance:
        issues.append('eigenvector orthogonality action failed')
    if not np.isfinite(eigen_max) or eigen_max > eigen_tolerance:
        issues.append('eigen-equation action does not match the matrix')
    details.update(validation_kind=kind,probe_count=int(probes.shape[1]),targeted_columns=selected,
                   orthogonality_action_max_l2=orth_max,orthogonality_action_tolerance=float(orth_tolerance),
                   eigen_equation_action_max_l2_scaled=eigen_max,eigen_equation_action_tolerance_scaled=float(eigen_tolerance),
                   eigenvectors_checked=True,eigenvectors_validated=not issues)
    return issues, details


def validated_eigh(A, *, eigvals_only=False, supplied=None, return_info=False,
                   drivers=('evd','evr','evx')):
    """Return validated ``values, vectors`` (or only values) like SciPy eigh.

    ``supplied=(vectors, values)`` follows the package's existing public cache
    convention. Supplied order is preserved and invalid supplied inputs raise
    ValueError without recomputation. Computed results are ascending; every
    returned target result is checked, and failed native attempts are retained
    in diagnostics. This helper never starts a child process or changes BLAS,
    LAPACK, MKL, environment, or thread settings. If every native attempt fails,
    update or replace the numerical runtime rather than use an invalid basis.
    With ``return_info=True``
    return ``(result, info)`` where result is the ordinary return value. Private
    ``drivers`` is a nonempty tuple of distinct supported drivers in retry order.

    Eigenvalues-only calls check spectral bounds, trace and squared-spectrum
    invariants, but cannot certify eigenvectors/correspondence without vectors.
    This distinction is explicit in info. No PSD or regularization policy is
    imposed here; even valid negative and zero eigenvalues are retained.
    """
    if (not isinstance(drivers,tuple) or not drivers or
            any(driver not in ('evd','evr','evx','ev') for driver in drivers) or
            len(set(drivers)) != len(drivers)):
        raise ValueError("drivers must be a nonempty tuple of distinct 'evd', 'evr', 'evx', 'ev' values.")
    A, statistics = _matrix_statistics(A)
    info = dict(attempts=[],accepted_driver=None,supplied=supplied is not None,
                eigvals_only=bool(eigvals_only),matrix_shape=list(A.shape),
                matrix_scale=statistics['matrix_scale'],
                configured_drivers=list(drivers),validation_is_numerical_not_formal=True)
    if supplied is not None:
        if not isinstance(supplied,(tuple,list)) or len(supplied)!=2:
            raise ValueError('Supplied eigenpairs must be (eigenvectors, eigenvalues).')
        vectors,values = supplied
        issues,details = _validate_pairs(A,values,vectors,statistics)
        info['attempts'].append(dict(driver='supplied',accepted=not issues,issues=issues,**details))
        if issues:
            error = ValueError('Supplied eigenpairs failed validation: '+'; '.join(issues))
            error.info = info
            raise error
        values,vectors = np.asarray(values,dtype=float),np.asarray(vectors,dtype=float)
        info['accepted_driver']='supplied'
        result=values if eigvals_only else (values,vectors)
        return (result,info) if return_info else result
    for driver in drivers:
        # Do not retain a rejected full basis while preparing its replacement.
        result = values = vectors = None
        try:
            result = eigh(A,eigvals_only=eigvals_only,driver=driver,check_finite=False,overwrite_a=False)
            if eigvals_only:
                values=result
                issues,details=_validate_values(values,statistics)
                details.update(validation_kind='spectral_invariants_only',eigenvectors_validated=False)
            else:
                values,vectors=result
                issues,details=_validate_pairs(A,values,vectors,statistics)
        except (LinAlgError,FloatingPointError,ValueError) as exc:
            attempt = dict(driver=driver,accepted=False,
                           issues=[f'{type(exc).__name__}: {exc}'])
            if getattr(exc, 'info', None) is not None:
                attempt['recovery'] = exc.info
            info['attempts'].append(attempt)
            continue
        attempt = dict(driver=driver,accepted=not issues,issues=issues,**details)
        info['attempts'].append(attempt)
        if issues:
            continue
        order=np.argsort(values,kind='stable')
        if not np.array_equal(order,np.arange(len(A))):
            values=np.asarray(values)[order]
            if not eigvals_only:
                vectors=np.asfortranarray(np.asarray(vectors)[:,order])
        info['accepted_driver']=driver
        result=np.asarray(values) if eigvals_only else (np.asarray(values),np.asarray(vectors))
        return (result,info) if return_info else result
    summary=' | '.join(f"{attempt['driver']}: {', '.join(attempt['issues'])}" for attempt in info['attempts'])
    raise EigenDecompositionError(
        'All native symmetric eigendecomposition attempts failed validation: '+summary+
        '. Update or replace NumPy/SciPy and their BLAS/LAPACK runtime, then retry. '
        'No validated eigenbasis was available; the input matrix and objective were not altered.',info)
