> Historical 1.3.1 record. The counts, artifact hashes and measurements below
> belong to 1.3.1; they do not assert 1.3.2 validation.

# Release 1.3.1

**Local source and actual-wheel validation passed:** 318 tests each; with CVXPY
blocked, 312 pass and six optional tests skip. Actual-wheel GridSearchCV, seven
documentation examples and twelve original-script interface groups pass. The
30-fit public stress replay returns 21 independently checked models and retains
nine documented extreme synthetic rejections. Historical 1.3.0 artifacts and
comparisons remain unchanged. See [VALIDATION.md](VALIDATION-1.3.1.md) for scope.

## Numerical repair

Difficult reference inverse actions now have bounded alternative full-eigenbasis
representations, including power-of-two equilibration and fresh validated native
inverse preconditioners. The reference still uses coefficient/eigenbasis MM;
ordinary optimized fitting still uses Cholesky. Original-equation checks,
positive kernel directions, regularization, free intercept and input matrix are
preserved. A reference recovery never substitutes Cholesky.

If an ordinary MM proposal remains inaccurate, a lazy exact-rank path can retry
the same intended function. It first proves the entire stored float64 identity
`K = C B_inverse C.T`, then evaluates the same MM update in selected original
kernel-column coordinates. This handles some singular kernels whose intended
function is representable although the extra coefficient-sum gauge of an
ordinary inverse route is not. No check is simply waived, no partial factor
becomes a model and no positive eigenmode is truncated. Exact arithmetic checks
the actual exported coefficients' function, decision and free-intercept errors.

Certification and each action have explicit entry, rank, operation and bit caps;
exhaustion makes this recovery inapplicable. The ordinary healthy path does no
exact factorization. Only float64 coefficients and small diagnostics survive in
the estimator. Generalized q, optional restarted MM, all existing stopping rules
and observer semantics are retained. Recovery actions inside rejected momentum
proposals are distinguished from accepted updates and from the representation
of a validation-restored model.

The actual 1.3.1 wheel returned 21 of 30 checked public models
on preserved extreme synthetic kernels, including all 18 exact rank-five cases.
Nine extreme nearly constant RBF fits still rejected because this bounded
complete certificate exceeds its rank budget. This does not establish that their
DWD functions are impossible. These are synthetic cases, not failed MNIST fits;
the final wheel preserves every outcome of the preceding candidate replay.

## Prediction and model state

Corrected kernel models use adaptive original-kernel products, compensating rows
whose conservative error estimates fail, or retain fully compensated evaluation
when required by reconstruction. Dense/CSR queries, batching and serialized models
follow the stored `prediction_precision_` policy. Validation and returned-state
checks use the same policy. An objective-only mismatch receives one accurate
reevaluation under the unchanged thresholds; an actually inconsistent state
still raises before exposure. The best validation model restores coefficients,
intercept, precision and representation metadata together.

KernGDWD invalidates its learned fitted state before a refit attempt. This is a
kernel-estimator repair, not a package-wide atomic-refit claim about unchanged
linear/SOCP/CV estimators. Generic numerical checks still assume supported
float64 arithmetic and fail explicitly for nonfinite or unresolved states.

## Removing repeated work

Two accepted changes remove repeated work: reuse the objective already checked
for the immediately accepted MM state; and use a conservative dense-row bound
before the existing fine prediction screen. The cheap bound can accept only rows
the fine screen would accept under their shared floating-point assumptions.
After one block cannot be fully shortcut, the remainder of that prediction call
uses the exact original fine path. CSR is unchanged. No learned threshold,
kernel-dependent tuning or numerical tolerance relaxation is involved.

The fixed development check ran 24 fresh processes: both implementations,
range-recovery baseline versus these two changes, two prespecified lambda values
and three same-seed repetitions. It used 3,000 official-training fit rows and
1,000 disjoint development rows, gamma .001, native initialization, four BLAS
threads and the unchanged objective 1e-5 / cap100 defaults. Complete Pipeline.fit
and Pipeline.predict were timed separately, including their standardization.

All 12 paired NPZ archives had exactly identical NPY member bytes, including
coefficients, intercept, objective histories, public scores, predictions, scaler,
dtype and row IDs. All fits reached the 100-update cap without claiming numerical
convergence. Complete-prediction medians fell by 3.4–7.2 ms (7.0–14.1%) across
the four fixed implementation/lambda cells. Whole-fit medians changed from a
0.2% increase to a 2.5% decrease, with mixed paired fit differences; this does not
establish a large training speedup. No classifier accuracy changed.

These are candidate-source measurements; verified production bytes match the
final release except its version literal. They do not compare against raw Slicersalt or sealed 1.3.0,
and they involve no CV, new parameter selection or official-test evaluation.
Initial saved-query component profiles and small synthetic objective-call timings
remain separate evidence, not a universal prediction or fitting claim. Final
source/wheel numerical and interface acceptance passed independently.

## Preserved defaults and boundaries

The defaults remain optimized ordinary MM, q=1, zero optimized/native normalized
Gaussian reference initialization, objective difference below 1e-5 and at most
100 updates. The names `lambd`, `q`, `kernel_kws`, the package/dissertation lambda
conversion and free intercept are unchanged. Numerical convergence, objective
stopping and predictive validation remain distinct. Optional acceleration stays
off; L-BFGS is optional. Explicit legacy reproduces its documented historical
algebra and is distinct from the repaired reference.

Generic validated native eigensolvers remain; removed MKL-specific subprocess
and provider-mode workarounds are not reintroduced. No runtime installation or
global provider changes are performed by the package. Base NumPy/SciPy/sklearn
and optional linear CVXPY/SOCP dependencies, earlier loss/MM/API/CV/cache fixes,
unsupported-weight and kernel implicit_P=False boundaries, MIT licensing and
Iain Carmichael / Kitware / upstream-fork credits are retained.
