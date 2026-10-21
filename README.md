# Overview

This package implements Distance Weighted Discrimination (DWD) with a
scikit-learn-style interface for fitting, prediction, and cross-validation.
The methods follow Marron, Todd and Ahn (2007) and Wang and Zou (2018).

The package implements:

- Original linear DWD, solved with second-order cone programming (SOCP)
  through the optional CVXPY dependency (`dwd.socp_dwd.DWD`).
- Generalized linear DWD, solved with majorization-minimization (MM)
  (`dwd.gen_dwd.GenDWD`).
- Kernel generalized DWD, with reference and optimized MM implementations
  (`dwd.gen_kern_dwd.KernGDWD`).

This is a fork of [slicersalt/dwd](https://github.com/slicersalt/dwd), originally
implemented by [Iain Carmichael](https://idc9.github.io/), with upstream maintenance
by David Allemang and [Kitware](https://kitware.com/). Subsequent development of
this fork is guided by [Chang Cheng](https://github.com/cheng338) and maintained
at [cheng338/dwd](https://github.com/cheng338/dwd). Original credits and the
[MIT license](LICENSE.txt) are retained.

This fork corrects update and numerical errors and adds an optimized kernel
implementation with optional compiled residual checks. It also adds numerical
validation and recovery, clears fitted state after failed fits, fixes
cross-validation issues, and repairs the examples. The DWD formulations and
unregularized intercepts follow the cited methods.

Marron, J. S., Todd, M. J., and Ahn, J. (2007).
[Distance-weighted discrimination](https://doi.org/10.1198/016214507000001120).
*Journal of the American Statistical Association*, 102(480), 1267-1271.

Wang, B., and Zou, H. (2018).
[Another look at distance-weighted discrimination](https://doi.org/10.1111/rssb.12244).
*Journal of the Royal Statistical Society: Series B*, 80(1), 177-198.

# Installation

Python 3.11+ and scikit-learn 1.6+ are required. The base package depends on
NumPy, SciPy, scikit-learn, and threadpoolctl. Download a compatible wheel from
the [1.3.18 GitHub release](https://github.com/cheng338/dwd/releases/tag/v1.3.18).
For conventional Windows AMD64 CPython 3.11 or later, install the native wheel:

```shell
python -m pip install ./dwd-1.3.18-cp311-abi3-win_amd64.whl
```

To omit the optional compiled helper, install the portable wheel (Python 3.11+):

```shell
python -m pip install ./dwd-1.3.18-py3-none-any.whl
```

The helper retains its numerical runtime guards; the ABI tag does not promise
acceleration on every compatible Python version. Local numerical validation
uses conventional CPython 3.12 with AOCL on Windows AMD64. Focused correctness
checks also cover CPython 3.11 and 3.13 with OpenBLAS wheels. These are not Ubuntu
or Intel-hardware performance results. Other platforms and free-threaded Python
remain outside those local checks.

The current validation policy targets Python 3.11 and 3.12 for complete suites.
Python 3.13 and later Python 3.x releases are supported without dedicated
complete-suite runs. The recorded 3.13 checks retain their historical scope.

Python 3.10 is no longer supported. Previously released companion wheels
remain available as historical artifacts and will not receive updates.

The `cp311-abi3` tag describes extension loading compatibility, not which
residual arithmetic DWD selects. Released version 1.3.17 qualified the compiled
residual and dense accurate-score paths on conventional Windows x86-64
CPython 3.11 through 3.13. Its separate built-in `math.sumprod` residual path
retained CPython 3.12 support and qualified Windows x86-64 CPython 3.13.

**Version 1.3.18:** runtime eligibility no longer has a Python 3 minor-version
ceiling. The independent compiled residual and dense-score routes accept
Windows x86-64 CPython 3.11 and later 3.x releases. The scalar residual route
accepts CPython 3.12 and later 3.x releases with a built-in `math.sumprod`, using
the same platform scope as the earlier 3.12 path. These routes require a
GIL-enabled build and the existing binary64 and arithmetic checks. The scalar
bound continues to assume CPython's extended/triple-length `sumprod` arithmetic.

Recorded numerical validation covers Python 3.11 through 3.13; forward
eligibility is not a claim of testing on later runtimes. Portable wheels can
use the guarded scalar route when available. Python 3.11 has no `math.sumprod`,
so a missing or declining compiled helper uses portable compensated arithmetic.
See the [runtime contract](docs/compiled_residual.md) for dispatch and fallback
details. These choices do not disable or select NumPy/SciPy BLAS acceleration.

To install from the release checkout instead:

```shell
python -m pip install .
```

The optional linear SOCP classifier requires CVXPY:

```shell
python -m pip install ".[socp]"
```

The extra does not switch the kernel classifier to CVXPY. This fork's release
artifacts are distributed through GitHub, not PyPI. Installing `dwd` by name
from PyPI may retrieve a different upstream release. See
[VALIDATION.md](VALIDATION.md) for the tested environments and evidence scope.

Version 1.3.11 includes the optional `rbf_computation='direct'`,
`affine_computation='joint'` and `exact_recovery='extended'` described below.
Those were opt-in when introduced; the 1.3.10 wheels do not provide them.
Version 1.3.17 makes automatic affine scoring and residual ordering the
defaults for new kernel fits, as described below. Explicit historical policies
remain available; models fitted and saved with earlier versions retain their
recorded scoring behavior.

# Kernel DWD

Kernel DWD provides a repaired reference implementation and an optimized
implementation. Both use the corrected DWD objective and update algebra, but
differ in numerical computation and default initialization; their fitted models
can differ with a finite iteration budget. See the
[release notes](RELEASE_NOTES.md) for compatibility changes.

```python
from sklearn.datasets import make_circles
from dwd.gen_kern_dwd import KernGDWD

X, y = make_circles(n_samples=120, noise=.12, factor=.5, random_state=1)
model = KernGDWD(lambd=.02, kernel='rbf', kernel_kws={'gamma': 1.0}).fit(X, y)
predictions = model.predict(X)
print(model.backend_, model.termination_reason_, model.n_iter_)
print(model.objective_tolerance_met_, model.converged_, model.rkhs_gradient_norm_)
```

The defaults are `implementation='optimized'`, `q=1`, `solver_mode='schur'`, `backend='auto'`,
`acceleration='auto'`, `residual_check_order='auto'`, `rbf_computation='standard'`,
`affine_computation='auto'`,
`exact_recovery='standard'`, `initialization='auto'`, `stopping='objective'`,
`obj_tol=1e-5`, and `max_iter=100`.
The cap applies per attempt. For eligible fits using an internally constructed
RBF kernel, a numerical failure can trigger one spectral restart with the same
initialization and objective; discarded work and total time are reported
separately. See the
[restart and stopping rules](docs/kernel_dwd.md#automatic-numerical-restart).
Auto initialization is zero for optimized fits. The intercept is unregularized.
Automatic residual ordering uses adaptive checks on eligible optimized
Schur/Cholesky paths and preserves the applicable historical order elsewhere.
Explicit `residual_check_order='refinement_first'` remains available.
Automatic acceleration selects restarted MM for optimized Schur MM, including
checked spectral solves, and ordinary updates for reference, legacy and L-BFGS
routes. Use `acceleration=None` to retain ordinary MM. The stopping default is unchanged.
The parameter names remain `lambd`, `q`, and `kernel_kws`.

Eligible new standard RBF fits selectively recompute cancellation-sensitive
distances using coordinate differences. The Gaussian kernel and DWD objective
are unchanged; the [construction policy](docs/kernel_dwd.md#rbf-construction-and-query-consistency)
describes its scope, fitted-model behavior and limits.
For sensitive inputs, `KernGDWD(kernel='rbf', rbf_computation='direct')`
uses direct coordinate differences for every pair. It can change rounded
scores, fits and CV choices, and can cost substantially more.
`KernGDWDCV` accepts the same option. See the
[supported settings, benchmarks and precision limits](docs/kernel_dwd.md#optional-direct-rbf-computation).

The default `affine_computation='auto'` uses joint scoring for corrected
kernel fits: it evaluates uncertain products and the intercept together
before rounding. It preserves the expression `K @ alpha + b`,
the free intercept and the class tie rule. `KernGDWDCV` accepts the same option.
It can change rounded scores, labels or validation choices and can be slower,
especially for difficult cancellation. Explicit `'standard'` and legacy fits
retain historical scoring; fitted and serialized models keep their policy. See
[joint affine scoring](docs/kernel_dwd.md#optional-joint-affine-scoring).

Choose `implementation='reference'` for the repaired Slicersalt algorithm: full
eigen-decomposition, coefficient MM updates, native Gaussian initialization
normalized to unit length, and the same default objective stopping rule. Its
computed basis must pass numerical consistency checks. Native EVD is tried first,
then EVR and EVX if necessary. It never substitutes Cholesky. Set `random_state`
for reproducible initialization.

The initial quadratic-form repair introduced in 1.2.1 is retained.
When the ordinary error bound cannot establish adequate precision, compensated
float64 accumulation bounds the error of the actual initial scalar quadratic;
the optimizer reuses the observed scores and quadratic. Coefficients, free
intercept, objective and default stopping rule remain unchanged; unsafe states
still raise. See [validation status](VALIDATION.md).

The optimized default solves the original free-intercept MM equations with
Cholesky, checks their residuals, and applies bounded refinement or a centered
factorization when needed. A low reciprocal-condition estimate alone no longer
rejects a fit. Centering changes how the linear system is solved; the fitted
kernel, penalty, and intercept remain unchanged. Unknown kernel families still
require PSD validation. Explicit spectral and L-BFGS backends, and checked eigenpair reuse,
remain available for optimized fits. None is a low-rank approximation.

Difficult reference inverse actions have bounded alternative eigenbasis
representations, including power-of-two equilibration. Every accepted correction
is checked against the original equations; no Cholesky substitution, added ridge
or positive-mode truncation occurs. If ordinary spectral refinement stagnates,
a bounded adjacent-float correction can refine the stored coefficients and free
intercept. It runs only on a failed solve and cannot bypass the residual,
coefficient-sum or RKHS accuracy checks. See the guide for its work limits.

If an ordinary MM update still fails numerically, a small exact-rank recovery
can try the same MM function in selected kernel-column coordinates. It first
certifies the entire stored float64 kernel exactly; a partial-rank approximation
is never accepted. This matters for singular kernels, where the intended function
can be representable even when an additional coefficient-sum convention is not.
The loss, regularization, free intercept and current update remain unchanged.
This exact computation runs only after a failed update, within fixed entry,
rank, and arithmetic budgets. For small difficult kernels,
`exact_recovery='extended'` raises the rank cap from 16 to 32 and the arithmetic
cap from 4096 to 8192 bits. Entry and operation caps, mathematical checks and
the standard default are unchanged. This can recover additional fits at extra
cost; it does not make ordinary fitting faster or establish convergence. See
[extended exact recovery](docs/kernel_dwd.md#optional-extended-exact-recovery).
An earlier characterization rejected nine extreme nearly constant synthetic
RBF fits; see [validation](VALIDATION.md) for the current replay outcome. This is
not a guarantee that every valid kernel will fit. See the numerical boundaries
in [the guide](docs/kernel_dwd.md).

Corrected kernel models now retain `prediction_precision_`: adaptive evaluation
uses ordinary query products only when their row error estimate passes, and
expanded-product accumulation handles uncertain rows. A model that needs fully
compensated evaluation keeps that policy for later queries, batching and
serialization. Validation and returned-state checks use the corresponding
policy, with the existing score and objective thresholds. No coefficient is
silently replaced to make a prediction check pass.

Supplied eigenpairs are checked against the actual matrix; malformed pairs raise
without replacement. Every computed native result is also validated before use.
If all native attempts fail, the error recommends updating or replacing
NumPy/SciPy and their BLAS/LAPACK runtime. The package does not start a numerical
worker process, change provider modes, or silently alter the kernel to obtain a fit.

Earlier local MKL failures motivated an automatic compatibility-mode workaround before 1.3.0.
The updated runtime passed the previously failing controls; 1.3.0 removed
that vendor-specific machinery while retaining the checks that reject invalid
eigenpairs. This is not a guarantee for every runtime or input. Dense memory and
eigen-preparation time still matter; EVD can need more workspace than other
symmetric drivers.
See [VALIDATION.md](VALIDATION.md) for the historical release checks and their scope.

The default stops on absolute successive-objective change. `converged_` instead
reports a checked numerical stationarity criterion on the returned model; inspect
it separately from `termination_reason_`. Numerical stopping can be requested
with `stopping='optimality', tol=1e-6`; `stopping='fixed'` requests a fixed MM
budget. A larger budget or a smaller numerical residual does not guarantee better
classification accuracy. See [the kernel guide](docs/kernel_dwd.md) for validation
stopping, callbacks, diagnostics, batching, and numerical boundaries.

Version 1.3.13 also provides optional
[`residual_profile()`](docs/compiled_residual.md#optional-residual-profiling)
to distinguish compiled residual dispatch from scalar fallback during a fit.
It records observations separately from the estimator and does not change
the numerical checks or solver defaults.

The default `acceleration='auto'` selects restarted optimized Schur MM. It extrapolates decision
values and restarts momentum when necessary, using the same full kernel,
regularizer, and free intercept. The reference and L-BFGS paths do not accept
explicit `acceleration='restart'`. The automatic policy resolves solely from
the configured implementation, solver mode and backend. Faster objective
convergence can reduce classification accuracy at a fixed update budget.
Compare acceleration and stopping through an external
validation procedure when predictive performance is the goal.

The constructor retains the requested policy; `requested_acceleration_` and
`acceleration_` report the fitted request and effective policy. Previously
serialized objects that lack this parameter retain ordinary MM. Momentum keeps
one additional length-n float64 score vector between updates and uses temporary
length-n work vectors; it does not retain another Gram matrix.

# Explicit parameter search

Keep preprocessing inside the CV pipeline. Gamma remains an entry in
`kernel_kws`, so tune whole dictionaries rather than `kernel_kws__gamma`:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), KernGDWD())
search = GridSearchCV(
    pipeline,
    {'kerngdwd__lambd': [.01, .1],
     'kerngdwd__kernel': ['rbf'],
     'kerngdwd__kernel_kws': [{'gamma': .1}, {'gamma': 1.0}]},
    cv=3, scoring='accuracy', error_score='raise',
).fit(X, y)
selected_model = search.best_estimator_
```

`KernGDWD.fit` does not split validation data or launch a search. The optional
`KernGDWDCV` wrapper and `run_cv` provide explicit standalone tuning with
fold-specific kernel caching. By default, optimized CV caches the training kernel;
reference CV also caches its validated eigenpairs. Corrected cache comparison uses
the same promoted float64 features as fitting, so float32 input can reuse a fold's
preparation across lambda values. Generic GridSearchCV clones estimators and does
not automatically share that cache. An external ensemble or support-selection
method should cross-validate its complete training procedure and final predictor.

For fixed-grid binary kernel DWD tuning with interruption recovery, use the
[resumable CV example](docs/resumable_cv.md). It saves completed candidate/fold
evaluations and optionally runs folds in parallel while retaining preparation
reuse within each fold. Its report identifies the best candidate among those
supplied, with exact ties, score gaps and numeric parameter positions within
matching conditional slices. It does not automatically extend or refine the
grid. The existing `run_cv` API and first-maximum selection rule are unchanged.

# Linear classifiers and SOCP

`dwd.gen_dwd.GenDWD` provides generalized linear DWD with corrected update
algebra by default. Set `solver_mode='legacy'` to use the historical algebra.
It can stop based on objective change or numerical optimality, or run for a
fixed number of iterations. The linear explicit-system path remains available
through `implicit_P=False`.

With `stopping='optimality'`, eligible built-in float64 paths reuse the current
gradient for the next MM step, avoiding a repeated loss derivative and
transposed feature product. The gradient is reused before the coefficients or
intercept change; final diagnostics are computed afresh at the returned model.
`GenDWDCV` inherits this behavior when optimality stopping is requested.
The default `stopping='objective'` and example stopping settings are unchanged;
there is no new public option. Custom hooks, other gradient dtypes and active
NumPy `call` or `log` error handlers keep the previous computation. This
optimization applies to linear DWD, with no direct benefit to kernel DWD or
ensemble-DWD. See [the checks and timing scope](VALIDATION.md#linear-gradient-reuse-in-local-source).

```python
from dwd.socp_dwd import DWD  # requires the socp extra

linear_socp = DWD(C='auto').fit(X, y)
print(linear_socp.C_, linear_socp.problem_.status)
```

SOCP uses a distinct constrained linear formulation and a penalty named `C`.
It preserves vectorized constraints, solver-status checks, and the unregularized
intercept. Its finite `optimal_inaccurate` results are accepted with CVXPY's
warning; that status does not certify a requested residual tolerance.

# Supported boundaries and tests

Sample weights remain explicitly unsupported. Kernel `implicit_P=False` and
`KernMD(naive_bayes=True)` also raise explicitly. Dense kernel calculations may
reject indefinite or numerically unstable inputs; they do not silently add a
ridge or discard positive eigenvalues. Corrected features are promoted before
Gram construction. External kernels/eigenpairs must have adequate precision.
Private numerical thresholds are floating-point consistency estimates, not
universal error certificates. Explicit `solver_mode='legacy'` retains known
historical algebra for compatibility; it is distinct from the repaired reference.

```shell
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

For a separate environment without CVXPY, install `.[test-base]` and run the same
suite; optional SOCP tests skip. Release results are in [VALIDATION.md](VALIDATION.md).
Passing numerical and API tests is not a universal accuracy or speed guarantee.
Historical notebooks and figures under `doc/` describe earlier implementations;
they have not been relabeled as new release results.

# Release history

Version 1.3.18, October 21, 2026. See the [current validation scope](VALIDATION.md).

## Changes in 1.3.18

Sparse coordinate canonicalization and selective RBF distance correction protect
the represented feature geometry. Guard scanning and eligible batched prediction
reuse avoid repeated work while retaining custom-helper fallbacks. Automatic
acceleration selects restarted MM on optimized Schur routes; ordinary MM remains
explicitly available. Factor solves remain the default and inverse action is
optional. Python 3.13 and later Python 3.x releases remain supported, with full
CI targeting 3.11 and 3.12. See the [release notes](RELEASE_NOTES.md) for numerical
scope, saved-model behavior and validation limits.

## Changes in 1.3.17

New kernel fits use automatic joint affine scoring and eligible adaptive
residual ordering, with explicit historical policies still available. Guarded
native arithmetic now covers conventional Windows x86-64 CPython 3.11 through
3.13. Internal buffer, coefficient, worker and RBF-norm reuse reduce repeated
work while retaining numerical guards and fallback paths.

Eligible restarted-MM failures can take one typed spectral retry from the
original initialization. New bounded literal-objective diagnostics are explicit,
and new valid legacy fits emit a warning. Fitted training data and kernel
parameters are owned by the model, post-fit prediction batching is honored, and
interruption and checkpoint cleanup preserve their original failures. Objectives,
free intercepts, stopping settings and numerical acceptance remain unchanged;
changed scoring and check order can change finite-iteration results. See the
[1.3.17 release notes](docs/history/RELEASE_NOTES-1.3.17.md) for all changes and validation limits.

## Changes in 1.3.15

Input validation now rejects complex starting intercepts and complex values
hidden in object arrays before conversion or optimization. The resumable CV
example preserves existing residual-check counts and reports exact ties, score
gaps and parameter positions within the supplied grid. Real-input solver
arithmetic, numerical tolerances, defaults and first-maximum candidate selection
are unchanged. See the [1.3.15 release notes](docs/history/RELEASE_NOTES-1.3.15.md)
for scope and checkpoint compatibility.

## Changes in 1.3.14

Optional adaptive residual checking now supports restart acceleration in
optimized Cholesky kernel MM, preserving the acceptance checks and default
ordering. The resumable CV example adds optional native-thread and memory
planning controls while retaining one worker and one native thread by default.
Python 3.10 companion maintenance has ended; main continues to require Python
3.11+. See the [release notes](docs/history/RELEASE_NOTES-1.3.14.md) for scope and compatibility.

## Changes in 1.3.13

Named accuracy scoring now compares exact, equally weighted fold proportions,
preserving the first candidate at a mathematical tie. The resumable example
records those counts and compact solver diagnostics; earlier checkpoints need
a new run directory. Optional residual profiling reports dispatch routes and
checking work without changing estimator mathematics or defaults. See the
[release notes](docs/history/RELEASE_NOTES-1.3.13.md) for scope and compatibility details.

## Changes in 1.3.12

Eligible dense accurate-score tiles use the optional C extension while
preserving the previous high/low product and summation order. This targets
prediction models requiring substantial compensated scoring; ordinarily
accepted dot products retain their existing path. The shared helper can
also serve fitting and validation scores. Unsupported runtimes, custom
hooks and exceptional states retain Python evaluation. See the
[implementation scope](docs/compiled_residual.md#dense-accurate-scores) and
[validation](VALIDATION.md).

## Changes in 1.3.11

This release adds optional direct RBF computation, joint affine scoring and
extended exact recovery, together with objective-overflow recovery, eligible
linear-gradient reuse and earlier release of prediction kernel blocks. The
resumable example records accelerator identity and permits an explicit final
thread limit while retaining one thread per native pool in each CV worker.
These changes preserve the DWD methodology; optional numerical policies can change rounded results and cost
more. See the [changelog](CHANGES.md#1311--2026-10-06) for individual changes.


## Changes in 1.3.10

Dense accurate scoring reuses coefficient preparation and evaluates suitable rows
in bounded groups, retaining the existing arithmetic and precision choices.
A separate resumable kernel DWD tuning example saves verified completed folds and
always performs a fresh final fit. The existing `run_cv` API, objective, free
intercept and solver defaults are unchanged.

Performance gains are specific to affected prediction workloads. Follow-up tests
identified concurrent work in the original full-kernel timing comparison and
found no persistent package regression. The validation retains the historical
measurements and an extreme-input training-query compatibility qualification
for the new example. See the [validation scope](docs/validation-1.3.10.md) and
[example guide](docs/resumable_cv.md).

## Changes in 1.3.9

The native residual evaluator reuses a double-length product while preserving
separate accumulators, signed-zero behavior and acceptance bounds. Windows AMD64
Zig builds explicitly target baseline x86-64. API fixes cover failed scaler and
kernel initialization, extreme scaling, complex initial coefficients, nonfinite
linear predictions and candidate-specific precomputed-kernel CV slicing.
Historical notebooks, figures and source-distribution fixtures are repaired.

`residual_check_order='adaptive'` is an opt-in numerical policy for eligible
optimized, unaccelerated Cholesky fits. The default remains `'refinement_first'`.
The objective, free intercept and stopping settings are unchanged; optional
ordering can change finite-iteration results and is not a convergence guarantee.
See the [release notes](RELEASE_NOTES.md) and [detailed scope](docs/candidate_changes.md).

## Changes in 1.3.8

Ordinary residual acceptance now accounts for floating-point evaluation error.
Final dual diagnostics use explicitly feasible weights and conservative bounds.
The objective, unregularized intercept and stopping defaults are unchanged.
See the [change-by-change validation](docs/validation-1.3.8.md), including
independent numerical checks, MNIST agreement and measured runtime cost.

## Changes in 1.3.7

Compensated residual checks reduce Python overhead by vectorizing the existing
outward-rounded error bounds and using the optional compiled row evaluator for
kernels with at least eight rows. Kernels below 2,048 rows use one worker without
repeated thread-pool inspection. The arithmetic guards, acceptance thresholds,
objective, unregularized intercept and stopping settings remain unchanged.
See the [release notes](RELEASE_NOTES.md).

## Changes in 1.3.6

Internally generated RBF kernels recover from roundoff-induced asymmetry using
consistent training and prediction construction. Supplied kernels and custom
kernel outputs retain strict symmetry validation.

Eligible optimized automatic MM fits can restart once through the spectral
backend after constrained numerical recovery is exhausted. The original kernel,
initialization, objective, unregularized intercept and stopping settings remain;
the iteration cap applies per attempt, with discarded work reported separately.
See the [kernel guide](docs/kernel_dwd.md) and [release notes](RELEASE_NOTES.md).

## Changes in 1.3.5

Failed fits now clear learned coefficients, class labels and other fitted state,
including failures during a refit. Calling prediction after a failed fit raises
`NotFittedError`; fitting valid data again restores normal use. This applies to
linear and kernel DWD, their CV wrappers, the optional SOCP classifiers and
kernel mean difference. Validated private precomputation caches remain reusable.
See [the fitted-state contract](docs/failed_refit_state.md).

A bounded intercept refinement and lazy anchor-coordinate LU recovery resolve
reproduced MNIST ensemble fit failures at very small regularization and RBF
coefficients. The original equation, coefficient-sum and RKHS acceptance checks
remain unchanged. See the [intercept explanation](docs/intercept_midpoint_refinement.md),
[anchor recovery equations](docs/anchor_linear_system.md), and
[release notes](RELEASE_NOTES.md) for the validation scope and remaining limits.

## Changes in 1.3.4

An optional compiled residual checker avoids repeated Python list conversion
and evaluates independent kernel rows in parallel. It preserves the existing
compensated arithmetic, numerical acceptance bounds, objective, free intercept,
and stopping settings. It respects the active BLAS thread limit, including
one-thread GridSearchCV workers, without copying the dense kernel per worker.

On the saved 12,089-row MNIST 2-versus-3 configuration, a fresh 100-update fit
took 47.12 seconds, compared with the previous 377.39-second median. Its fitted
coefficients, intercept, objective history, preprocessing and prediction probes
were bitwise identical to the saved model. This is a scoped workload result;
ordinary fits that do not need compensated checks may see little change.

The accelerated wheel contains a small optional C extension with no NumPy C API.
Source builds use a local compiler when available; the existing Python checker
remains available when compilation or loading is unavailable. That release
qualified the native screen on CPython 3.12; current eligibility is described in
[build and runtime details](docs/compiled_residual.md), [release notes](RELEASE_NOTES.md),
and [validation](VALIDATION.md). The previous 1.3.3 repair remains intact.
