# Overview

This package implements Distance Weighted Discrimination (DWD) with a
scikit-learn-style interface for fitting, prediction, and cross-validation.
The methods follow Marron, Todd and Ahn (2007) and Wang and Zou (2018).

The package implements:

- Original linear DWD, solved with second-order cone programming (SOCP)
  through the optional CVXPY dependency (`dwd.socp_dwd.DWD`).
- Generalized linear DWD, solved with majorization-minimization (MM)
  (`dwd.gen_dwd.GenDWD`).
- Kernel generalized DWD, with checked full-kernel numerical backends
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

DWD 2.0.2 requires Python 3.11+, NumPy 1.26.3+, SciPy 1.11.4+,
scikit-learn 1.6.1+, joblib 1.6.0+ and threadpoolctl 3.7.0+. Download a
compatible wheel from the [2.0.2 release assets](https://github.com/cheng338/dwd/releases/tag/v2.0.2).
The native wheel for 64-bit Windows and standard CPython 3.12 contains three
compiled helpers for residuals, ordered reductions and feature expansions.

```shell
python -m pip install ./dwd-2.0.2-cp312-cp312-win_amd64.whl
```

The portable wheel supports Python 3.11+:

```shell
python -m pip install ./dwd-2.0.2-py3-none-any.whl
```

To install from source, run `python -m pip install .` in the root of the
extracted archive or checkout.
See the [project metadata](pyproject.toml) and the
[compiled-helper guide](docs/compiled_residual.md) for build prerequisites
and runtime eligibility.

The current validation policy targets Python 3.11 and 3.12 for complete suites.
Python 3.13 and later Python 3.x releases are supported without dedicated
complete-suite runs. The recorded 3.13 checks retain their historical scope.

Python 3.10 is no longer supported. Previously released companion wheels
remain available as historical artifacts and will not receive updates.

The `cp311-abi3` tag on earlier native wheels identifies a loading ABI.
Each numerical route also checks runtime eligibility. Release 1.3.17 qualified the compiled
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

**Native scoring in 2.0.0.** General dense accurate scoring remains eligible
on Windows x86-64. Linux x86-64 also admits the private reporting score path,
including final-stationarity `K*w`; a separate native entry handles eligible
joint-affine reporting. Qualified linear-feature reports can use these private
reporting routes too. They require GIL-enabled CPython 3.11 or later 3.x,
binary64 arithmetic and the existing runtime, input and numerical guards.
This does not enable either reporting entry for general Linux prediction.

Residual dispatch is separate: on CPython 3.12+ a built-in `math.sumprod` and
the scalar arithmetic guards can admit a compatible compiled residual helper
before scalar fallback on Linux or another platform. That scalar gate has no
OS or architecture restriction; the independent compiled-residual gate above
remains Windows x86-64. A present binary or an eligible runtime does not prove
that a particular fit executed native arithmetic or accepted its result.

Recorded numerical validation covers Python 3.11 through 3.13; forward
eligibility is not a claim of testing on later runtimes. Portable wheels can
use the guarded scalar route when available. Python 3.11 has no `math.sumprod`,
so a missing or declining compiled helper uses portable compensated arithmetic.
See the [runtime contract](docs/compiled_residual.md) for dispatch and fallback
details, including the current Linux reporting scope and its recorded
validation. These choices do not disable or select NumPy/SciPy BLAS acceleration.

To install from the release checkout instead:

```shell
python -m pip install .
```

**Source builds.** The residual and ordered-reduction helpers are
required by default. Qualified GIL-enabled x86-64 CPython 3.12 Windows/Linux
builds additionally require the feature-expansion helper. These mixed-ABI
builds use a platform-specific `cp312-cp312` wheel, not an `abi3` tag; other
native builds retain the prior stable-ABI helpers. Compiler/linker failure or
a missing/empty required output stops the build. Set `DWD_BUILD_ACCEL=0`
to build a portable wheel. The flag acts during source compilation. A wheel
already installed or taken from pip's cache retains the helpers it was built with.
When changing build mode, use a separate clean source/build tree and a fresh
source build with `--no-cache-dir`. See the [build details](docs/compiled_residual.md#building-and-installing).

The second helper is a private ordered-sum accelerator for the existing dense
named-linear L-BFGS gradient report. It activates automatically on qualified
Windows/Linux x86-64 CPython runtimes and compatible inputs; missing or declined
native execution retains the original checked sum. The separate expansion
helper reuses qualified splits for gradient tiles and eligible warm beta
calculations, while preserving exact arithmetic and the original fallback.
Its full-ABI interpreter version/build must match; unsupported builds keep
package support through fallback. Neither helper alters optimizer work or adds
a public selector. Wider tested fits
benefited, but ordinary and missing-binary fits can cost more. See the
[runtime contract](docs/compiled_residual.md#ordered-linear-feature-gradient-reduction)
and [measured scope](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting).

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
defaults for new kernel fits, as described below. Explicit scoring and residual-order policies
remain available; models fitted and saved with earlier versions retain their
recorded scoring behavior. This does not retain the retired legacy fitting
algorithm described in historical release notes.

## Companion package and citation

Choose a companion release whose declared DWD dependency matches the version
you install. [Ensemble DWD 0.4.1](https://github.com/cheng338/ensemble-dwd/releases/tag/v0.4.1)
requires exactly `dwd==2.0.2`. For the earlier pair, Ensemble DWD 0.4.0
requires exactly `dwd==2.0.1`.

The companion Ensemble DWD package implements the ensemble kernel DWD method
described in Chapter 5 of the following dissertation.

Cheng, Chang (2026). *Ensemble Kernel Distance-Weighted Discrimination and
Supervised Nonnegative Matrix Factorization for Classification* [Doctoral
dissertation, Purdue University]. Purdue University Graduate School.
[https://doi.org/10.25394/PGS.34548831.v1](https://doi.org/10.25394/PGS.34548831.v1)

## License

The bundled arithmetic adapted from CPython is covered by the complete
[Python license](dwd/CPYTHON-LICENSE.txt), identified as `Python-2.0.1` in SPDX.
The distribution metadata therefore declares `MIT AND Python-2.0.1` for the
combined contents. This applies to source distributions and both native and
portable wheels: portable wheels also include the adapted C sources. Each
component retains its license and attribution; this does not relicense the
original DWD code or the CPython-derived code.

# Kernel DWD

For practical reasons to select an optional numerical, resource or workflow
setting, see [choosing options](docs/choosing_options.md). Maintainers comparing
the retained private reference implementation should use the separate
[reference validation guide](docs/developer_reference.md).

Kernel DWD uses the corrected optimized implementation with explicit choices
of numerical backend, initialization, acceleration and stopping policy. The
repaired reference implementation is private validation machinery, not a public
estimator option. Different numerical policies can produce different models at
a finite iteration budget; see [migration](docs/kernel_dwd.md#removed-public-solver-controls).

```python
from sklearn.datasets import make_circles
from dwd.gen_kern_dwd import KernGDWD

X, y = make_circles(n_samples=120, noise=.12, factor=.5, random_state=1)
model = KernGDWD(lambd=.02, kernel='rbf', kernel_kws={'gamma': 1.0}).fit(X, y)
predictions = model.predict(X)
print(model.backend_, model.termination_reason_, model.n_iter_)
print(model.objective_tolerance_met_, model.converged_, model.rkhs_gradient_norm_)
```

New fitting, CV preparation and solver/step calls use corrected update algebra.
The public `implementation` and `solver_mode` selectors and kernel-only
`implicit_P` parameter have been removed. Remove those keys from constructor calls, `set_params`
requests and parameter grids; passing them now raises instead of selecting a
different algorithm. Linear `GenDWD` and `GenDWDCV` still support `implicit_P`.
See [API migration](docs/kernel_dwd.md#migration-from-legacy-fitting), including
the keyword-only boundaries that prevent old positional controls shifting
into another parameter.

The current defaults are `q=1`, `backend='auto'`,
`acceleration='auto'`, `residual_check_order='auto'`, `rbf_computation='standard'`,
`affine_computation='joint'`,
`exact_recovery='standard'`, `initialization='zero'`, `stopping='objective'`,
`obj_tol=1e-5`, and `max_iter=100`.
The cap applies per attempt. For eligible fits using an internally constructed
RBF kernel, a numerical failure can trigger one spectral restart with the same
initialization and objective; discarded work and total time are reported
separately. See the
[restart and stopping rules](docs/kernel_dwd.md#automatic-numerical-restart).
`GenDWD`, `GenDWDCV`, `KernGDWD` and `KernGDWDCV` now default explicitly to
`initialization='zero'`, preserving the previous default's zero starting state.
The `'auto'` initialization alias is no longer accepted: replace it with `'zero'`
in estimator settings and parameter grids. Explicit `'random'` initialization
and caller-supplied initial coefficients/intercepts remain supported. This
removes only the initialization alias; other controls' `'auto'` values retain
their existing behavior. The intercept remains unregularized. See the
[initialization details](docs/kernel_dwd.md#backends-and-initialization).
Automatic residual ordering uses adaptive checks on eligible optimized
Schur/Cholesky paths and preserves the applicable historical order elsewhere.
Explicit `residual_check_order='refinement_first'` remains available.
Automatic acceleration selects restarted MM for optimized Schur MM, including
checked spectral solves, and ordinary updates for L-BFGS.
Use `acceleration=None` to retain ordinary MM. The stopping default is unchanged.
The parameter names remain `lambd`, `q`, and `kernel_kws`.

Eligible new standard RBF fits selectively recompute cancellation-sensitive
distances using coordinate differences, independently of NumPy error-policy
selection for supported ordinary parameter representations. Nonordinary caller
policies remain observable during arithmetic. The Gaussian kernel and DWD objective
are unchanged; the [construction policy](docs/kernel_dwd.md#rbf-construction-and-query-consistency)
describes its scope, fitted-model behavior and limits.
For sensitive inputs, `KernGDWD(kernel='rbf', rbf_computation='direct')`
uses direct coordinate differences for every pair. It can change rounded
scores, fits and CV choices, and can cost substantially more.
`KernGDWDCV` accepts the same option. See the
[supported settings, benchmarks and precision limits](docs/kernel_dwd.md#optional-direct-rbf-computation).

The default `affine_computation='joint'` uses joint scoring for corrected
kernel fits: it evaluates uncertain products and the intercept together
before rounding. It preserves the expression `K @ alpha + b`,
the free intercept and the class tie rule. `KernGDWDCV` accepts the same option.
The former affine `'auto'` alias is removed; replace it with `'joint'` in
constructor settings and parameter grids. The previous default already used
joint scoring, so this spelling change adds no numerical work.
Compared with separate product-and-intercept evaluation, joint scoring can
change rounded scores, labels or validation choices and can take longer,
especially for difficult cancellation. Explicit `'standard'` retains the
separate product-and-intercept calculation; already-fitted legacy and other
serialized models keep their recorded scoring policy. See
[joint affine scoring](docs/kernel_dwd.md#optional-joint-affine-scoring).

Users of the removed reference option must choose their numerical policies
explicitly. `backend='spectral'`, `initialization='random'`, `acceleration=None`
and `residual_check_order='refinement_first'`, with the same integer seed,
retain its former automatic policy choices through the supported solver. This
is not a promise of identical coefficient trajectories or predictions. Removing
the keyword alone selects the existing optimized defaults, including zero
initialization and restarted MM, which can change finite-budget results.

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

Difficult checked spectral inverse actions have bounded alternative eigenbasis
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
budget. Choose optimality when a stationarity target is part of your acceptance
criteria, or when comparing solvers at a common achieved optimization tolerance.
It can require substantially more iterations and full-fit/CV time. Keep the
objective default for ordinary predictive work when its returned-model
diagnostics are adequate for your application; tighter optimization does not
guarantee better classification accuracy.
See [when to use optimality](docs/kernel_dwd.md#choosing-optimality-stopping)
for iteration-budget guidance, checked examples for linear/kernel estimators,
and the limits of CV refit diagnostics. The same guide covers validation
stopping, callbacks, diagnostics, batching, and numerical boundaries.

Version 1.3.13 also provides optional
[`residual_profile()`](docs/compiled_residual.md#optional-residual-profiling)
to distinguish compiled residual dispatch from scalar fallback during a fit.
It records observations separately from the estimator and does not change
the numerical checks or solver defaults.

The default `acceleration='auto'` selects restarted optimized Schur MM. It extrapolates decision
values and restarts momentum when necessary, using the same full kernel,
regularizer, and free intercept. The L-BFGS path does not accept
explicit `acceleration='restart'`. The automatic policy resolves solely from
the configured backend. Faster objective
convergence can reduce classification accuracy at a fixed update budget.
Compare acceleration and stopping through an external
validation procedure when predictive performance is the goal.

The constructor retains the requested policy; `requested_acceleration_` and
`acceleration_` report the fitted request and effective policy. Missing requested
acceleration settings use the current `auto` default for parameter inspection,
cloning and new fits. An explicitly stored `None` retains ordinary MM; existing
fitted predictions remain unchanged. Momentum keeps
one additional length-n float64 score vector between updates and uses temporary
length-n work vectors; it does not retain another Gram matrix.

# Explicit parameter search

The built-in CV wrappers use immutable default penalty/exponent tuples with
the same 50 combinations as before. Replace grids through the constructor or
`set_params`; explicit NumPy arrays remain supported. See
[default CV grids](docs/kernel_dwd.md#default-cv-grids-and-display) and the
separate [scikit-learn HTML display note](docs/dependency_display.md).

Keep preprocessing inside the CV pipeline. Gamma remains an entry in
`kernel_kws`, so tune whole dictionaries rather than `kernel_kws__gamma`:

```python
from joblib import effective_n_jobs, parallel_config
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

pipeline = make_pipeline(StandardScaler(), KernGDWD())
search = GridSearchCV(
    pipeline,
    {'kerngdwd__lambd': [.01, .1],
     'kerngdwd__kernel': ['rbf'],
     'kerngdwd__kernel_kws': [{'gamma': .1}, {'gamma': 1.0}]},
    cv=3, n_jobs=effective_n_jobs(None), scoring='accuracy', error_score='raise',
)
# Keep the selected model's timed refit serial. The search retains its CV worker count.
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    search.fit(X, y)
selected_model = search.best_estimator_
```

`KernGDWD.fit` does not split validation data or launch a search. The optional
`KernGDWDCV` wrapper and `run_cv` provide explicit standalone tuning with
fold-specific kernel caching. By default, CV caches the training kernel;
explicit spectral or L-BFGS CV also caches validated eigenpairs. Corrected cache comparison uses
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

For serial `KernMD` searches over wide float64 feature matrices, the explicit
[`fit_shared_fold_search` helper](docs/shared_fold_search.md) can reuse immutable
feature subsets across candidates. Exact contiguous row selections use private
views. On this serial route, scikit-learn scores the candidates and handles
refitting. The final model owns its feature data.

On the stock `GridSearchCV.fit` fallback in DWD `2.0.2`, the helper
prepares mapped NumPy features, targets and direct array-valued fit parameters
when multiple process workers are configured on entry. It copies inputs backed
by copy-on-write mappings and mapped views with a reversed non-singleton axis
or a contiguous memory order that differs from the backing map. These copies
apply to arrays of exact type `numpy.ndarray` or `numpy.memmap` whose dtype
contains no Python objects. If a setup callback will switch a serial search
to process workers, make owning copies of those affected inputs before calling
the helper.

For supported explicit fold indices and exact pandas DataFrame features, the
helper uses the workers selected for candidate evaluation, including after a
setup callback changes the worker count. The parent splitter and final refit
use the original DataFrame.
For explicit train/test selectors of exact type `pandas.Index` or `pandas.Series`,
standard joblib process workers receive the parent's row order or Boolean mask,
including private edits to supported mapped buffers.
Keep object-valued data in ordinary in-memory arrays for
process fitting. With the published
DWD 2.0.1 or a direct process-parallel `GridSearchCV.fit`, make an owning copy
of each affected view before fitting. For fold formats and custom scorers, see the
[process-parallel guidance](docs/shared_fold_search.md#process-parallel-fallback).
Wheel users can find `docs/shared_fold_search.md` in the matching source archive.
Measure the complete search. Ordinary or moderate-sized workloads may favor
ordinary search, and caching can add storage through the final refit. The helper
clears its private fold registry when it exits, after recording cache use.
Keep statistically chosen folds; ordinary search remains the general route,
and existing fit and CV defaults are unchanged.

# Linear classifiers and SOCP

`dwd.gen_dwd.GenDWD` provides generalized linear DWD with corrected update
algebra. It can stop based on objective change
or numerical optimality, or run for a fixed number of iterations. The correct
linear explicit-system path remains available through `implicit_P=False`;
`implicit_P=True` selects the default implicit path. Both are supported current
solvers. Removing the redundant mode selector does not remove either path.

Sparse input does not make either linear route sparse throughout: both prepare
a dense system with `n_features + 1` rows and columns. One float64 matrix alone
requires `8 * (n_features + 1)**2` bytes; copies, eigenvectors or factors and native
workspace add to that cost. Dense eigendecomposition or factorization has cubic
feature-dimension cost. Implicit CV preparation caches a dense eigensystem;
explicit `cv_init` validates inputs but skips that unused preparation. Explicit
fitting still forms and factors its dense system. See
[linear resource costs and alternatives](docs/linear_resources.md) before fitting
a design with many features.

When the represented MM penalty shift is zero (normally `lambd=0`), both
paths require the augmented design `[1, X]` to have numerically resolved full
column rank. They apply the same column-scaled SVD check, including when an
eigensystem is supplied or cached. Redundant feature/intercept directions are
refused; no ridge or minimum-norm solution is substituted. A positive penalty
that underflows to a zero shift follows the same rule.

The check uses a dense augmented copy and adds preparation work to each
zero-shift fit; positive-shift fits skip it. Subsequent spectral, factorization
and residual checks remain necessary, and passing the rank check does not
guarantee convergence. See the [measured cost and validation scope](VALIDATION.md#zero-shift-linear-rank-admission-in-local-source).

For a positive represented shift, the built-in dense float64 implicit Schur
path can recover automatically from two numerical preparation failures: a
nonpositive shifted augmented-Gram spectrum or an invalid free-intercept
correction denominator. The recovery solves the original-feature MM system
with bounded exact rational arithmetic. Each next coefficient and intercept
is verified as the nearest-even binary64 rounding of the exact solution for
the represented current coordinates and computed loss-gradient vector.
The penalty, free intercept and MM target are unchanged; no eigenvalue clipping,
added ridge or rank truncation is used.

This recovery is limited to 16 features and 65,536 feature entries, with finite
work, integer-width and modeled-workspace caps. Unsupported inputs or exhausted
limits still refuse safely. Existing successful paths, supplied-eigensystem
validation, the zero-shift rank check, custom numerical hooks
and stopping rules are preserved for supported new fits. This is not a general accuracy certificate
for already-accepted positive-spectrum solves, nor a promise of convergence
or arbitrary-query score accuracy. Exact recovery adds cost only when reached;
see the [contract, limits and measured costs](VALIDATION.md#bounded-exact-linear-recovery-in-local-source).

New linear fits promote features to float64 before solver arithmetic, including
gradients, objectives and custom hooks.
Ordinary binary64 rounding and the supported solver's numerical acceptance
checks remain in force.

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

See [choosing conic options](docs/choosing_options.md#optional-conic-models)
for when to use the distance-based `C='auto'` heuristic, how to assess its
training-fold cost, and when to supply solver-specific `solver_kws`.

New conic fits retain a serializable, re-solvable CVXPY problem and readable
solver diagnostics for the validated solver routes. Saving preserves known
data-only warm starts and leaves the live model unchanged; native solver
handles are rebuilt after loading. Native diagnostic record types can change,
and arbitrary custom objects can still prevent serialization. The SVM margin
expression also avoids a confirmed false-bound canonicalization problem while
preserving its mean-hinge/L1 objective and free intercept. See
[conic persistence, correctness and measured costs](docs/fitting_and_conic_persistence.md#conic-problems-and-persistence).

# Supported boundaries and tests

Training sample weights remain unsupported; weighted accuracy scoring remains
available. `KernMD(naive_bayes=True)` also
raises explicitly. The removed `implementation`, kernel `implicit_P` and `solver_mode` parameters
are not accepted. Dense kernel calculations may
reject indefinite or numerically unstable inputs; they do not silently add a
ridge or discard positive eigenvalues. Corrected features are promoted before
Gram construction. External kernels/eigenpairs must have adequate precision.
Private numerical thresholds are floating-point consistency estimates, not
universal error certificates. Private reference tests remain separate from the
public fitting API and from the retired incorrect legacy algorithm.

Owned default residual worker pools retry interrupted cleanup and preserve the
primary failure. After wake-up, an exactly owned unfinished startup may receive
a private 10 ms readiness wait. This limits that readiness attempt, not total
cleanup. Known registered workers can still be drained when whole completion
is unproved; completion waits can take longer. Completion relies on checked
CPython capabilities, and unresolved cleanup remains explicit. This does not
guarantee protection from arbitrary asynchronous interruption or change custom
executor lifecycles, numerical computations or acceptance checks. See the
[worker lifecycle contract](docs/compiled_residual.md#threads-and-memory).

For development checks, run the following from the source checkout in a separate
development environment. DWD uses a flat package layout, so this command can
import the checkout's `dwd/` even after pip installed a built package. These are
checkout tests, not qualification of an installed wheel or its native helpers.

```shell
python -m pip install ".[test]"
python -m pytest tests -o "python_files=test*.py" -ra
```

To qualify a specific wheel, use the [installed-wheel test procedure](docs/installed-tests.md):
install the exact artifact in a fresh environment, prepare tests outside the
checkout, verify installed origins and payloads, then run the detached suite.

For a separate environment without CVXPY, install `.[test-base]` and run the same
suite. Optional SOCP tests skip. The candidates for DWD 2.0.0 and Ensemble DWD
0.4.0 were tested in the [November 19, 2026 comparison](VALIDATION.md#november-19-2026-paired-release-comparison).
Passing numerical and API tests is not a universal accuracy or speed guarantee.
Historical notebooks and figures under `doc/` describe earlier implementations;
they have not been relabeled as new release results.

## Classification and input support

The base classifiers `GenDWD`, `GenDWDCV`, `KernGDWD`, `KernGDWDCV`, `KernMD`,
`dwd.socp_dwd.DWD` and `dwd.svm.SVM` require exactly two classes. Their
scikit-learn tags describe this binary contract. For multiclass classification,
use the separate `ensemble_dwd.OvOClassifier` around a supported binary
classifier; the wrapper retains multiclass support and requires dense feature
input. `EnsembleKernelDWD` and `EnsembleKernelSVM` themselves remain binary
classifiers with dense feature input.

| Estimator/configuration | Training input |
|---|---|
| `GenDWD`, `GenDWDCV`, SOCP `DWD`, conic `SVM` | Dense features or sparse features validated as CSR. `GenDWD` and `GenDWDCV` still prepare dense feature-square systems; see [linear resource costs](docs/linear_resources.md). The conic classifiers require the optional CVXPY dependency and a suitable solver. |
| `KernGDWD`, `KernGDWDCV` with `linear`, `poly`, `polynomial`, `rbf`, `laplacian`, `sigmoid` or `cosine` | Dense features or sparse features validated as CSR, with valid kernel arguments. Sparse features do not remove the full dense training-kernel cost. |
| `KernGDWD`, `KernGDWDCV` with a callable kernel | Dense or CSR features, subject to the callback's own input contract and the required kernel output. |
| `KernGDWD`, `KernGDWDCV` with `chi2` or `additive_chi2` | Dense features, subject to the kernel's numerical requirements. |
| `KernGDWD`, `KernGDWDCV` with `precomputed` | A dense square training Gram matrix. Sparse training Gram matrices are rejected. |
| `KernMD` | Dense training input, including a dense square Gram matrix when `kernel='precomputed'`. |

Where sparse features are supported, validation accepts CSR, CSC and COO
matrices and CSR arrays; conversion to CSR can allocate a copy. Callable kernels
receive the training features as their first argument and the supplied
training/query features as their second. A callback used with sparse features
must handle those sparse
arguments and return a usable dense training Gram matrix. Its output has one
row per training sample and one column per supplied sample. The callable sparse
tag describes this supported delegation; it does not guarantee that an
arbitrary callback accepts sparse input or produces a valid kernel.

For precomputed kernels, prediction input is query-by-training and may be dense
or sparse. This does not imply sparse training support. The pairwise tag lets
scikit-learn searches slice both axes for training folds and retain the training
columns for validation queries. `KernelScaler` is a separate dense, square-kernel
transformer; it does not provide sparse or rectangular query-kernel scaling.
Its default `transform(K, copy=True)` preserves the supplied array even if
scaling raises. Choose `copy=False` only when modifying compatible writable
storage is acceptable: a failed transform can leave the input and its aliases
changed or nonfinite, with no rollback. Dtype conversion can still allocate.
Use the default for read-only or internally overlapping views, or whenever the
original data must remain usable after failure. See the
[copy-mode contract](docs/kernel_dwd.md#kernelscaler-copy-modes).

At supported sparse input boundaries, signed and unsigned integer duplicates
are summed exactly by coordinate, then each sum is converted once to float64.
Linear prediction skips extra conversion for verified-unique storage and keeps
its usual dot-product conversion. Integer sparse precomputed queries use the
same duplicate rule; precomputed training remains dense-only.
Floating and Boolean input policies are unchanged.
Callable returned kernels and direct score helpers retain their separate
working-representation contracts. Large final integers can still round in
float64, and earlier overflow or rounding cannot be undone. See
[integer sparse coordinates](docs/kernel_dwd.md#integer-sparse-coordinate-inputs)
for compatibility and saved-model guidance.

Tags describe the configured estimator and are recomputed after parameter
changes. Inspect them through `sklearn.utils.get_tags(estimator)`. The metadata
does not convert input, bypass validation, change solver arithmetic or select
different defaults.

## Fitting weights and solver arguments

Training sample weights are unsupported. DWD 2.0.0 removes the
`sample_weight` argument from the six linear, kernel, CV and conic classifier
`fit` methods that previously accepted only `None`, and from their four
standalone solvers. Omit the argument entirely, including an explicit `None`;
direct calls that supply it now raise `TypeError`. Genuine fitting controls
remain available through their documented keyword arguments.

`score(X, y, sample_weight=weights)` still computes weighted accuracy; it does
not make fitting weighted. These classifiers cannot serve as AdaBoost base
estimators, which must support weighted fitting. See
[argument and metadata migration](docs/fitting_and_conic_persistence.md#fitting-arguments-and-metadata)
for positional-call boundaries and saved metadata requests. Cloning an old
object does not remove stale requests.

# Release history

DWD 2.0.2 corrects sparse integer and class-label handling and preserves file-backed inputs during parallel cross-validation.

If you are upgrading from 1.3.19, see the [migration guide](RELEASE_NOTES.md#upgrading-from-1319) for changes to fitting arguments, parameter grids, checkpoints, and Ensemble DWD compatibility.

See the [changelog](CHANGES.md) for changes by version and the [validation record](VALIDATION.md) for tested environments and results.
