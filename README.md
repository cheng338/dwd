# Distance Weighted Discrimination 1.3.5

`dwd` provides linear and kernel Distance Weighted Discrimination classifiers
with sklearn-style fitting, prediction, and cross-validation. One package now
offers a repaired reference kernel implementation and an optimized implementation.
Both use the corrected DWD objective and update algebra. They differ in numerical
computation and default initialization, so finite-budget fitted models can differ.
Read [release notes](RELEASE_NOTES.md) when upgrading from upstream 1.0.5,
the audit1 fork, or the 1.1.x releases.

This fork is led and maintained by [Chang Cheng](https://github.com/cheng338)
at [cheng338/dwd](https://github.com/cheng338/dwd), building on his earlier
development of the fork. His contributions include the research direction,
methodological requirements, package design, and experimental design and review.
Codex assisted with code auditing, implementation, and testing under his direction.

The project builds on [slicersalt/dwd](https://github.com/slicersalt/dwd), originally
implemented by [Iain Carmichael](https://idc9.github.io/), with upstream maintenance
by David Allemang and [Kitware](https://kitware.com/). Original credits and the
[MIT license](LICENSE.txt) are retained.

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
remains available when compilation or loading is unavailable. The current native
screen remains guarded to audited CPython 3.12 arithmetic. See
[build and runtime details](docs/compiled_residual.md), [release notes](RELEASE_NOTES.md),
and [validation](VALIDATION.md). The previous 1.3.3 repair remains intact.

## Install from this release checkout

Python 3.11+ and scikit-learn 1.6+ are required. The base package depends on
NumPy, SciPy, and scikit-learn:

```shell
python -m pip install .
```

The optional linear SOCP classifier requires CVXPY:

```shell
python -m pip install ".[socp]"
```

The extra does not switch the kernel classifier to CVXPY. Installing `dwd` by
name from PyPI may retrieve a different upstream release; the commands above
install the contents of this checkout. See [VALIDATION.md](VALIDATION.md) for
the tested environments and release checks.

## Kernel DWD

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
`acceleration=None`,
`initialization='auto'`, `stopping='objective'`, `obj_tol=1e-5`, and `max_iter=100`.
Auto initialization is zero for optimized fits. The intercept is unregularized.
The parameter names remain `lambd`, `q`, and `kernel_kws`.

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
rejects a fit. Centering is a solve representation; the fitted kernel, penalty,
and intercept remain unchanged. Unknown kernel families still require PSD
validation. Explicit spectral and L-BFGS backends, and checked eigenpair reuse,
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
Exact work has fixed entry/rank/arithmetic budgets and stays lazy on healthy fits.
An earlier characterization rejected nine extreme nearly constant synthetic
RBF fits; see [validation](VALIDATION.md) for the current replay outcome. This is
not a guarantee that every valid kernel will fit. See the numerical boundaries
in [the guide](docs/kernel_dwd.md) and [validation](VALIDATION.md).

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
eigenpairs. This is not a guarantee for every runtime or input. A detected failure
on another installation must be resolved there. Dense memory and eigen-preparation
time still matter; EVD can need more workspace than other symmetric drivers.
See [VALIDATION.md](VALIDATION.md) for the historical release checks and their scope.

The default stops on absolute successive-objective change. `converged_` instead
reports a checked numerical stationarity criterion on the returned model; inspect
it separately from `termination_reason_`. Numerical stopping can be requested
with `stopping='optimality', tol=1e-6`; `stopping='fixed'` requests a fixed MM
budget. A larger budget or a smaller numerical residual does not guarantee better
classification accuracy. See [the kernel guide](docs/kernel_dwd.md) for validation
stopping, callbacks, diagnostics, batching, and numerical boundaries.

Optimized MM also offers `acceleration='restart'`. It extrapolates decision
values and restarts momentum when necessary, using the same full kernel,
regularizer, and free intercept. The reference and L-BFGS paths do not accept
this option. Faster objective convergence can reduce classification accuracy at
a fixed update budget, as observed in the development comparisons, so ordinary
MM remains the default. Compare acceleration and stopping through an external
validation procedure when predictive performance is the goal.

## Explicit parameter search

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

## Linear classifiers and SOCP

`dwd.gen_dwd.GenDWD` provides generalized linear DWD with corrected/default or
explicit historical algebra and objective/fixed/optimality stopping. Its linear
explicit-system path remains available through `implicit_P=False`.

```python
from dwd.socp_dwd import DWD  # requires the socp extra

linear_socp = DWD(C='auto').fit(X, y)
print(linear_socp.C_, linear_socp.problem_.status)
```

SOCP uses a distinct constrained linear formulation and a penalty named `C`.
It preserves vectorized constraints, solver-status checks, and the unregularized
intercept. Its finite `optimal_inaccurate` results are accepted with CVXPY's
warning; that status does not certify a requested residual tolerance.

## Supported boundaries and tests

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

Background: Marron, Todd and Ahn (2007), *Distance-weighted discrimination*;
Wang and Zou (2018), *Another look at distance-weighted discrimination*,
[DOI 10.1111/rssb.12244](https://doi.org/10.1111/rssb.12244).
