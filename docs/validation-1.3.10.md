# DWD 1.3.10 validation scope

This release of the slicersalt/dwd fork was developed under Chang Cheng's guidance.
It retains the DWD objective, practical unregularized intercept, solver tolerances
and defaults. The executable classifier changes since 1.3.9 are limited to
coefficient preparation reuse and bounded batching in accurate kernel scoring.
The new resumable CV functionality is a separate example; the existing `run_cv`
API is unchanged.

## Implementation validation

Comparison used the actual package builds from the preceding six-classifier
study, including its unchanged native extension. Shared checks passed 486 tests
on each version. Focused scoring checks passed 41 tests with two platform skips
and 318 direct three-way helper comparisons. The skipped checks require a NumPy
floating type wider than binary64, unavailable on the tested Windows runtime.
Standalone checkpoint/CV checks passed 33 tests. These sets overlap and are not
counts of independent scientific experiments.

The helper comparisons cover ordinary layouts and dtypes, cancellation, signed
zero, subnormals, large coefficients, sparse fallback, allocation failures and
exception ordering. The existing three-pair study comparison retained 69 paired
cases and 138 fits with identical recorded fitted outputs, predictions and
accuracy. Neither this evidence nor a solver stopping flag proves convergence
or a universal accuracy guarantee.

## Performance limits

In the original paired study comparison, ensemble DWD warm prediction mean time
improved 19.69% on digits 2 versus 3 and 12.76% across all 30 ensemble DWD cases.
Fitting speed did not show an established improvement. Nine full-kernel cases
were 1.59% slower to fit and 1.57% slower to predict. Those historical
observations are retained.

Follow-up investigation on October 5, 2026 found unequal concurrent numerical
work during the original comparison. A 144-fit repeat across the same nine
saved cases included 54 baseline/current pairs and 18 identical-baseline
control pairs. The aggregate current-versus-baseline fit difference was
-0.060%, with an approximate 95% block-bootstrap interval of -0.161% to +0.047%.
The repeated fit and scoring estimates did not establish a persistent package
regression. This interval describes the fixed cases and conditions, not a
universal or worst-case 2% bound.

A separate 24-fit controlled comparison used three saved cases, both original
package versions, and an identically prepared companion that was either idle
or running a representative prediction workload. Competing work increased
mean fit time by 3.85% for the baseline and 4.48% for the current version, with
a slowdown in all 12 within-version comparisons. Eight additional instrumented
fits on the 4-versus-9 case found added time in unchanged native residual
arithmetic, thread discovery and other fit work. The earlier case-specific
version difference did not persist; the changed scoring helper ran once per
fit and took 2.9 to 3.6 milliseconds. All 32 new fits retained the original
recorded fitted state, predictions and accuracy exactly.

Concurrent work is therefore a demonstrated timing confound and a credible
contributor to the original result. Its precise share of the full historical
percentage cannot be reconstructed from the saved telemetry, and the controlled
workload was not an exact replay of every historical activity. The timing
investigation is closed without a demonstrated package regression. These
results do not promise improvement on every dataset or machine. The follow-up
changed no estimator code, mathematical method, solver defaults or release
binaries.

## Resumable-example numerical qualification

The example owns a defensive copy of its training data. The older `run_cv` path
can retain the caller's original array. The existing sklearn RBF calculation
can produce slightly different self-kernel and equal copied-query values.
On a deliberately unscaled, duplicated/collinear 36-row fixture with a feature
scaled by 1e5, this distinction changed one training prediction in each solver
implementation, with a maximum score difference of approximately 9.76e-7.

CV scores, selection, fitted coefficients and intercepts agreed. The original
build reproduces the same ownership sensitivity. Common independent-query and
matched-ownership controls agree exactly. This is an observable training-query
compatibility limitation, not a change to the DWD objective, and unconditional
prediction-bit equivalence is not claimed. The protective copy is retained.

Checkpoint reuse verifies data, parameters, folds and implementation identity.
Unfinished fits restart; completed CV replay still requires a fresh final model.
The example deliberately rejects unsupported callable and precomputed kernels.
Native and portable release artifacts are checked separately from the preceding
implementation comparisons; published release notes identify those checks.
