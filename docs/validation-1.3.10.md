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

In the paired study comparison, ensemble DWD warm prediction mean time improved
19.69% on digits 2 versus 3 and 12.76% across all 30 ensemble DWD cases. Fitting
speed did not show an established improvement. Nine full-kernel cases were
1.59% slower to fit and 1.57% slower to predict; the cause remains unresolved and
the stated 2% non-regression bound was not established. A narrower same-model
readout follow-up did not reproduce that slowdown. The primary observations
remain part of the evidence. These are workload-specific results, not a promise
of improvement on every dataset or machine.

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
