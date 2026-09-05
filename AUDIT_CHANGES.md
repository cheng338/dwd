# Local audit candidate: 1.0.5+audit1

This is a development/release candidate, not the upstream PyPI release.
It retains the upstream MIT license and original author credit. Original implementation:
Iain Carmichael; upstream maintenance: David Allemang / Kitware; upstream repository:
https://github.com/slicersalt/dwd . User-maintained compatibility fork:
https://github.com/cheng338/dwd . Audit/optimization work was prepared with Codex.

## Changes that can change scientific results

- `KernGDWD(..., solver_mode='schur')`: repairs two kernel-MM algebra errors in
  the Schur denominator and the intercept/coefficient correction. The intercept
  is still unregularized. Reference: Wang and Zou, *Another Look at
  Distance-Weighted Discrimination*, JRSS B 80(1), 177-198, DOI10.1111/rssb.12244,
  Section4.1. Direct augmented-system, finite-difference and singular-system
  checks are independent references in the audit tests.
- `GenDWD(..., solver_mode='schur')`: repairs the linear Sherman-Morrison
  correction. `implicit_P=False` is now actually forwarded to the linear solver.
- Both generalized estimators retain `solver_mode='legacy'` as their library
  default only for compatibility. This mode deliberately retains known wrong
  update algebra; it is not recommended as a mathematically correct solver.
  Select `schur` explicitly when using the corrected solver.
- `KernMD`: correct RKHS mean-difference coefficients and half-norm intercept.
  The formerly inconsistent `naive_bayes=True` path explicitly raises rather
  than silently fitting/scoring two different representations.
- Standalone `dwd.svm.SVM` maps binary labels before its original L1-regularized
  mean-hinge optimization. This is not sklearn's SVC and its C convention differs.
- Built-in CV now enumerates the Cartesian grid, trains on training folds,
  stratifies classifiers, invalidates wrong-kernel caches, slices precomputed
  kernels correctly and refits the selected estimator. These changes concern
  the package's built-in CV, not scikit-learn's GridSearchCV implementation.

## Speed and API changes

- Vectorized NumPy loss/gradient replaces Python scalar loops without changing
  scalar formulas; integer-input loss and gradient return floating values.
- Reuse scaled eigenvectors and score products; materialize negative-stride
  eigenvectors once instead of triggering a matrix copy in every BLAS call.
- Expose max_iter, obj_tol, random_state, explicit precomputation and objective/
  iteration/convergence/gradient diagnostics. No warm starts are enabled by default.
- SOCP constraints are vectorized with the same feasible set/objective. Solver
  status and finite values are checked. Auto-C no longer mutates constructor C.
- Repair classifier tags, scalar extraction on NumPy2.4, callable/precomputed
  kernels, missing validation, helper solver argument forwarding and plotting
  edge handling. Sample weights remain unsupported and are rejected explicitly.
- Final release checks normalize array-like precomputed K before diagnostics,
  reject incompatible dense/sparse or solver-mode caches, and guard nonfinite
  initial/iteration objectives. Corrected mode promotes raw features to float64
  **before** constructing Gram matrices. Legacy float32 arithmetic is preserved.
  External precomputations must be computed at adequate precision; casting a
  rounded float32 Gram matrix to float64 cannot undo a PSD violation.

## Numerical and API boundaries

The kernel solver requires a finite symmetric dense Gram matrix. Corrected mode
rejects materially indefinite matrices. It also rejects numerically rank-deficient
kernels when the shifted spectral condition estimate exceeds 1/sqrt(machine eps),
because coefficient nullspace growth can corrupt the reported RKHS norm. This is
a conservative implementation boundary, not an accuracy guarantee below it.
No hidden eigenvalue truncation, diagonal ridge, approximation, intercept penalty,
or silently changed regularization parameter is introduced. A range-space solver
for extremely small penalties on singular kernels remains a future extension.
The auxiliary SOCP-parameter conversion C_ can equal zero for a zero fitted norm;
that degenerate conversion is not a usable positive SOCP penalty or a guarantee
that rescaling recovers an equivalent SOCP fit.

Explicit K and K_eig are advanced trusted precomputations for identical data/order,
preprocessing and kernel. Shape/finiteness are checked, but an O(n^3) decomposition
reconstruction is not repeated per candidate. Do not slice full-data eigenvectors
to construct fold or subset eigensystems. Cached decompositions must be specific
to the training fold, data values, row order and kernel parameters.

Kernel `implicit_P=False` remains explicitly unsupported. `KernelScaler` preserves
its historical square-kernel diagonal-n normalization; it is not a centering or
general rectangular-query transformer. A small objective change is only that
stopping rule, not proof that the optimum or sufficient statistical accuracy was reached.

The repository test suite is self-contained; frozen historical references carry
their original MIT notice and provenance. Correctness tests and finite-step
legacy-parity tests are separate. Validate speed, accuracy and convergence for
each application independently. A finite iteration cap is not evidence of a
converged solution. Preserve earlier package versions and results when upgrading.
