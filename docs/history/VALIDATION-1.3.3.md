# DWD 1.3.3 validation

The release source was tested in the user's py12_df environment: CPython
3.12.14, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0 and MKL 2025.3.
An independent Python 3.11.16 / NumPy 2.4.6 / SciPy 1.17.1 / OpenBLAS
environment exercises the portable fallback.

| Source configuration | Passed | Expected skips |
| --- | ---: | ---: |
| MKL with CVXPY | 356 | 0 |
| MKL, CVXPY blocked | 350 | 6 optional CVXPY tests |
| OpenBLAS with CVXPY | 353 | 3 CPython 3.12 native-arithmetic tests |
| OpenBLAS, CVXPY blocked | 347 | 6 CVXPY and 3 native-arithmetic tests |

All configurations collected the same 356 test IDs. Source, test and fixture
hashes remained unchanged. Base-only runs confirmed CVXPY was not imported.
Wheel validation receipts are distributed alongside the built artifacts; source
passes alone do not assert that a particular uploaded wheel passed CI.

The actual published ensemble-dwd 0.1.1 package passed all 85 integration/unit
tests against this DWD source. Main-process and worker imports were checked,
including serial/parallel GridSearchCV, shared parameter forwarding, caching,
serialization and stopping controls. The full classifier comparison stayed
stopped throughout this repair.

Twenty-two new tests cover exact portable arithmetic compatibility, native
forward-error bounds and guards, and solver acceptance/fallback behavior.
An independent Fraction-based probe checked 920 residual, score and constraint
inequalities, including cancellation and magnitude boundaries. No native
arithmetic method is used outside the guarded CPython 3.12 implementation.

## Actual slow MNIST fold

The isolated repair test uses digit pair 2 versus 3, fold 0 of the saved fivefold
split: 9,671 training rows and 2,418 validation rows. StandardScaler is fitted
only to training rows. RBF gamma is `1e-8`, `lambd=2**-30`, `q=1`, zero
initialization, optimized Cholesky MM, and four BLAS threads. No official MNIST
test labels were used. This is a solver regression, not resumed tuning.

| Measurement | Published 1.3.2 | Repair |
| --- | ---: | ---: |
| Three-update fit | 94.27 s | 23.69 s |
| Seven required residual checks in that fit | 84.88 s | 14.24 s |
| Three-update model / objective / prediction bytes | Identical | Identical |
| Complete 100-update isolated repair fit | Not rerun in isolation | 452.50 s |
| Full-fit validation accuracy | 97.0223% saved result | 97.0223% |

The three-update baseline includes one diagnostic state write. The historical
complete baseline fit took 4,317.29 seconds with competing CV workers; its time
cannot be compared as if execution conditions were identical. The new full fit
performed 100 original-system solves, retained the unchanged 100-update cap,
used no kernel approximation or added regularization, and reconstructed its
scores and objective with zero reported discrepancy. Like the saved baseline,
it reached the cap without satisfying the convergence check.

Twelve additional paired full-budget checks across 2/3, 4/9 and 3/8 used RBF,
linear and Laplacian kernels and different regularization/loss settings. All
coefficient, intercept, objective-history and prediction bytes matched 1.3.2;
the new path was exercised in 603 numerical checks. These small paired runs
were numerical regressions, not parameter searches or estimates of test accuracy.

## Remaining limitations

A frozen 30-case public synthetic replay retained 21 returned models and the
same nine extreme nearly constant RBF rejections documented for 1.3.2. All 21
returned models passed Decimal100 score/objective, finite-output, class and
pickle checks and retained identical stored states and decisions. There were no
new rejections. This release repairs the performance bottleneck; it does not
claim those nine unresolved cases are solved or promise success for every input.

Runtime guards retain the portable method on other Python versions and outside
the supported native-arithmetic range. Error bounds rely on IEEE binary64
round-to-nearest and the documented implementation assumptions. Objective-change
stopping remains distinct from stationarity. Historical results remain in the
[1.3.2 validation record](docs/history/VALIDATION-1.3.2.md).
