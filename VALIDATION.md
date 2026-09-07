# DWD 1.3.2 publication validation

This record describes the revised publication snapshot. The earlier local 1.3.2
snapshot passed 324 tests under MKL, but initial GitHub checks subsequently
exposed two test portability problems and a real spectral rounding failure
under OpenBLAS. Those earlier passes are not evidence that the original snapshot
passed other runtimes. The original local archives and receipts are preserved.

## Current source and actual-wheel checks

| Check | Result |
|---|---|
| Revised source, updated MKL | 334 passed; no skips, failures or errors |
| Actual rebuilt wheel, updated MKL | 334 passed; no skips, failures or errors |
| Same wheel, MKL with CVXPY blocked | 328 passed; six expected optional skips; no CVXPY import |
| Same wheel, OpenBLAS | 334 passed; no skips, failures or errors |
| Same wheel, OpenBLAS with CVXPY blocked | 328 passed; six expected optional skips; no CVXPY import |
| Focused rounding-repair tests, both runtimes | 22 passed per runtime, including the previously failing cases |
| Stored public synthetic cases, actual rebuilt wheel | 21 returned with all independent checks passing; nine known extreme RBF cases rejected |

Each suite verifies the imported package path, version, complete package and
test/fixture hashes, actual wheel archive, build receipt, validation helper and
native runtime before and after execution. The wheel is extracted in isolation;
testing does not install it into the user's environment. Numerical suites run
serially with at most four BLAS threads; the public synthetic replay uses two.
Public release checksums identify the exact uploaded archives.

The MKL checks used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0,
and the updated native MKL runtime reporting 2025.3-Product. The OpenBLAS checks
used Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.0; NumPy and
SciPy reported OpenBLAS 0.3.31.188.0 and 0.3.30 respectively.

The [GitHub workflow](.github/workflows/tests.yml) additionally tests Linux and
Windows, Python 3.11 and 3.12, and base/SOCP installations: eight combinations.
All eight jobs must pass on the publication revision before merge and release.
The [Actions record](https://github.com/cheng338/dwd/actions) reports those runs.
This matrix tests the dependencies resolved at execution time, not every
possible combination satisfying the package's dependency bounds.

## What publication checks repaired

Windows Git checkouts could translate line endings in frozen Python test
oracles, breaking their exact-byte provenance checks. A narrow `.gitattributes`
rule preserves those oracle bytes, and the sdist explicitly includes the rule.
No oracle content or expected hash was weakened.

Some native solves already satisfied the original equations without needing
the scalar intercept correction that one natural-case test assumed. That test
now permits zero corrections while retaining its Decimal accuracy thresholds
and bounded refinement checks. A separate injected case still requires exactly
one correct scalar adjustment.

An extreme n=120 synthetic kernel exposed a real float64 readout problem under
OpenBLAS: multiple checked eigenbasis representations converged to the same
rounded coefficients, whose original-equation residual was too large. Nearby
float64 coefficients could satisfy the existing checks. The revised spectral
solver therefore has a lazy final correction that proposes adjacent floats,
recenters the free intercept and accepts only freshly checked improvements.
The full residual, coefficient-sum and RKHS gates remain unchanged. At most
eight accepted moves and a fixed abstract dense-work budget are shared across
all five inverse representations of one solve action. Failed private changes
are discarded. Healthy solves bypass this correction. See the
[kernel guide](docs/kernel_dwd.md) for precise limits and diagnostics.

Regression controls include the frozen failing case, a permutation, a small
right-hand-side perturbation, independent Decimal80 equations, healthy solves,
wrong solver outputs, invalid states, unchanged final gates, rollback and shared
budgets. These are numerical refinements of the same full-kernel DWD objective
with an unregularized intercept; they do not change the MM update count,
regularization convention, kernel, stopping defaults or classification threshold.

## Public synthetic boundary replay

The actual rebuilt wheel replayed exactly 30 stored public-estimator cases:
ten saved kernel/size/shift combinations, each with reference zero initialization,
reference native initialization and optimized zero initialization. Sizes are
30, 75 and 120; q=1, seed=314159, package lambd=2*shift/n, and five ordinary MM
updates with a free intercept. This is a bounded numerical diagnostic, not a
classification or default-convergence benchmark. No MNIST data was accessed.

The replay took 16.26 seconds including checks. All 21 returned models passed
independent Decimal100 decisions and original-objective checks, finite-output
checks, label checks and serialization equality. Maximum absolute score and
objective errors were 2.271e-13 and 1.197e-13. All 30 input identities and fit
success/failure outcomes matched the historical 1.3.1 replay. Nine nearly
constant RBF cases at shift=1e-14 still failed their first MM update after bounded
recovery, across the three implementation/initialization controls. This remains
an unresolved valid-input numerical limitation, not proof that their intended
MM functions are mathematically impossible.

## Earlier performance and CV evidence

The incremental internal-CV defect was reproduced with a metadata-only estimator:
a NaN-scored candidate could beat a finite candidate and be refitted. The suite
now rejects invalid scores before selection/refitting while retaining supported
finite-score aggregation and first-tie behavior. Task6 used a guarded external
scorer, so this defect does not explain its accuracy differences.

The compensated-prediction change converts the same ordered float64 product
terms to Python floats before the unchanged `math.fsum`. An earlier saved-model
prototype used 1,000 training queries, four BLAS threads, warmups and six paired
timings. Scores were bitwise identical. Reference and optimized 3-vs-8 decision
evaluation improved from 0.663190 to 0.559000 seconds (15.7%) and from 0.642350 to
0.545819 seconds (15.0%). A control with no compensated rows changed by less than
1%. These scoped prediction timings were not rerun as new whole-fit benchmarks.

The accumulated repairs have shown faster training than original Slicersalt in
MNIST comparisons, with mixed accuracy. Historical results and the new numerical
tests do not establish universal accuracy gains, convergence at the 100-update
cap, or success for every valid kernel. The defaults remain ordinary MM,
q=1, objective tolerance 1e-5 and max_iter=100, with optional acceleration off.
Earlier [1.3.1 release notes](docs/history/RELEASE_NOTES-1.3.1.md) and
[validation](docs/history/VALIDATION-1.3.1.md) retain their historical scope.
