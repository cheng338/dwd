# DWD 1.3.4 validation

The source was validated in py12_df: CPython 3.12.14, NumPy 2.5.2, SciPy 1.18.0,
scikit-learn 1.9.0 and MKL 2025.3.

| Check | Result |
|---|---|
| Complete source suite with CVXPY | 367 passed |
| Complete source suite with CVXPY blocked | 361 passed; six expected optional-solver skips |
| Published ensemble-dwd 0.1.1 | 85 passed; parent and worker imports verified |
| Exact original full-training 2/3 fit | 100 updates; all stored states and probes bitwise identical |
| Frozen public synthetic replay | Same 21 returned models and nine rejected case IDs |

Source, extension, test and runtime hashes were checked before and after suite
execution. Actual-wheel, portable-wheel and additional-runtime results are in
the accompanying build and validation receipts. A source pass alone does not
certify a different binary artifact.

## Independent arithmetic checks

The five compiled accumulation helper bodies retain CPython's order of operations.
The core is built without reassociation, fast-math or implicit contraction.
An independent core oracle checked 153 cases with 2,444 exact-Fraction enclosures
and 768 Decimal crosschecks. The actual-extension oracle checked 139 cases and
compared all six outputs against disabled compilation: scores, expanded residuals,
coefficient-sum residual and all three bounds matched bitwise wherever returned.
Acceptance/declines, thread budgets and partial-output fallback also matched.

Eleven new bundled tests cover arithmetic, unusual strides and unaligned buffers,
readonly inputs, output validation, thread budgeting and optional-extension
fallback. A separate compiled-core test verifies that unsupported rounding and
underflow modes are declined and restores the floating-point state afterward.

## Fixed full-MNIST regression

The fit uses all 12,089 official training images, the same StandardScaler,
gamma=1e-4, package lambd=2^-31, q=1, zero initialization, optimized Cholesky MM,
eight native threads and the unchanged 100-update/objective-change defaults.
No parameter search or official test-label evaluation was performed for this repair.

The 47.12-second fit retained all saved coefficients, intercept, objective history,
class mapping, preprocessing and scores/predictions on 257 fixed training probes
bit for bit. All stopping/convergence flags matched. It still reached 100 updates
without declaring convergence, as the saved model did. This verifies the same
finite-iteration classifier, not a newly converged optimum.

The same 105 compensated checks and five numerical refinements occurred. Their
elapsed component fell from 352.11 seconds in the saved primary fit to 20.03 seconds.
Native and compensated timer fields overlap and must not be added. The previous
377.39-second figure is a three-fit median; 47.12 seconds is the initial fresh
repaired fit. Accompanying receipts record actual-wheel repetitions separately.

## Preserved limitations

All 21 returned public synthetic models passed independent Decimal100, finite-output,
classification and serialization checks, retaining saved coefficients, intercept,
decisions and objective histories. The nine extreme nearly constant RBF failures
retain their original case identities. No additional rejection was introduced.

The accelerator is optional. Unsupported runtimes/arithmetic retain the previous
checker. Benefits are largest when compensated checking dominates; ordinary fits,
eigendecomposition and kernel construction have different costs. These checks do
not establish universal speed or accuracy improvements.
