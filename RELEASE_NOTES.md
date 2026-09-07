# Release 1.3.2

This local point release fixes generic internal CV selection and reduces Python
conversion work in compensated prediction. Source and exact-wheel suites each
passed all 324 tests. The same wheel passed 318 tests with CVXPY blocked; six
optional tests skipped, and CVXPY was not imported.

- Generic internal CV rejects nonfinite, non-real, or nonscalar scorer results
  before selection or refitting, with candidate/fold/train-test context. It also
  checks finite score means. It fails clearly instead of silently excluding a
  candidate. Supported finite Python/NumPy real numeric scalars retain their
  values, aggregation order, and first-tie behavior.
- Compensated query evaluation converts the same ordered float64 product terms
  to Python floats before the unchanged `math.fsum`. Kernel values, coefficients,
  intercept, compensation decisions, thresholds, and summation order are unchanged.
- Two docstrings now correctly describe supplied-eigenpair validation and
  unnormalized functional decision values.

A separate saved-model prototype measured decision evaluation on 1,000 official
**training** rows for each of three existing models, with four BLAS threads and
six paired repetitions after warmup. In the two 3-8 full models using compensated
rows, median time decreased by 15.7% (reference) and 15.0% (optimized); score bytes
were identical. The control with no compensated rows changed by less than 1%.
These are scoped query measurements, not a new fitting benchmark, test-set
experiment, or universal speedup. See [VALIDATION.md](VALIDATION.md).

Both repaired implementations and their defaults remain unchanged: reference
coefficient/eigenbasis MM with native normalized Gaussian initialization, and
optimized ordinary MM with native zero initialization; q=1, free intercept,
objective tolerance 1e-5, cap 100, and acceleration disabled. Optional linear SOCP
support and the existing numerical-recovery limits are unchanged.

The validated wheel is `dwd-1.3.2-py3-none-any.whl`, SHA256
`aad920b143b206efd596a22c8b241332de6921b2941fd6dd4c2566c276affd1f`. Assembly preserves these exact wheel bytes. This patch does
not claim to fix Task6 classification accuracy: its guarded external CV did not
use the defective generic-CV selection path. No accuracy or universal numerical
success guarantee is made.

Previous [1.3.1 release notes](docs/history/RELEASE_NOTES-1.3.1.md) and
[1.3.1 validation](docs/history/VALIDATION-1.3.1.md) are preserved as historical
records. Their old counts, model outcomes, and timings are not new 1.3.2 checks.
