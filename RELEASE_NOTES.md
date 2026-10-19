# DWD 1.3.16 — October 19, 2026

This release makes guarded joint affine scoring and eligible adaptive residual
ordering the defaults for new kernel fits. It also extends qualified native
arithmetic, reduces repeated internal work, and strengthens recovery, prediction
ownership and interruption handling. The DWD objectives, unregularized intercepts,
solver tolerances, stopping defaults and numerical acceptance criteria are
preserved. Floating-point scores and finite-iteration results can change.

## Numerical policies and recovery

- `affine_computation='auto'` resolves to joint affine scoring in corrected
  Schur mode and standard scoring in legacy mode. Explicit standard scoring
  remains available, and saved models retain their fitted policy. A guarded
  exact-zero shortcut avoids unnecessary rational summation without bypassing
  difficult-margin or unsupported-arithmetic fallbacks.
- `residual_check_order='auto'` selects adaptive checks on eligible optimized
  Cholesky paths. Reference, spectral, L-BFGS, legacy and supplied-eigenbasis
  auto routes retain their applicable historical order. Requested, initial and
  effective policies remain observable through recovery.
- An eligible typed numerical failure in restarted MM can take one fresh
  spectral retry from the original initialization. Restart acceleration and
  L-BFGS selection remain explicit; acceptance checks and tolerances are not
  weakened, and arbitrary failures are not retried silently.
- Explicit `literal_objective_diagnostics` evaluates the represented kernel
  objective, coefficient/intercept gradients and a dual lower bound within
  fixed arithmetic limits. It does not replace production acceptance or certify
  an unrounded analytic kernel.
- New valid legacy kernel fits emit `FutureWarning` after input validation.
  Historical update equations and prediction from saved models remain available.
  Direct RBF computation and extended exact recovery also remain explicit options.

## Internal computation and fitted-model behavior

- The guarded compiled residual and dense accurate-score paths are qualified
  on conventional Windows x86-64 CPython 3.11 through 3.13. The separate scalar
  `math.sumprod` route retains CPython 3.12 support and qualifies conventional
  Windows x86-64 CPython 3.13. Every arithmetic guard and portable fallback remains.
- Eligible native paths prepare coefficient splits once per bounded chunk,
  reuse solver-scoped residual workers and call-local dense-score buffers, and
  avoid a redundant scalar attempt after the original compiled helper refuses
  an operand outside their shared domain. Allocation failure, custom helpers
  and unsupported arithmetic retain the corresponding fallback.
- Eligible dense RBF prediction reuses training norms within one call. Direct
  RBF prediction validates its fitted basis once per eligible call while keeping
  query checks, sparse canonicalization and custom behavior. No persistent
  prediction cache is introduced.
- `KernGDWD` owns retained dense or sparse training features and snapshots kernel
  parameters. Later caller edits no longer alter an already fitted predictor;
  refitting uses the newly requested parameters.
- `KernGDWDCV` honors post-fit `prediction_batch_size` changes. Both kernel
  estimators accept a per-call decision-function batch override without changing
  the fitted model.
- Compiled residual work is bounded and completion is polled for interruption.
  Workers are joined and partial outputs discarded. Checkpoint cleanup preserves
  the original failure or interruption and serializes strict JSON before creating
  a temporary file.

## Validation and compatibility

Prior application checks ran 57 focused tests on each of Python 3.11, 3.12 and
3.13 with native and portable wheels, with no failures, errors or skips. They
verified exact installed runtime payloads and actual native or portable execution.
The earlier 960-method full suite, with four documented platform/runtime skips,
was reused after checking the runtime sources and compiled helper; it was not
presented as another full-suite run. Independent RBF primal/dual checks cover
generalized loss, duplicate rows, imbalance and finite update budgets. Strict
Sphinx checks also passed. See [VALIDATION.md](VALIDATION.md) for the dates,
environment and evidence boundaries. These prior checks are separate from the
release artifact checks and hosted CI; no universal speedup is claimed.

CI verifies installed wheel payloads and execution outside the package source
tree. An existing matrix leg supplies portable coverage. Push CI runs on branches,
avoiding a duplicate matrix for release tags while retaining pull-request and
manual runs. Changed example and dependency sources require fresh checkpoints
under the existing strict identity checks; historical receipts are not migrated.

Python 3.11+ remains required. The Windows AMD64 native wheel is
`dwd-1.3.16-cp311-abi3-win_amd64.whl`; `dwd-1.3.16-py3-none-any.whl` omits the
optional helper. Extension-loading compatibility does not guarantee numerical
path eligibility. NumPy/SciPy BLAS acceleration is independent. Python 3.10
companion maintenance remains ended; historical assets are retained.

This is a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license are
retained. See the [changelog](CHANGES.md), [installation](README.md#installation)
and [previous release notes](docs/history/RELEASE_NOTES-1.3.15.md).
