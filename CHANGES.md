# Updates

This changelog covers the fork of [slicersalt/dwd](https://github.com/slicersalt/dwd)
maintained at [cheng338/dwd](https://github.com/cheng338/dwd). Subsequent fork
development is guided by Chang Cheng. The original implementation by Iain
Carmichael, upstream maintenance by David Allemang and Kitware, and the MIT
license remain credited in the [README](README.md) and [license](LICENSE.txt).

## 2.0.2

- The `test` and `test-base` extras support pandas 2.2.0 with NumPy 1.26
  on Python 3.11 and 3.12. The pandas minimum is 2.2.3 on Python 3.13,
  2.3.3 on Python 3.14, and 3.0.6 on Python 3.15 and later.
- Require NumPy 1.26.3 or later and SciPy 1.11.4 or later. NumPy 1.26.3
  fixes a spurious overflow when scaling very small kernel values. SciPy 1.11.4
  preserves 32-bit indices when cross-validation converts diagonal sparse arrays
  and supports the EVD driver for eigenvalue decomposition of a one-by-one matrix.
- Estimators that accept sparse features now handle two-dimensional integer COO matrices and arrays,
  including inputs with duplicate entries at the same coordinate.
- On Windows with NumPy 1.26, spectral recovery uses the supported exponent
  type, and compiled affine reporting uses pointer-width row indices.
- Training rejects NaN and infinite class labels in extended-precision and
  byte-swapped floating arrays on the minimum scikit-learn version. Accuracy
  scoring preserves exact integer IDs outside the signed integer range when
  supplied in these floating dtypes.

- Classifier accuracy accepts NumPy variable-width string labels, including
  those supplied by a pandas sparse target. Weighted scoring preserves
  distinct labels such as `"q"` and `"q\0"`.

- Fitting rejects datetime and duration targets with a missing (`NaT`)
  training row. A sparse target may still use `NaT` as its fill value when
  every training row has a stored label.

- Fitting, scoring and validation preserve distinct instants and durations
  in pandas sparse targets, including object-typed targets whose observed
  labels use different time units. If fitting cannot infer a shared unit,
  supply a NumPy datetime or timedelta array built from exact tick counts
  in a unit that represents every label. Scoring can compare individual
  observations across units even when the whole target has no such unit.
  When sparse fitting produces native datetime or timedelta classes,
  scoring and validation accept the same scalars in an object array or
  an object-typed pandas Series.

- Preserve distinct class IDs when a boxed target mixes a NumPy floating-point
  label with an adjacent large integer. This includes NumPy scalar subclasses
  with inherited numeric ordering and pandas sparse object labels. Binary
  estimators and the kernel mean-difference helper recognize both classes and
  reject a third distinct ID. Fitting, validation monitoring and scoring
  preserve custom scalar equality for boxed integer class labels, including
  comparisons with numeric IDs stored in zero-dimensional arrays. Numeric
  lists and tuples preserve scalar-defined comparisons during training,
  scoring and held-out validation, including for a single row. Numeric labels
  that override integer or floating conversion retain their stored IDs, and
  validation checks the stored values for finiteness. Realness checks respect
  the original label's dtype and NumPy dispatch. Targets and individual rows
  with explicit array representations use their projected labels consistently
  during fitting, scoring and validation. Training rejects mixed
  comparisons that assign observations to overlapping classes.
  A pandas sparse integer target with an occupied text fill keeps integer
  `1` distinct from string `"1"`. Fitting rejects the mixed labels, and
  accuracy scoring compares each prediction with the original observed label.
  Scoring also accepts a fill such as a `str` subclass that inherits ordinary
  string behavior, on pandas versions that support that sparse dtype.
  When a sparse target supplies labels by iteration, fitting and validation
  also distinguish numeric IDs from such strings. For example, an integer
  `1` in validation data cannot match the training class `"1"`. A trailing
  NUL also distinguishes text labels such as `"a"` and `"a\0"`.
  Large integer IDs also survive mixed sparse fill and stored values,
  numeric scalar object arrays, and positional sequences such as a UserList or deque.
  A DataFrame or sparse categorical level subclass can supply values that
  match an ordinary pandas conversion that changes a stored class ID. When
  that leaves two possible label sets, pass `np.asarray(y)` to choose the
  projected labels or supply exact labels built from the stored sparse scalars.

- Preserve large integer class IDs in pandas nullable integer targets and
  categoricals with sparse integer categories, including during kernel DWD
  validation monitoring. Fitting and scoring also preserve sparse `uint64`
  labels. Mixed numeric sequences preserve the original scalars when a
  single native integer dtype cannot hold them. Shared-fold search checks
  for label changes during pandas row selection and array conversion,
  including a second sparse output alongside a categorical column whose
  integer levels use sparse object storage. It also checks sparse datetime
  and duration labels whose fill or unit can change during conversion, and
  integer endpoints in directly stored sparse intervals that can round or
  overflow when a fold selects only intervals.

- Require threadpoolctl 3.7.0 or later. The dependency-floor CI check exercises
  residual worker planning with BLAS limits of one and two threads.

- Require joblib 1.6.0 or later so the process backend can track and clean up temporary files on CPython 3.15. See the [shared-fold search guide](docs/shared_fold_search.md#process-parallel-fallback) for how a process-parallel fallback handles training data and fold indices.

- The conic extra requires CVXPY 1.6.7, Clarabel 0.11.0, OSQP 1.0.4,
  SCS 3.2.4.post3 and highspy 1.7.1 or later, excluding SCS 3.3.0.
  See the [project metadata](pyproject.toml) for the complete dependency ranges.

## 2.0.1

- Repair the native-build tests' lookup of the platform-error exception in
  setuptools and the public-fit affinity test's expectations for Linux
  CPython 3.11. The affinity test exercises portable compensated recovery on
  that platform and checks the one-worker limit when native dispatch applies.
  [Test results for the 2.0.0 source tag](VALIDATION.md#dwd-200-source-tag-tests).

## 2.0.0

- Remove the unused private feature-gradient interface that recomputed operand
  splits, together with its test selector. The selected implementation reuses
  existing splits in gradient tiles and uses an existing context for eligible
  coefficient calculations. Numerical guards and fallback paths still apply.

- Clarify `KernelScaler.transform` copy ownership and its opt-in in-place
  failure behavior. The default copy preserves input on overflow refusal;
  `copy=False` has no rollback and can modify aliases before raising. Add
  focused copy-contract regressions without changing scaling arithmetic.

- Sum duplicate signed and unsigned integer sparse coordinates exactly before
  their single float64 conversion, including linear prediction and supported
  precomputed queries. Preserve floating and Boolean input policies and
  separate callable-return/direct-score contracts. This strengthens the
  prior entry-wise conversion contract; affected fits and predictions can
  change. Verified-unique linear queries retain ordinary dot-product conversion
  without an extra float64 copy. Historical records are not rewritten, and
  precomputed training remains dense-only.

- Remove the redundant root-logger warning around the optional CVXPY
  import. Preserve the original import failure so applications retain
  control of their logging configuration.

- Clarify that linear `P0_eig` is used and validated only with
  `implicit_P=True`. Explicit fitting ignores this optional argument and
  prepares its own checked factor. Solver behavior is unchanged.

- Keep the historical 2D notebook kernel examples independent of optional
  CVXPY. Missing CVXPY skips only the linear SOCP cells; unrelated import
  and solver failures remain visible. Clarify that the combined figure
  script requires CVXPY, and clear stale outputs only in changed cells.

- Document that generalized linear and kernel DWD can have non-unique free
  intercepts and different predictions at equal global objective values.
  Include an exact example and comparison guidance; solvers, prediction rules,
  numerical checks and defaults are unchanged.

- Clarify that exact-recovery rank, entry, bit and operation caps apply
  simultaneously. Rank is not row count, and satisfying the rank cap does not
  guarantee recovery. Arithmetic guards, profiles and defaults are unchanged.

- Decline the optional compiled residual route when its loaded module lacks
  an `evaluate` entry or exposes `None`, retaining the existing guarded
  fallback. Preserve errors from present evaluators and all arithmetic,
  acceptance and worker-cleanup behavior.

- Normalize maintained text to LF using repository attributes, preserving
  frozen byte-hashed test oracles, the exact 1.3.16 archive and Windows
  batch-file endings. Source bytes and their identities change, while
  executable semantics are unchanged. Exact-source checkpoints require
  their original source snapshot or a fresh run; historical identities
  are not rewritten.

- Restore omitted tagged-version history for 1.3.16 and 1.3.6, including
  an explicitly historical 1.3.16 notes archive. Preserve prior release
  claims and distinguish 1.3.17's test-fixture correction from inherited changes.

- Repair archived release-note paths and direct historical version references
  to the corresponding notes. Keep current runtime guidance separate from
  historical validation, without changing release claims or package behavior.

- Clarify current Windows general scoring, Linux private reporting scores and
  the separate joint-affine reporting entry, alongside scalar-eligible residual
  dispatch on other platforms. Distinguish guarded runtime eligibility from
  measured validation scope; executable code and numerical policies are unchanged.

- Distinguish checkout development tests from installed-wheel qualification.
  Document the existing detached test bundle, exact installed-artifact checks
  and isolated pytest commands without changing package code or CI behavior.

- Describe the combined DWD and CPython-derived arithmetic in distribution
  metadata with the SPDX expression `MIT AND Python-2.0.1`. Retain both complete
  license texts and the original source attribution in source distributions,
  native wheels and portable wheels. Require setuptools 77.0.3 or later for
  this metadata; numerical code, build modes and runtime dependencies are
  unchanged.

- Allow an explicit fresh retry of the resumable kernel-CV example after an
  interrupted first identity write leaves only its recognized unpublished
  temporaries. Preserve those files without reading or adopting their contents;
  retain strict resume, live-owner locking and later-stage evidence checks.
  Estimator APIs and numerical behavior are unchanged. See
  [initial setup recovery](docs/resumable_cv.md#interrupted-initial-checkpoint-setup).

- Reuse the current public query validation inside eligible dense batched
  `KernGDWD` prediction with guarded norm-sum RBF construction. Fresh training
  norms and storage/source checks qualify reuse only within that call; retain
  checked fallback for unsupported inputs, policies or replaced hooks. Kernel
  arithmetic and public validation rules are unchanged. See
  [prediction reuse](docs/kernel_dwd.md#kernels-batching-and-caches).

- Retain the private fold-token registry through the ordinary final refit in
  `fit_shared_fold_search` and clear it during helper exit cleanup. This removes
  the earlier pre-refit release while preserving cache statistics and the final
  estimator's owned feature copy. The helper remains opt-in; eligibility,
  numerical results and defaults are unchanged. Neither registry lifetime
  establishes a universal speed advantage. See
  [shared-fold ownership](docs/shared_fold_search.md#state-and-ownership).

- Share the unchanged zero-shift design-rank validator between the linear solver
  and explicit factor. The solver binds the existing validation function at
  import time, avoiding an unnecessary first-fit factor-module import. Numerical
  acceptance and defaults are unchanged; startup and repeated-fit costs remain
  workload dependent.

- Enlarge qualified warm native gradient/beta batches to 16,384 entries while
  preserving original first groups, arithmetic, bounds and fallback. Feature
  input staging is bounded at 128 KiB per larger call; the expansion artifact
  requires a rebuild for private API version 2. No public option is added.
  Saved-state reporting measurements show workload-dependent gains; see the
  [batching contract](docs/compiled_residual.md#exact-feature-expansion-tiles)
  and [reporting evidence](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting).

- Defer private native expansion-context creation until the ordered reducer
  finishes initialization. Preserve original fallback during loading, completed
  negative caching and later mutation checks. This prevents legitimate helper
  initialization from invalidating a prematurely captured context, without
  changing arithmetic, solver defaults or public options. See the
  [initialization validation](VALIDATION.md#native-loader-initialization-order-correction-2026-11-10).

- Update shared-fold search guidance with direct comparisons against current
  ordinary search. Retain the existing explicit helper and recommend choosing
  the route from representative whole-search measurements. Document the tested
  training-score settings, workload-dependent benefits and costs, and timing
  exclusions. No runtime behavior or search default changes; see the
  [shared-fold guidance](docs/shared_fold_search.md#when-to-use-it).

- Extend qualified dense named-linear L-BFGS reporting with native gradient
  tiles and warm beta calculations, reusing existing compensated splits and
  report-specific score products. Preserve arithmetic order, error bounds,
  callbacks, acceptance, optimizer work and guarded fallback. No public option
  is added. Eligible CPython 3.12 builds include a full-ABI expansion extension
  and use interpreter-specific wheel tags; other builds retain the existing
  helpers/fallback. Wider and repeated fits improve in the measured panels,
  while cold startup and missing-extension costs remain workload dependent.
  See [reporting costs](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting)
  and [native build policy](docs/compiled_residual.md#building-and-installing).

- Accelerate the final ordered sums in qualified dense named-linear L-BFGS
  gradient reports with a separate private native reducer. Preserve expansion,
  restoration, numerical checks, scalar/error recovery and the original checked
  fallback. This reducer itself adds no estimator option or beta/coefficient
  acceleration; the subsequent expansion extension is described above.
  Native builds require the eligible extensions; `DWD_BUILD_ACCEL=0` omits all.
  Wider tested fits benefited, while some ordinary and missing-binary fits
  cost more; see [the reporting measurements and limits](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting).

- Reuse private immutable-root views for exact contiguous row selections inside
  the explicitly invoked `fit_shared_fold_search` helper. Keep the original
  indexed-copy fallback, eligibility, public API, owned final refit and cleanup.
  Fit and CV defaults are unchanged. Document logical payload accounting and
  workload-dependent speed benefits and costs; see the
  [shared-fold guidance](docs/shared_fold_search.md).

- Reduce repeated work in final reports for eligible dense named-linear
  L-BFGS fits. Reuse compensated operand expansions within coefficient and
  gradient calculations, and share raw/joint-affine score products on the
  compatible portable path. Preserve ordered sums, numerical checks, guarded
  fallbacks, exceptional recovery and existing native scoring. Optimizer steps,
  prediction, defaults and public options are unchanged. Benefits depend on
  the workload, and small ordinary reports can cost slightly more; see
  [the reporting scope and measured costs](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting).

- Remove unsupported `sample_weight` arguments from six classifier `fit`
  methods and four standalone solvers. Even explicit `None` must be omitted;
  new keyword-only boundaries reject obsolete positional placeholders instead
  of rebinding real controls. Preserve weighted accuracy scoring, genuine
  initialization/preparation arguments, objectives and defaults. Weighted-fit
  meta-estimators such as AdaBoost remain unsupported. Document fresh-estimator
  migration for old metadata requests: cloning preserves those requests.

- Preserve fitted conic DWD/SVM problems through supported pickle, joblib and
  cloudpickle round trips. Keep primal/dual values, statistics, graph aliases
  and known data-only warm starts; rebuild native solver handles after loading
  without mutating the live cache when saving. Recognized native diagnostic
  objects become attribute-readable Python records. Unknown custom objects
  retain their serialization behavior, including errors. Copy builtin nested
  solver-option containers before solving to protect caller-owned options while
  preserving array, callback and custom-solver identities.

- Express conic SVM margins by signing constant features before the matrix
  product. This repairs false canonical hinge bounds reproduced with dense
  input in CVXPY 1.9.3's HIGHS/SCIPY routes. Preserve the mean-hinge loss, L1
  penalty, free intercept, solver choices, tolerances and finite-budget failure
  handling. This is not an AOCL/MKL attribution. Document measured total-fit
  costs and persistence limits in the
  [fitting and conic guide](docs/fitting_and_conic_persistence.md).

- Report the existing binary-classification and sparse-input contracts through
  the public scikit-learn estimator tags. Linear classifiers advertise sparse
  features; kernel DWD and its CV wrapper advertise them for the seven supported
  named feature kernels and callable kernels, whose input support remains the
  callback's responsibility. Preserve dense-only training for precomputed
  kernels, chi-squared kernels and `KernMD`, and preserve dynamic pairwise tags
  for precomputed inputs. Document the separate multiclass OvO wrapper and the
  training/query distinction. Algorithms, validation rules and defaults are
  unchanged; see [classification and input support](README.md#classification-and-input-support).

- Replace the array-valued default `lambd_vals` and `q_vals` in `GenDWDCV`
  and `KernGDWDCV` with immutable tuples of the same `np.logspace` values.
  Preserve all 10 penalty values, five loss-exponent values, their order and
  the default 50 parameter combinations. This repairs scikit-learn parameter
  display and prevents in-place mutation of shared default arrays. Replace a
  grid through the constructor or `set_params`; explicitly supplied NumPy
  arrays remain supported. Numerical fitting and stopping settings are unchanged.

- Document the scikit-learn 1.9.1 HTML decoding failure under affected
  non-UTF-8 interpreters. Tests exercised a separate UTF-8 backport. DWD ships
  the tuple-default fix for parameter representation and uses the installed
  scikit-learn renderer for HTML. See
  [estimator display and dependency limitation](docs/dependency_display.md).

- Make generic scikit-learn fitted-state checks on `KernGDWD` require the same
  coefficient, class and training-basis attributes as prediction. `cv_init()`
  alone now remains unfitted, including after it invalidates a previous fit.
  Preserve cache reuse, failed-fit cleanup, valid nonconverged predictors and
  explicit attribute checks. Solver arithmetic and defaults are unchanged.

- Use current joint/auto/auto defaults when requested affine, acceleration or
  residual-order fields are missing from an estimator, including parameter
  inspection, cloning and refitting. Preserve explicitly stored choices and
  the interpretation of existing fitted prediction state. No new solver
  arithmetic or deserialization rewrite is introduced. Cache identity and
  unsupported explicit-value checks remain enforced.

- Make `'joint'` the explicit affine-scoring default for `KernGDWD` and
  `KernGDWDCV`. Remove the redundant affine `'auto'` alias; migrate explicit
  requests and parameter grids to `'joint'`. Retain explicit `'standard'` and
  existing fitted scoring policies. Default numerical scoring is unchanged;
  private/low-level solver and `KernMD` defaults remain outside this change.

- Make `'zero'` the explicit default initialization for generalized linear and
  kernel DWD and their CV wrappers. Remove the redundant `'auto'` initialization
  alias: replace `initialization='auto'` with `initialization='zero'` in caller
  settings and parameter grids. Retain `'random'` and explicit initial
  coefficients/intercepts. The previous default already selected zero, so this
  changes the exposed spelling without changing that starting state. Other
  controls' `'auto'` choices, the objective stopping default, tolerances and
  numerical acceptance rules remain unchanged.

- Document when to opt into `stopping='optimality'`, its potential full-fit/CV
  cost, iteration-budget limits, and returned-model convergence checks. Clarify
  that CV refit diagnostics do not certify every fold. This is documentation
  only; the objective stopping default and numerical behavior are unchanged.

- Validate `GenDWDCV` and `KernGDWDCV` parameter names before `set_params`
  assigns any values. Invalid names now raise the intended parameter error
  without invoking sklearn's problematic representation of array-valued CV
  defaults. Supported requests still delegate to sklearn; numerical behavior
  and defaults are unchanged.

- Remove the public kernel `implementation` selector; the existing optimized
  implementation remains the public solver and corrected coefficient-MM
  reference computation remains private regression machinery. Remove the
  ignored `reference` keyword from `get_K_eig`. Keep explicit backend,
  initialization, acceleration, residual-order and stopping controls, and keep
  their existing optimized defaults. Reject removed constructor/CV requests
  instead of silently translating former reference settings. Require current
  optimized preparation identity before reuse. Ensemble DWD 0.4.0 requires the
  DWD 2.0.1 patch. The upgrade does not migrate old checkpoints
  or guarantee arbitrary old serialized objects. See the kernel guide for
  reference-user migration and the bounded quality/performance evidence.

- Remove the redundant public `solver_mode` selector and kernel-only
  `implicit_P` parameter. New computations always use corrected update algebra;
  linear implicit/explicit routes and the supported optimized kernel
  backends remain available. Reject removed keys in DWD CV grids before fold
  preparation. Make trailing solver options keyword-only at the documented
  boundaries so obsolete positional controls cannot shift into other options.
  Invalidate preparations without the current geometry marker. This supersedes
  the earlier retention of `solver_mode` for historical configuration inspection;
  it adds no old-object migration or promise of arbitrary historical refitting.

- Superseded earlier unreleased milestone: retired `solver_mode='legacy'`
  before new numerical work and removed known-incorrect implicit-linear and
  kernel update equations. That stage retained the selector with only `'schur'`
  accepted for new fits and left stored prediction behavior unchanged. The
  removal above replaces that transitional API; the mathematically valid
  explicit linear solver remains available with `implicit_P=False`.

- Require a nonempty compiled extension for default and explicitly accelerated
  source builds; propagate compiler and linker failures. Keep explicit
  `DWD_BUILD_ACCEL=0` portable builds, strict compiler flags, the stable ABI and
  numerical runtime fallbacks. Document the build-only setting, cached-wheel
  behavior and clean build trees when changing modes.

- Build and verify the source distribution in each CI job before creating
  the tested wheel and detached test bundle. Require all test and example
  resources, root metadata and declared documentation fixtures to match
  the checkout; preserve exact-wheel/native-route checks and full test suites.

- Require each CI job to demonstrate its expected scalar and native residual
  and scoring routes with execution witnesses and exact small-case results.
  Keep the declared platform and portable fallbacks and the full test suites;
  package numerical dispatch and acceptance criteria are unchanged.

- Remove the startup-capability test's arbitrary CPython 3.14 ceiling while
  retaining the genuine capability assertion for CPython 3.11 and later 3.x.
  Keep Python 3.12 as the primary development target and the existing private
  thread-layout adapters. Identity-policy regressions cover the guard without
  claiming execution on future interpreters; production recovery is unchanged.

- Accelerate compensated joint affine scores in final stationarity reports with
  a guarded native evaluator. Preserve ordered expanded products plus the
  intercept, numerical acceptance, exact recovery and automatic Python fallback;
  general prediction defaults, optimizer steps and stopping remain unchanged.
  Add native/Python parity, buffer, hook and rounding-mode regressions. Dense
  reporting remains quadratic; see the [measured helper costs and limits](VALIDATION.md#guarded-native-affine-reporting-isolated-evidence-2026-11-02).

- Add a private bounded readiness wait for proved owned worker startup after
  wake-up, using a high-resolution clock. If startup remains unresolved, drain
  independently qualified registered workers while keeping whole completion
  explicitly unproved. Preserve original exceptions and numerical behavior;
  the private 10 ms readiness budget does not bound full cleanup or terminate
  hung workers. See the [worker lifecycle contract](docs/compiled_residual.md#threads-and-memory).

- Protect wake-up and completion of owned default residual worker pools from
  repeated `KeyboardInterrupt` during cleanup. Preserve the original failure
  or first cleanup interruption, check supported CPython completion capabilities,
  and report unproved cleanup explicitly. Custom executor lifecycles and
  numerical computations and checks are unchanged. Cleanup adds overhead,
  especially for tiny tasks; this is not a universal interruption guarantee.

- Recover automatically from the two positive-shift implicit preparation
  failures on eligible built-in dense Schur fits, using bounded exact
  original-feature MM solves and verified nearest-even next coordinates.
  Preserve the penalty, free intercept, zero-shift guard, successful ordinary
  paths, supplied-pair validation, hooks and stopping behavior. Limit recovery
  to 16 features and finite resources; do not clip eigenvalues or promise
  convergence. Add exact-rounding, routing and public-fit regressions; see
  [the scoped evidence and material recovery cost](VALIDATION.md#bounded-exact-linear-recovery-in-local-source).

- Align implicit zero-shift linear DWD admission with the existing explicit
  column-scaled augmented-design rank check, including supplied and cached
  eigensystems and positive penalties whose computed shift underflows to zero.
  Refuse numerically unresolved rank without adding a ridge or pseudoinverse.
  Preserve positive-shift updates and subsequent numerical checks; add rank,
  exact-system, scaling, cache, cleanup and CV regressions. The extra rank
  check adds preparation cost only to formerly unchecked zero-shift fits.

- Make kernel spectral PSD admission relative to the actual spectrum, removing
  the unit-scale tolerance floor that could admit relatively indefinite tiny
  kernels. Share the policy between initial admission and spectral recovery,
  handle subnormal thresholds explicitly, and retain every positive mode.
  Preserve known-family routing and other numerical checks. Add scale,
  supplied-basis, recovery and public-fit/CV regressions.

- Preserve completed kernel fits when the final dual certificate detects a
  contradiction of its PSD assumption. Report an unavailable dual bound and
  gap with an explicit reason, preserving existing numerical-error handling,
  fitting checks and optimizer behavior. Add certificate-status, public-fit,
  CV, serialization and earlier-guard regressions.

- Build qualified standard named-RBF self kernels with the guarded norm-sum
  evaluator from the outset, mirroring one computed triangle. Preserve selective
  distance repairs and numerical acceptance checks; exact symmetry alone is not
  a PSD certificate. Version self-construction caches and preserve historical
  fitted query modes. Add construction, provenance, cache, exact-oracle and
  caller-error-policy regressions. New-fit/query rounding can change, including
  near-zero signs; construction and whole-fit costs remain workload dependent.

- Report the complete final diagnostic family in linear feature geometry for
  qualified dense named-linear fits with explicit L-BFGS. Preserve optimizer
  steps, prediction arithmetic, reconstruction checks and numerical tolerances;
  refuse unsupported feature arithmetic rather than claim stationarity. Add
  independent arithmetic, dispatch and reference-solve regressions. This
  correctness path has a material cost on very wide inputs; see
  [the kernel guide](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting).

- Construct the adaptive kernel score bound's minimum binary64 subnormal
  exactly, preventing an intentional constant from raising underflow or
  generating warnings and callbacks under caller-selected NumPy policies.
  Preserve the gradual-underflow guard, real arithmetic failures and score
  acceptance thresholds. Add focused error-policy regressions.

- Promote dense integer and Boolean operands before constructing the explicit
  linear DWD Gram matrix and column sums, preventing overflow and Boolean dot
  semantics in legacy mode. Preserve legacy floating arithmetic and original
  inputs passed to custom hooks. Add exact-system and compatibility regressions.

- Make `KernMD` retain a private feature basis and fitted kernel settings,
  preventing caller data or parameter edits from changing an already fitted
  model. Keep precomputed training shape stable without copying unused values.
  Add mutation, refit, serialization, failure-cleanup and numerical regressions.
  Feature and nested-keyword copies add fit-time work.

- Add explicit `dwd.model_selection.fit_shared_fold_search` for serial
  `KernMD` grid searches with immutable shared feature folds. Keep ordinary
  fit ownership, scoring, selection and final refit behavior; delegate
  unsupported configurations to stock search. Add ownership, failure,
  repeated-call and serialization regressions. Reuse can help wide-feature
  searches with larger grids, at the cost of storing cached subsets; no fit
  or CV default changes.

- Protect caller-owned CSR storage during named Laplacian kernel evaluation.
  Check actual coordinate ordering and copy noncanonical query or fitted-basis
  storage privately before upstream sorting and duplicate aggregation. Preserve
  read-only inputs, shared aliases and cached ordering flags without changing
  kernel arithmetic. Add ownership, validation, batching and historical-basis
  regressions. These checks and copies can add prediction cost.

- Separate standard RBF cancellation repair from NumPy error-policy selection.
  Accept bounded ordinary NumPy scalar and OrderedDict representations, honor
  nonordinary caller policies in both preliminary and repaired arithmetic, and
  preserve historical fitted models and the public direct-RBF path. Version new
  CV preparation and rebuild incompatible older caches. Add numerical, callback,
  type, batching and historical-state regressions. Newly corrected difficult
  inputs can require substantially more construction time.

- Include concrete parameter types in kernel cross-validation cache identity.
  Rebuild cached kernels when scalar or container types change, preventing
  coercive equality from reusing a matrix prepared with different parameters.
  Preserve same-type reuse and add real-fit, CV and type-sensitive kernel tests.

- Prepare sparse cross-validation input using each candidate's effective
  solver precision before duplicate coordinates are combined. Preserve
  legacy arithmetic, select matching training/scoring/refit views, and
  reuse preparation within each precision and kernel-layout policy.
  Add candidate-order, sparse-layout and direct-fit consistency tests.

- Promote sparse queries for named feature kernels to the pairwise working
  dtype before format conversion can sum duplicate coordinates. Preserve
  float32 when both operands are float32, existing cosine preparation, and
  callable/precomputed kernel contracts. Add cancellation, intermediate
  overflow, sparse-layout, batching and dtype-compatibility regressions.

- Reject nonfinite kernel decision values before returning scores or labels.
  Check unbatched results and each stored prediction batch, including overflow
  during extended-precision conversion. Preserve finite scoring arithmetic
  and caller numerical-error callbacks. Add dense/sparse, batch, callback and
  extended-precision regression coverage.

- Factor the explicit linear GenDWD MM system once and validate each built-in
  solve against the original system. Reject unresolved zero-shift designs,
  inaccurate or unresolvable actions and material objective increases without
  changing the penalty, intercept, stopping tolerances or implicit solver.
  Preserve custom-step inverse arguments. Add singular, small-penalty and
  extreme-scale checks, and separate exact gradient-reuse regression from
  inverse-versus-factor rounding comparisons. Use triangular vector solves with the retained Cholesky factor
  and vectorized residual checks to reduce recurring solve/check overhead
  without relaxing the accuracy conditions. Per-step validation retains a
  workload-dependent runtime cost.

- Normalize duplicate sparse coordinates privately for cosine kernels, using
  the pairwise working dtype before query conversion. Preserve caller buffers
  and unrelated kernels. Require refitting historical cosine models with
  duplicate training storage and invalidate affected cached Gram matrices.
  Add regression coverage for sparse layouts, mixed dtypes, historical states
  and fitted predictions.

- Preserve separate stored contributions when adaptive sparse kernel and affine
  scoring convert non-binary64 CSR data. Convert at most 128 query rows at a
  time and use the same signed binary64 entries for the ordinary product,
  error bound and accurate refinement, without modifying caller-owned buffers.
  Add regression coverage for duplicate cancellation, row-block boundaries,
  read-only inputs and extended-precision conversion behavior.

## 1.3.19 — 2026-10-26

- Promote sparse feature entries before conversions that may sum duplicates,
  including linear, named-kernel, CV, SVM and conic input boundaries. Preserve
  validation ordering, supported domains and caller-owned sparse buffers.
- Evaluate final stationarity with compensated affine scores and residual
  coefficients formed before the kernel action. Preserve optimizer steps,
  stopping tolerances, objective/dual fields and prior refusal checks. Final
  convergence labels can change when an old gradient estimate was inaccurate.
  The qualified Linux compiled K*w route is private to final reporting.
- Drain an owned native worker omitted from executor registration when an
  exact CPython post-launch startup interruption is proved. Recognize legacy
  thread-state sentinels and native thread handles by capability, without a
  minor-version allowlist. Publish cancellation before cleanup and retain the
  original exception; unidentified startup situations remain unqualified.
- Run regression tests with pytest, with psutil used by process-ownership tests.
  Both remain optional test dependencies, absent from package runtime imports.

- Separate exact workflow checks from independent eigensolver comparisons in
  four numerical regression tests. Keep exact policy, retry, initialization and
  checkpoint checks; verify independent preparations against original equations
  and high-precision references. Add deliberate fault checks without changing
  solver arithmetic, tolerances or numerical acceptance.
- Initialize native libraries before thread-restoration tests capture their
  limits, so libraries loaded on first use are included in the checked scope.
  Retain all thread-count and runtime-provenance assertions.

## 1.3.18 — 2026-10-21

- Bind the isolated recovery-import test to the package under test and verify
  its child-process import location, including when tests run outside a source
  checkout.

- Target Python 3.11 and 3.12 for routine complete-suite validation. Retain
  Python 3.13 and later Python 3.x support without dedicated full-suite runs;
  earlier 3.13 results remain historical evidence.

- Admit later CPython 3 minor releases without a version ceiling: built-in
  `math.sumprod` residual evaluation requires 3.12 or later, while independent
  Windows x86-64 compiled residual and dense-score paths require 3.11 or later.
  Require GIL-enabled builds and retain arithmetic guards, bounds and fallbacks.
  Preserve the scalar path's existing platform scope and state its extended-
  accumulation assumption separately from the recorded 3.11–3.13 test coverage.

- Add explicit `linear_action="inverse"` for optimized Cholesky MM with
  unchanged original-kernel checks and recovery. Keep factor as default;
  document setup cost, per-solve ownership, and construction memory.

- Default kernel MM acceleration to `'auto'`: use restarted optimized Schur MM
  and ordinary updates on reference, legacy and L-BFGS routes. Preserve explicit
  `None`, older serialized objects, stopping rules and numerical acceptance.
  Expose requested and effective fitted acceleration policies.
- Apply the same evaluator-identity checks to dense and sparse batched
  prediction before reusing training norms. Customized dot and guarded
  kernel helpers retain the original per-batch behavior.
- Include the documentation Makefile and Windows build wrapper in source
  archives. Restore the private kernel solver module docstring and remove
  its duplicate import.

- Reuse training norms and freshly checked canonicality within each eligible
  batched float64 CSR prediction call. Preserve custom-hook fallbacks and the
  refusal of unsafe historical fitted bases; no cache persists between calls.
- Qualify finite RBF guard masks once per scan and skip self-kernel tiles proved
  clear by the existing envelope. Preserve the flagged pairs, repair order and
  kernel arithmetic while reducing repeated checks and temporary masks.
- Canonicalize private sparse feature copies before named RBF calculations and
  automatic SOCP penalty selection. Duplicate coordinates must represent the
  same features during fitting, cache preparation, validation and prediction;
  nonfinite duplicate sums are rejected without mutating caller data.
- Require refitting an older norm/dot RBF model whose stored training features
  contain duplicate sparse coordinates. Correcting query distances alone cannot
  repair coefficients fitted to the historical incorrect Gram matrix.
- Resolve repeated source paths once per batched-prediction initialization
  check. Preserve helper provenance checks, wrapper detection and fresh checks
  after reload; no filesystem cache persists between initialization passes.
- Selectively recompute cancellation-sensitive RBF distances for eligible new
  standard Schur fits. Preserve the Gaussian kernel definition, fitted
  construction ownership, solver checks and explicit direct option. Use exact
  self-row identity to avoid unnecessary diagonal repair; retain off-diagonal
  and cross-query checks. Older fitted construction modes retain their policy.
  Bounded flag scans and ordered traversal of flagged rows avoid unnecessary
  per-row dispatch without changing which distances receive correction.

## 1.3.17 — 2026-10-19

This entry retains the release's cumulative notes. Except for the worker-cleanup
test correction below, these changes were already included in 1.3.16; see its
[historical notes](docs/history/RELEASE_NOTES-1.3.16.md).

- Make the worker-cleanup regression deterministic across numerical backends;
  production solver behavior is unchanged.

- Allow an eligible numerical failure in restarted kernel MM to take one fresh
  spectral retry from the original initialization. Preserve typed failure
  restrictions, numerical acceptance and tolerances; restart acceleration
  remains an explicit option.
- Add explicit, bounded `literal_objective_diagnostics` for the stored kernel
  objective and coefficient/intercept gradients. Keep these diagnostics separate
  from production acceptance and refuse inputs beyond their arithmetic limits.
- Emit a `FutureWarning` after validation when starting a new legacy kernel fit.
  Preserve legacy update equations and prediction from saved models.
- Run push-triggered CI on branches, avoiding a duplicate matrix for release
  tags while retaining pull-request and manual runs and the existing jobs.
- Default new kernel estimators to `affine_computation='auto'`: use joint
  affine scoring in corrected Schur mode and standard scoring in legacy mode.
  Preserve explicit standard scoring, fitted policy ownership and old serialized
  models. A guarded exact-zero shortcut avoids unnecessary rational summation;
  uncertain margins and unsupported arithmetic retain the exact fallback.
- Default estimator residual ordering to `auto`, selecting adaptive checks on
  eligible optimized Cholesky paths. Preserve explicit orders and historical
  ordering for reference, spectral, L-BFGS, legacy and supplied-eigenbasis auto
  routes. Record requested, initial and effective policies through recovery.
  Numerical acceptance and stopping settings are unchanged; finite results can differ.
- Prepare native coefficient splits once per bounded chunk, with unchanged
  arithmetic and an allocation-failure fallback. Reuse trusted residual workers
  only within one solver invocation, draining them on exit or required fallback transitions.
- Reuse training norms within eligible dense batched RBF prediction calls.
  Keep the original route for custom helpers, sparse input and unsupported
  layouts; no persistent prediction cache or new public option is introduced.
- Repair five relative links in the archived 1.3.11 release notes.

- Enable the existing guarded compiled residual and dense-score paths on
  conventional Windows x86-64 CPython 3.11 through 3.13. Keep scalar
  `math.sumprod` eligibility separate: preserve CPython 3.12 support and qualify
  conventional Windows x86-64 CPython 3.13 after direct scalar validation. Other
  runtimes use portable compensated arithmetic when the compiled path declines.
- Reuse bounded dense-score tile buffers within each eligible call. Release
  them before scalar tails or fallback, preserving row and arithmetic order.
- Validate the fitted basis once per eligible direct-RBF prediction call,
  retaining per-batch query checks, private sparse canonicalization and custom
  method behavior. The underlying direct-RBF computation remains an explicit
  option; no objective, solver tolerance or numerical acceptance rule changes.
- Own the dense or sparse training features retained by `KernGDWD` and snapshot
  its kernel parameters. Later caller edits or constructor changes no longer
  alter an already fitted predictor; refitting uses the new parameters.
- Honor post-fit `prediction_batch_size` changes in `KernGDWDCV`. Both kernel
  estimators also accept a per-call decision-function batch override without
  mutating the fitted model.
- Bound compiled residual calls and poll threaded completion so interruption
  can stop remaining chunks promptly. Join all workers and discard partial
  outputs. Native row arithmetic, worker limits and acceptance checks are unchanged.
- Skip a redundant scalar residual attempt when the original compiled helper
  has already refused an operand outside their shared arithmetic domain. Keep
  portable compensated fallback, custom-hook behavior, worker cleanup and
  numerical acceptance unchanged; no new estimator option is introduced.
- Preserve the original checkpoint failure or interruption when temporary-file
  cleanup also fails; serialize strict JSON before creating the temporary file.
- Verify exact installed wheel payloads and native/portable execution in CI,
  running copied tests outside the package source tree. Use one existing matrix
  leg for portable coverage without increasing the number of CI jobs.
- Add independent RBF primal/dual objective checks for generalized loss,
  duplicate rows, class imbalance and finite update budgets.

## 1.3.16 — 2026-10-19 (superseded by 1.3.17)

- Introduced the historical `affine_computation='auto'` and
  `residual_check_order='auto'` defaults and guarded Windows x86-64 CPython
  3.11–3.13 native arithmetic qualification described in the
  [archived tagged notes](docs/history/RELEASE_NOTES-1.3.16.md).
- The later 1.3.17 tag corrected a worker-cleanup test fixture; its production
  package code differed only in the version literal.
  The archive records historical behavior and validation, not current defaults
  or a new validation run.

## 1.3.15 — 2026-10-17

- Reject complex starting intercepts and complex values hidden in object arrays
  in initial coefficients, class labels and supplied eigendecompositions before
  conversion or optimization. Failed fits leave no usable fitted state; valid
  real inputs retain the same solver arithmetic and stopping rules.
- Preserve existing residual-check and refinement-step counts in the resumable
  CV example's compact diagnostics and validated receipts. Missing counts remain
  unknown; no additional solver checks are performed.
- Qualify the resumable CV example's winner as best among the supplied
  candidates, with exact tie counts, score gaps and numeric positions within
  matching conditional slices. Preserve first-maximum selection, duplicate
  candidate labels and final refitting; do not expand or refine the grid.

## 1.3.14 — 2026-10-15

- End maintenance of the separate Python 3.10 companion. Python 3.11 or later
  remains required; historical release assets and validation records are retained.
- Support the existing optional adaptive residual-check order with restart
  acceleration in optimized Cholesky kernel MM. The default remains
  `refinement_first`; residual acceptance, objective-descent rules, recovery
  limits and stopping tolerances are unchanged. Different accepted states can
  still change finite-iteration results.
- Let the resumable CV example set native threads per fold explicitly, with
  one thread remaining the default. Bind this setting to checkpoint identity
  and restore the caller's thread limits on success or failure. Optional paired
  memory allowances cap simultaneous fold workers; these are caller estimates,
  not measured process limits. Record the actual allocation in work summaries.
- Extend the CI matrix to Python 3.13 on Windows and Ubuntu, alongside Python
  3.11 and 3.12, for the base and optional SOCP configurations. The compiled
  helper's numerical runtime guards are unchanged.

## 1.3.13 — 2026-10-09

- Add optional `dwd.profiling.residual_profile()` to observe guarded residual
  dispatch, compiled-helper results, scalar calls, refusals and elapsed time.
  Profiles are local to the current execution context; nested scopes collect
  separately. A bounded native return is distinct from solver acceptance.
  Preserve numerical arithmetic, acceptance bounds and estimator defaults.
  See the [profiling guide](docs/compiled_residual.md#optional-residual-profiling).
- Compare named accuracy scores using exact correct/sample-count proportions
  in `run_cv` and the resumable kernel example. Preserve equal fold weights,
  first-candidate tie selection, and custom-scorer behavior. Record counts and
  reduced mean fractions without additional fitting or prediction. This fixes
  rounding-sized differences between mathematical accuracy ties; it can change
  the selected tied candidate without changing estimator mathematics.
- Extend the resumable example's compact diagnostics with the final objective,
  independent stopping flags, residuals and bounded solver/certificate/work
  summaries. Missing facts remain unknown; fitted models and histories are
  not saved. Receipt schema 2 and its aggregation identity require a new run
  directory for checkpoints from earlier source versions. See the
  [accuracy and diagnostic records](docs/resumable_cv.md#accuracy-counts-and-diagnostics).

## 1.3.12 — 2026-10-06

- Execute eligible dense accurate-score tiles in the existing optional C
  extension. Preserve the previous ordered mantissa splitting, separately
  scaled high/low products and high-then-low summation, without changing the
  score screen, solver or fitted model. The initial native score path requires
  Windows CPython 3.12 on x86-64; unsupported inputs, custom arithmetic hooks, older
  extensions and exceptional states retain the original Python evaluation.
  This also applies where fitting uses the shared score helper. See the
  [implementation and fallback scope](docs/compiled_residual.md#dense-accurate-scores).

## 1.3.11 — 2026-10-06

- Add optional `final_native_threads=1` and `--final-native-threads` to the
  resumable kernel tuning example. Keep `jobs=1` and one native thread per
  detected pool in each CV worker as defaults; larger `jobs` values remain
  bounded by unfinished folds. Apply an explicitly requested final limit only
  to full-data scaling, fresh fitting and diagnostics, record its runtime, and
  restore caller limits on success or failure. Compatible CV receipts may be
  reused when only final threads change; source/runtime identity checks remain
  strict. Estimator code and mathematics are unchanged, while different thread
  counts can change floating-point reductions. See the
  [guide](docs/resumable_cv.md#parallel-execution) and
  [validation scope](VALIDATION.md#final-refit-thread-control-in-local-source).
- Release each completed prediction kernel block before constructing the next
  one. This removes one overlapping allocation for owned multi-batch kernels,
  while preserving kernel calls, scoring arithmetic and returned values. It is
  automatic internal behavior, with no new option or general speedup claim. See
  [scope and validation](VALIDATION.md#prediction-kernel-block-lifetime-in-local-source).
- Reuse the current gradient in eligible built-in float64 `GenDWD` iterations
  with `stopping='optimality'`, avoiding a repeated loss derivative and
  transposed feature product. Preserve the iterate, MM arithmetic order,
  stopping rules and fresh final diagnostics. Keep the previous computation
  for custom hooks, other gradient dtypes and active NumPy `call` or `log`
  handlers. Ordinary duplicate warning counts can decrease. The optimization
  is automatic within the existing optional stopping mode; the objective
  default and example settings are unchanged. See
  [validation and timing scope](VALIDATION.md#linear-gradient-reuse-in-local-source).
- Add opt-in `exact_recovery='extended'` to `KernGDWD` and `KernGDWDCV`.
  Raise the complete exact-factor rank limit from 16 to 32 and exact-arithmetic
  limit from 4096 to 8192 bits, for both certification and MM recovery. Retain
  the standard default, matrix-entry and operation caps, original kernel and
  mathematical acceptance checks. Prepare exact arithmetic only after an
  ordinary MM update fails. Record the requested setting in fitted provenance,
  CV caches and resumable identities. The option extends coverage for small
  difficult kernels at additional cost; see the
  [limits and validation scope](docs/kernel_dwd.md#optional-extended-exact-recovery).
- Add opt-in `affine_computation='joint'` to `KernGDWD` and `KernGDWDCV` for
  corrected Schur fits. Include the intercept in adaptive error screening and
  accurate summation for public prediction and explicit validation stopping.
  This repairs a demonstrated decision-boundary rounding error without changing
  the DWD objective, free intercept, solver acceptance checks or zero tie rule.
  Keep `'standard'` as the default; the optional arithmetic can change rounded
  scores and validation choices, and exact ties can be substantially slower.
  Preserve the fitted choice through serialization and distinguish it in CV
  caches and resumable checkpoint identities. See the
  [scope and costs](docs/kernel_dwd.md#optional-joint-affine-scoring).
- Add opt-in `rbf_computation='direct'` to `KernGDWD` and `KernGDWDCV`.
  Evaluate named RBF kernels from coordinate differences, with range-safe
  handling of exceptional distances, to repair demonstrated self/equal-copy
  inconsistencies and large-offset cancellation. Keep `'standard'` as the
  default; direct computation can be substantially slower. Preserve the RBF
  function, DWD objective and solver acceptance checks, while allowing changed
  rounded kernels, coefficients and predictions. Retain old fitted policies
  and distinguish the requested policy in CV cache and checkpoint identities.
  See the [scope, cost and precision limits](docs/kernel_dwd.md#optional-direct-rbf-computation).
- Record startup accelerator availability, native-screen support and the loaded
  extension's path and hash in resumable CV checkpoint identities. Reject
  incompatible startup states and unidentified extension bindings before fitting.
  Existing checkpoints without this information require the matching old example
  and runtime, or a new run directory with the updated example. Estimator
  mathematics, CV selection and fitting are unchanged.
- Recover representable DWD objectives when summing finite losses or adding a
  penalty before averaging overflows. Preserve ordinary arithmetic in linear,
  legacy kernel, corrected kernel and accelerated proposal evaluations. Keep
  loss hooks, gradients, the unregularized intercept, solver updates and
  acceptance thresholds unchanged; reject nonfinite inputs and objectives
  outside floating-point range.

## 1.3.10 — 2026-10-05

- Batch dense rows requiring accurate score evaluation in bounded groups,
  preserving elementwise arithmetic and each row's summation order. Retain
  row-wise evaluation for sparse inputs, wider floating dtypes and allocation
  fallback; leave precision selection and estimator defaults unchanged.
- Add a fixed-grid binary kernel DWD tuning example that saves completed
  candidate/fold evaluations, verifies checkpoint compatibility before reuse,
  and optionally runs folds in parallel. Retain within-fold matrix preparation,
  scoring and candidate selection; final fitting remains fresh. The estimator
  mathematics and existing `run_cv` API are unchanged.
- Reuse coefficient preparation within each dense accurate score call. Keep
  product arithmetic, summation order, numerical validation and sparse scoring
  unchanged; prepare nothing when adaptive scoring needs no expanded rows.

## 1.3.9 — 2026-10-03

- Reuse the native residual core's double-length product across separate score
  and residual accumulators, retaining the original negative-product calculation
  for zero components to preserve signed-zero behavior. Acceptance bounds and
  strict floating-point arithmetic remain unchanged.
- Target baseline x86-64 instructions explicitly for Zig Windows AMD64 builds,
  rather than inheriting the build host's instruction set.
- Add opt-in `residual_check_order='adaptive'` to `KernGDWD` and `KernGDWDCV`
  for eligible Cholesky fits. Keep `'refinement_first'` as the default and retain
  the objective, equation, constraint, RKHS and descent acceptance checks.
  Adaptive ordering can change finite-iteration states; it is a separate
  numerical policy, not a convergence guarantee.
- Correct historical notebook and figure imports to the current module paths.
- Keep both gamma alternatives in the notebook kernel grid and remove
  redundant fit calls. Use one CV worker in the introductory example.
- Clear outputs only from changed notebook cells and identify retained output
  as historical. Solver equations, estimator defaults and upstream credits
  are unchanged.
- Clear unusable state after a failed `KernelScaler.fit` and invalidate an old
  kernel predictor when `cv_init` prepares new training rows. Keep the scaler's
  copied diagonal independent of caller input, recover representable scaling
  when a positive subnormal diagonal underflows in the preliminary division,
  and reject nonfinite transformed values.
- Reject complex explicit linear and legacy-kernel initial coefficients instead
  of discarding their imaginary parts. Reject nonfinite linear decision values
  instead of converting a NaN score into a class label.
- Slice cross-validation kernels according to each candidate's `kernel`
  parameter, including when the candidate selects `precomputed` instead of the
  constructor default. Reuse each fold representation across compatible settings.
- Declare setuptools for native build-policy tests, and include historical
  notebook fixtures in source distributions for the bundled documentation tests.
- Repair relative links in historical documentation and align current test
  instructions with the bundled unittest runner, retaining historical execution
  and validation claims as historical records.

## 1.3.8

- Include floating-point error allowances in ordinary residual acceptance.
- Construct exactly feasible final dual weights and conservatively bound their
  diagnostic value; report fallback or unavailable cases explicitly.
- Add independent numerical regression tests and a
  [validation matrix](docs/validation-1.3.8.md) for major retained and new changes.
- Keep the DWD objective, free intercept and stopping defaults unchanged.


## 1.3.7

- Vectorize compensated-residual error bounds with the same outward rounding
  points and numerical acceptance checks.
- Use the optional compiled row evaluator from eight rows and avoid repeated
  thread-pool inspection below 2,048 rows, where the worker budget is one.
- Add regression coverage for dispatch, fallback, input immutability, bitwise
  arithmetic agreement and conservative error bounds. Solver equations,
  unregularized intercept, stopping defaults and SOCP support are unchanged.

## 1.3.6 — 2026-09-11

- Repair roundoff-induced asymmetry in trusted internally generated RBF
  kernels while retaining strict checks for precomputed and custom kernels.
- Permit one restricted spectral restart after exhausted constrained recovery
  in eligible unaccelerated automatic MM fits without callbacks. Preserve the
  original kernel, initialization, objective, free intercept and stopping
  settings; explicit backends retain their behavior. See the
  [historical 1.3.6 notes](docs/history/RELEASE_NOTES-1.3.6.md) for the original
  validation scope and limitations.

## 1.3.5

- Clear learned state before fitting and after any failed fit across the public
  classifier interfaces. Failed refits cannot combine previous coefficients
  with replacement labels or retain an old best estimator. Keep fit signatures,
  constructor parameters and validated private precomputation caches intact.
- Add early- and late-failure regressions, changed-label refits, successful
  recovery and cache-reuse checks. No objective, solver equation or stopping
  default changes are introduced by this estimator-state repair.
- Add lazy anchor-coordinate LU recovery after original and centered Cholesky
  recovery is exhausted. It solves the same stored-kernel equations without
  jitter, symmetrization, kernel approximation or positive-mode truncation.
  Every candidate must pass the unchanged original-equation, coefficient-sum
  and RKHS checks. Healthy Cholesky fits do not construct the LU factor.
- Add a bounded midpoint residual correction when the usual intercept mean
  correction cannot satisfy the existing maximum original-equation residual.
  Each proposal still requires a fresh compensated equation, coefficient-sum
  and RKHS check. The kernel, regularization and acceptance thresholds stay
  unchanged, and the intercept remains unregularized.
- Repair a reproduced MNIST 3-versus-8 ensemble base failure at lambd=1e-12
  and gamma=1e-5/784. The exact base and all six fits in its ensemble complete
  100 MM updates without changing the sampled rows or parameters.
- Add regression coverage for midpoint rescue, unchanged mean behavior,
  constraint and RKHS rejection, fresh-check failures and floating-point limits.
  The historical 30-fit replay still returns 21 valid models and retains the
  same nine documented extreme synthetic numerical rejections.
- The preceding numerical repair passed all 390 source tests. A first-fold,
  45-pair MNIST validity screen
  at 10% base sampling completes all 270 learner fits; five use anchor recovery.
  The earlier 20% screen also completes all 270 learner fits. These extreme
  parameter checks are separate from the tuned multiclass comparison.

## 1.3.4

- Add an optional strict-arithmetic C extension for compensated residual rows,
  preserving the existing bounds, guarded domain and Python fallback.
- Parallelize independent rows within the caller's active BLAS thread budget;
  use shared read-only inputs and O(n) extra output storage.
- Preserve MM iterations, objective, unregularized intercept, recovery decisions,
  classifier API and optional linear CVXPY/SOCP support.
- Build platform-tagged CPython stable-ABI wheels with setuptools; include the
  adapted CPython arithmetic source and its complete PSF license.

## 1.3.3

- Reduce Python scalar conversion overhead in compensated residuals and reuse
  the exact coefficient-vector split within each residual evaluation.
- Add guarded native compensated residual evaluation on audited CPython 3.12
  builds. Outward error estimates cover scores and the original equation;
  uncertain checks fall back to the portable expanded calculation. A residual
  lower bound can confirm that refinement is necessary without a duplicate pass.
- Preserve equation, coefficient-sum and RKHS thresholds, including conservative
  propagation of native score uncertainty into the relative RKHS scale.
- Retain the full kernel, unregularized intercept, parameter notation, ordinary
  MM, 100-update cap, objective tolerance and optional CVXPY dependency boundary.
- Report native attempts, accepted checks, equation failures, portable declines
  and fallback time separately. Historical release evidence is preserved.

## 1.3.2

- Reject invalid/nonfinite generic-CV scalar scores before candidate selection or
  refitting; preserve valid-score aggregation and first-tie selection.
- Reduce scalar-boxing work in compensated prediction while retaining identical
  float64 product values, ordering and `math.fsum` arithmetic.
- Correct supplied-eigenpair and functional-score documentation. Solver defaults,
  objective, labels and numerical thresholds remain unchanged.
- Current source/wheel validation remains separately documented; older release
  evidence is preserved as historical.

## 1.3.1

- Recover difficult reference inverse actions through a bounded sequence of
  validated full eigenbasis representations while retaining original-equation
  checks and avoiding a Cholesky substitution.
- Add lazy, bounded exact certification of the entire stored kernel and recovery
  of the same MM function after a numerical update failure. Singular-kernel
  coefficients may use equivalent selected-column coordinates; no positive mode
  is discarded and no kernel approximation or extra regularization is introduced.
- Preserve transactional updates, callback exceptions, stopping/validation and
  acceleration counts; report recovery actions separately from accepted updates
  and the returned model's actual coefficient representation.
- Carry adaptive/compensated prediction precision through original-kernel checks,
  validation restoration, batching and serialization. Retry objective-only
  evaluation mismatches accurately and invalidate KernGDWD state before refit.
- Performance changes reuse an accepted state's objective and cap
  an optional dense query screening shortcut to one unsuccessful block per call.
  A 24-fit paired development check preserves every saved model-state byte;
  prediction medians improve modestly and fitting times remain similar.
- Retain parameter notation, native initialization policies, objective 1e-5 / cap
  100 defaults, optional acceleration and linear SOCP dependency boundaries.
- Preserve 1.3.0 artifacts and historical experiments. Nine known extreme
  synthetic nearly constant RBF fits remain unresolved; see VALIDATION.md.

## 1.3.0

- Reassess difficult original-system residuals with compensated float64
  arithmetic, checked free-intercept refinement and a bounded a-posteriori
  RKHS estimate, preserving the equations and tolerances.
- Retain reference coefficient/eigenbasis MM and optimized Cholesky; use checked
  EVD-first native preparation and remove MKL-specific subprocess recovery.
- Check exposed spectral/L-BFGS states before callbacks or validation stopping.
- Add optional restarted optimized MM, keeping ordinary MM and the original
  objective-change/cap defaults; acceleration need not improve accuracy.
- Reuse corrected linear/kernel CV preparation for identical float32 input.
- Preserve earlier repairs, source credits and optional linear CVXPY support;
  see the [validation history](VALIDATION.md) for the scope of recorded checks.

## 1.2.1

- Refine an overconservative initial quadratic-form check with compensated
  float64 accumulation and an error bound for the actual observed initial state.
- Reuse the checked initial scores and quadratic without changing coefficients,
  free intercept, objective, parameter notation or the 1e-5 / 100-update defaults.
- Preserve rejection of unsafe cancellation, overflow and unsupported arithmetic;
  expose the check method, bounds, discrepancy and cost. Add focused regressions.
- Keep earlier distributions and numerical-runtime measurements separate; see
  [VALIDATION.md](VALIDATION.md) for current verification status.

## 1.2.0

- Provide repaired reference and optimized kernel implementations in one package.
  Reference retains checked eigendecomposition, coefficient-MM and native Gaussian
  initialization; optimized uses original-system Cholesky/refinement and zero init.
- Validate full eigenpairs and original constrained solves, with bounded isolated
  numerical recovery and unchanged full-kernel objective/free intercept.
- Preserve original objective-change stopping defaults, optional SOCP and earlier
  compatibility repairs. Historical release-specific evidence remains separate.

## 1.1.1

- Recheck actual eigenvalues after auto Cholesky conditioning rejection; retain
  Cholesky under a conservative two-norm bound or use guarded spectral fallback.
- Preserve explicit Cholesky rejection and all spectral/input/final safeguards.
- Use explicit SciPy EVR for computed solver and kernel/linear helper spectra.
- Report requested/effective backends, condition checks, fallback and setup times.
- Preserve all 1.1.0 defaults and objective/parameter notation; add 16 regressions.

## 1.0.5+audit1 (release candidate)

- Add opt-in `solver_mode='schur'` corrections for linear and kernel generalized
  DWD; retain the explicitly documented legacy default for reproducibility.
- Accelerate loss/gradient evaluation and repeated matrix operations; expose
  iteration, initialization, stopping and convergence diagnostics.
- Correct built-in cross-validation, kernel mean-difference and binary-label
  handling; vectorize equivalent SOCP constraints and validate solver outcomes.
- Promote corrected-mode features to float64 before forming Gram matrices;
  normalize array-like kernels, invalidate incompatible caches and reject
  nonfinite objectives. No kernel ridge or eigenvalue truncation is introduced.
- Ship self-contained attributed regression fixtures, independent numerical
  tests and updated packaging metadata. See [AUDIT_CHANGES.md](AUDIT_CHANGES.md)
  for result-changing corrections and numerical limitations.

## 1.0.5+compat2

- Replace the removed `np.int` alias with Python's `int` for prediction indices.
- Put scikit-learn classifier mixins before `BaseEstimator`, following current
  estimator method-resolution-order requirements.
- Declare floating output for the vectorized DWD loss gradient so NumPy does not
  infer an integer output type from an integer-valued first element.
- Add compatibility regression tests and provenance documentation.

## 1.0.5 (2022-01-10)

- Since `cvxpy` is not supported on all platforms, make this an optional dependency installable via `pip install dwd[socp]`. 
- Remove the import aliases from `dwd.__init__`; one must explicitly import the solver to be used:
  - `dwd.socp_dwd.DWD` (only in `dwd[socp]`)
  - `dwd.gen_dwd.GenDWD`
  - `dwd.gen_kern_dwd.KernGDWD`
- Add `socp` requirements to readthedocs config so `dwd.socp_dwd` autodoc will generate.

## 1.0.4 (2021-11-19)

- Rolled back dependency version pinning to restore compatibility with other versions of Python.
- Remove matplotlib from the main dependencies.

## 1.0.3 (2021-11-19)

- Add sphinx documentation for Read the Docs
- Fix pyproject.toml to match [PEP 621](https://www.python.org/dev/peps/pep-0621/)
- Add property dwd.direction
- Pin dependency versions to support `pip install --require-hashes`

## 1.0.2 (2021-05-28)

- Convert `README.rst` to markdown to be more consistent with other documentation.
- Reverted changes to `solve_dwd_socp` to make it [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming), as it caused DWD to stall in new versions of cvxpy.

## 1.0.1 (2021-05-17)

- Fix warnings and errors regarding deprecation of [sklearn private API](https://scikit-learn.org/stable/whats_new/v0.22.html#clear-definition-of-the-public-api) so that sklearn 0.22.x and higher are supported
- Fix warnings from cvxpy that `solve_dwd_socp` was not [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)

## 1.0.0 (2019-08-17)

- First release!
- Published to PyPI on 2021.05.17 
