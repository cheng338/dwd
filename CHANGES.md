# Updates

This changelog covers the fork of [slicersalt/dwd](https://github.com/slicersalt/dwd)
maintained at [cheng338/dwd](https://github.com/cheng338/dwd). Subsequent fork
development is guided by Chang Cheng. The original implementation by Iain
Carmichael, upstream maintenance by David Allemang and Kitware, and the MIT
license remain credited in the [README](README.md) and [license](LICENSE.txt).

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
  see [release notes](RELEASE_NOTES.md) and [validation](VALIDATION.md).

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
