# Updates

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
- Remove the import aliases from `dwd.__init__`; one must explictly import the solver to be used:
  - `dwd.socp_dwd.DWD` (only in `dwd[socp]`)
  - `dwd.gen_dwd.GenDWD`
  - `dwd.gen_kern_dwd.KernGDWD`
- Add `socp` requirements to readthedocs config so `dwd.socp_dwd` autodoc will generate.

## 1.0.4 (2021-11-19)

- Rolled back dependency version pinning to restore compatibility with other versions of Python.
- Remove matplotlib main dependency

## 1.0.3 (2021-11-19)

- Add sphinx documentation for Read the Docs
- Fix pyproject.toml to match [PEP 621](https://www.python.org/dev/peps/pep-0621/)
- Add property dwd.direction
- Pin dependency versions to support `pip install --require-hashes`

## 1.0.2 (2021-05-28)

- Convert `README.rst` to markdown to be more consistent other documentation.
- Reverted changes to `solve_dwd_socp` to make it [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming), as it caused DWD to stall in new versions of cvxpy.

## 1.0.1 (2021-05-17)

- Fix warnings and errors regarding deprecation of [sklearn private API](https://scikit-learn.org/stable/whats_new/v0.22.html#clear-definition-of-the-public-api) so that sklearn 0.22.x and higher are supported
- Fix warnings from cvxpy that `solve_dwd_socp` was not [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)

## 1.0.0 (2019-08-17)

- First release!
- Published to PyPI on 2021.05.17 
