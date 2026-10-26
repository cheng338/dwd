# DWD 1.3.18 — October 21, 2026

This release corrects sparse feature handling and cancellation-sensitive RBF
distances, reduces repeated guarded prediction work, and enables automatic
restarted MM on eligible kernel fits. The DWD objective, unregularized intercept,
solver tolerances and numerical acceptance rules are preserved. Corrected
distances and a different optimization trajectory can change rounded scores or
finite-iteration results.

## Correctness and computation

- Canonicalize private sparse copies before named RBF calculations and automatic
  SOCP penalty selection. Duplicate coordinates now represent the same features
  throughout fitting and prediction; invalid duplicate sums are rejected without
  changing caller data. Older norm/dot RBF models fitted with duplicate training
  coordinates must be refitted because their stored coefficients came from an
  incorrect Gram matrix.
- Selectively recompute cancellation-sensitive distances for eligible new
  standard RBF fits. Preserve the Gaussian kernel and fitted construction policy,
  with an explicit direct-distance option and the existing numerical safeguards.
- Reduce repeated guard scans, source-path resolution and canonicality checks;
  reuse training norms within eligible dense or sparse batched prediction calls.
  Custom helpers retain checked fallbacks. No persistent prediction cache is
  introduced.

## Defaults and options

- `acceleration='auto'` selects restarted optimized Schur MM, including checked
  spectral solves. Reference, legacy and L-BFGS routes retain ordinary updates.
  Explicit `None` and older serialized policies remain supported.
- `linear_action='factor'` remains the default. Explicit `'inverse'` is available
  for optimized Cholesky MM, with unchanged original-equation checks and recovery.
  Its setup work and additional dense storage are documented; it is not chosen
  automatically from workload size.

## Python support and validation

Python 3.11 and later Python 3.x releases are supported. Complete-suite CI
targets 3.11 and 3.12. Python 3.13 and later releases do not require separate
complete-suite runs. Guarded scalar residual evaluation accepts CPython 3.12+;
the independent compiled residual and dense-score routes accept Windows x86-64
CPython 3.11+. GIL, floating-point and operand guards remain, together with
portable fallbacks. Forward eligibility does not imply tests on future runtimes.

Recorded application checks collected 1,069 DWD and 947 ensemble-DWD test
methods on Windows Python 3.12, with no failures or errors and five and one
conditional skips. Focused forward-runtime checks passed on actual Python 3.11
and 3.12 with native and portable arithmetic. Simulated future version identities
tested dispatch, not future interpreter arithmetic. Release-artifact verification
and hosted CI are separate from those application checks. See
[VALIDATION.md](VALIDATION.md) for scope and measured costs; no universal speedup
is claimed.

Install the native `dwd-1.3.18-cp311-abi3-win_amd64.whl` on compatible Windows
AMD64 CPython, or use `dwd-1.3.18-py3-none-any.whl` without the optional helper.
Numerical path eligibility is separate from wheel loading and NumPy/SciPy BLAS.

This remains a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license are
retained. See the [changelog](CHANGES.md), [installation](README.md#installation)
and [1.3.17 release notes](docs/history/RELEASE_NOTES-1.3.17.md).
