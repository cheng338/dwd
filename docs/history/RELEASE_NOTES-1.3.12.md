# DWD 1.3.12 — October 6, 2026

This release reduces Python overhead in dense accurate kernel scoring. Eligible
tiles use a separate entry in the existing optional C extension, preserving the
mantissa split, separately scaled high/low products and high-then-low summation.
Exact exponent-field operations avoid library calls for normal results; other
ranges retain the library operations. The DWD model, coefficients, intercept,
solver acceptance checks, precision screen and estimator defaults are unchanged.

The new score path is automatic on conventional Windows CPython 3.12/x86-64 with the
extension available. Missing or older extensions, unsupported layouts/dtypes,
custom arithmetic hooks, active NumPy call/log handlers and exceptional states
retain the original Python path. Single-row tails and error ordering are
preserved. The existing residual evaluator is unchanged. Because the score
helper is shared, fitting and validation can also use the new execution path.

For the affected MNIST 2/3 ensemble at its unchanged selected parameters,
matched prediction timing decreased from 96.707 to 63.084 ms (34.8%). The other
five DWD/ensemble cases showed only small fluctuations. These Windows AMD/AOCL
results are workload-specific, with recorded background-load screening; they
are not a general speedup claim. Timing used the preceding private build,
before the Windows-only eligibility predicate; Windows arithmetic is unchanged,
and the final release wheels were not retimed. Saved scores/labels and fresh-fit controls
matched the previous implementation. See [validation](../../VALIDATION.md).

The main release requires Python 3.11+. Windows AMD64 wheels use
`dwd-1.3.12-cp311-abi3-win_amd64.whl`; `dwd-1.3.12-py3-none-any.whl` omits the
extension. The ABI permits loading on supported CPython 3.11+ versions, while
the residual and dense-score runtime gates remain narrower. CPython 3.11 uses
portable residual and score calculations; a portable wheel on CPython 3.12 can
still use guarded built-in `math.sumprod` residuals. NumPy/SciPy BLAS remains
independent. The separately built `cp310-none-any` companion and reproducible
compatibility source bundle remain restricted to CPython 3.10; main source is
not changed to accommodate that interpreter.

This is a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license
are retained. See the [changelog](../../CHANGES.md), [installation](../../README.md#installation)
and [previous release notes](RELEASE_NOTES-1.3.11.md).
