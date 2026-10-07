# DWD 1.3.11 — October 6, 2026

This release packages the numerical safeguards, optional precision settings and
workflow improvements developed since 1.3.10. The DWD objective, practical
unregularized intercept, regularization convention and default stopping settings
are preserved.

- Add opt-in direct RBF computation, joint affine scoring and extended exact
  recovery. Keep their standard defaults and existing mathematical acceptance
  checks; the optional policies can change rounded results and increase cost.
- Recover representable objectives after intermediate overflow, reuse eligible
  current gradients for optimality stopping, and release completed prediction
  kernel blocks before constructing the next batch.
- Strengthen resumable accelerator identity and add an explicit final-refit
  native-thread option, retaining existing CV and final-thread defaults.

The main release requires Python 3.11 or later. Its Windows AMD64 native wheel
is `dwd-1.3.11-cp311-abi3-win_amd64.whl`; the portable alternative is
`dwd-1.3.11-py3-none-any.whl` and has the same Python requirement. See the
[installation instructions](README.md#installation), [changelog](CHANGES.md)
and [validation scope](VALIDATION.md). The historical 1.3.10 timing investigation
did not establish a persistent package regression; its qualifications remain
in the separate [1.3.10 record](docs/validation-1.3.10.md).

Wheel compatibility and residual arithmetic are separate: CPython 3.11 can
load the Windows `cp311-abi3` extension but uses DWD's portable compensated
residual check. The guarded compiled or built-in `math.sumprod` paths remain
eligible only on conventional CPython 3.12; a portable wheel there can still
use `math.sumprod`. NumPy/SciPy BLAS acceleration is independent. The separate
`cp310-none-any` compatibility asset is portable and restricted to CPython 3.10;
its source overlay is distributed separately from the main source archive.

This is a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license are
retained. Earlier [1.3.9 release notes](docs/history/RELEASE_NOTES-1.3.9.md) remain
available as a historical record.
