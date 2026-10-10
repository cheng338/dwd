# DWD 1.3.13 — October 9, 2026

This release corrects rounding-sized accuracy ties in cross-validation and
adds diagnostic records for investigating difficult fits. The DWD objective,
free intercept, fitting arithmetic, numerical acceptance checks and estimator
defaults are unchanged.

For explicitly named `scoring='accuracy'`, `run_cv` now ranks candidates using
exact correct/sample-count proportions with equal fold weights. A mathematical
tie selects the first candidate. Custom scorers retain their existing scalar
aggregation. This can change the selected candidate when the previous rounded
means incorrectly distinguished an accuracy tie; it performs no additional
fitting or prediction.

The resumable kernel example uses the same exact accuracy comparison and saves
the counts, final objective, independent stopping flags, residuals and compact
solver summaries. Missing diagnostics remain unknown. Receipt schema 2 and
its aggregation identity require a new run directory for earlier checkpoints;
existing records are not silently converted. See the
[example guide](docs/resumable_cv.md#accuracy-counts-and-diagnostics).

The new optional `dwd.profiling.residual_profile()` context observes guarded
native residual dispatch, compiled-helper results, scalar calls, refusals and
elapsed time. A bounded native return is distinct from solver acceptance.
Profiles are separate from estimator state and caches, and cover only the
current thread and asyncio task. The context reads clocks and records counters
when enabled; it is not an overhead-free timing method. See the
[profiling guide](docs/compiled_residual.md#optional-residual-profiling) and
[validation scope](VALIDATION.md).

The main release requires Python 3.11+. Windows AMD64 wheels use
`dwd-1.3.13-cp311-abi3-win_amd64.whl`; `dwd-1.3.13-py3-none-any.whl` omits the
optional extension. The extension, strict compiler flags and narrower numerical
runtime guards are unchanged. An ABI-compatible interpreter does not
necessarily use native numerical acceleration. NumPy/SciPy BLAS is independent.

The separate CPython 3.10 companion remains at
[version 1.3.12](https://github.com/cheng338/dwd/releases/tag/v1.3.12) and does not
include these changes. Its refresh is deferred; this release does not change
main source to accommodate Python 3.10.

This is a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license
are retained. See the [changelog](CHANGES.md), [installation](README.md#installation)
and [previous release notes](docs/history/RELEASE_NOTES-1.3.12.md).
