# DWD 1.3.14 — October 15, 2026

This release extends optional adaptive residual checking to restarted kernel
MM and adds explicit resource controls to the resumable CV example. The DWD
objective, free intercept, residual acceptance, restart rules, recovery limits
and stopping tolerances are unchanged. Estimator and example defaults remain
unchanged.

## Optional adaptive checking with restarted MM

Eligible optimized Cholesky fits can combine `residual_check_order='adaptive'`
with `acceleration='restart'`. The existing adaptive policy can avoid a
speculative correction after repeated evidence that accurate checking already
accepts the untouched state. Every returned state still passes the original
numerical requirements. Reference, explicit spectral, legacy and L-BFGS paths
remain ineligible. The default is still `residual_check_order='refinement_first'`.

The retained option improved complete fitting time on the tested difficult
cases, but offered no consistent benefit on a healthy case. Changing the order
can choose a different valid finite-iteration state; universal bitwise equality
or speed improvement is not promised. See the [kernel guide](docs/kernel_dwd.md)
and [validation record](VALIDATION.md#version-1314-development-validation).

## Optional CV resource controls

The resumable kernel example accepts `cv_native_threads` / `--cv-native-threads`,
defaulting to one native thread per fold. Optional paired
`cv_memory_budget_mib` and `cv_worker_memory_mib` values cap simultaneous fold
workers using caller-supplied planning estimates. They are not measured peaks,
live memory admission or operating-system memory limits. The Python interface's
`jobs=-1` accounts for native threads before applying fold and memory caps;
explicit positive `jobs` values remain the caller's choice.

CV native threads enter checkpoint identity, so changing that setting requires
a new run directory. Worker counts and memory allowances may change during a
compatible resume. Final fitting remains separately controlled by
`final_native_threads` and needs its own memory assessment. Caller thread limits
are restored on success or failure. Source identity changes in this release
also prevent silently reusing checkpoints from older example sources. See the
[resumable CV guide](docs/resumable_cv.md#parallel-execution).

## Python support and distribution

Python 3.11+ is required. Windows AMD64 wheels use
`dwd-1.3.14-cp311-abi3-win_amd64.whl`; `dwd-1.3.14-py3-none-any.whl` omits the
optional extension. The extension sources, strict compiler flags and numerical
runtime guards are unchanged. Loading an ABI-compatible extension does not
enable all accelerated paths: the qualified residual and accurate-score paths
retain their narrower runtime conditions. NumPy/SciPy BLAS is independent.

Python 3.10 companion maintenance has ended. Historical companion assets and
their validation records remain available; no Python 3.10 wheel is produced
for this release. Main functions were not changed to accommodate Python 3.10.
The CI matrix now includes Python 3.11, 3.12 and 3.13 on Windows and Ubuntu,
with base and optional SOCP configurations. Source, built-artifact and CI
checks have distinct scopes; see [validation](VALIDATION.md).

This is a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license
are retained. See the [changelog](CHANGES.md), [installation](README.md#installation)
and [previous release notes](docs/history/RELEASE_NOTES-1.3.13.md).
