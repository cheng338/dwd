# DWD 1.3.19 — October 26, 2026

This maintenance release corrects sparse input promotion, improves final
stationarity reporting, and handles proved native-worker startup interruptions.
It preserves the DWD objective, unregularized intercept, optimizer steps,
stopping tolerances and numerical acceptance rules. Corrected sparse arithmetic
can change a fit; more accurate final gradients can change a reported convergence
label without changing the optimizer's stopping decision.

## Correctness and ownership

- Promote sparse entries before conversions can sum duplicate coordinates in
  integer or narrower floating-point arithmetic. Cover linear and named-kernel
  fits, CV, prediction, SVM and conic boundaries while preserving caller-owned
  data, validation ordering and supported domains. Automatic SOCP penalty
  selection retains its existing float32 distance/range policy after coalescing.
- Evaluate final stationarity using compensated affine scores and residual
  coefficients formed before the kernel action. Preserve objective and dual
  reporting, optimizer arithmetic and prior refusal checks. The qualified Linux
  compiled K*w route is private to final reporting; it does not enable general
  Windows-only compiled prediction paths on Linux.
- Drain an owned native worker omitted from executor registration only when
  the supported post-launch interruption is proved. Publish cancellation before
  cleanup and retain the original exception. Recognize legacy thread-state
  sentinels and newer native thread handles by capability; unqualified startup
  situations remain explicit rather than being treated as safely recovered.

## Regression tests

Run the suite with pytest. Pytest and psutil are optional test dependencies and
are not imported by package runtime code. Numerical regression tests distinguish
exact policy, retry, initialization and checkpoint contracts from independently
computed eigensystems. Independent preparations are checked against original
equations and high-precision references. Thread-restoration tests initialize
native libraries before capturing their limits and retain the existing runtime
checks.

## Validation and compatibility

The pre-release Python 3.12.15 source snapshot completed all 1,245 selected DWD
pytest items on each current target: 1,240 passed and five skipped on Windows
AOCL 5.3; 1,231 passed and 14 skipped on Robust Linux MKL 2026.2 with AUTO.
Both runs had zero failures and direct exit zero, with unchanged source
readbacks. Platform/runtime skips are retained and are not passes. These are
source-validation results, separate from release-wheel and hosted-CI checks.

Python 3.11 and later Python 3.x releases remain supported. Complete-suite CI is
configured for 3.11 and 3.12; the current-target complete runs above used 3.12.
Focused startup checks on 3.11 and 3.13 have their own recorded scope. Forward
capability checks do not imply testing every future interpreter or arbitrary
asynchronous-interruption window. See [VALIDATION.md](VALIDATION.md) for the
evidence and limits. No numerical-library bug or general performance improvement
is claimed, and the canceled Resource18 benchmark remains canceled.

Install `dwd-1.3.19-cp311-abi3-win_amd64.whl` on compatible Windows AMD64 CPython,
or `dwd-1.3.19-py3-none-any.whl` without the optional helper. Wheel loading
compatibility and numerical-path qualification remain separate.

This remains a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license are
retained. See the [changelog](CHANGES.md), [installation](README.md#installation)
and [1.3.18 release notes](docs/history/RELEASE_NOTES-1.3.18.md).
