# DWD 2.0.1

This patch corrects two regression tests present in the public 2.0.0 source
tag. The build tests now resolve the platform-error
exception for the installed setuptools layout. The kernel-fit affinity test
checks portable compensated recovery on Linux CPython 3.11 and the one-worker
limit on platforms where the guarded native residual evaluator runs.
The estimator, solver and native-extension implementations match the 2.0.0
sources. [2.0.0 CI results](VALIDATION.md#dwd-200-source-tag-tests).

## Upgrading from 1.3.19

Install DWD 2.0.1 with Ensemble DWD 0.4.0 when using the ensemble estimators.
Earlier companion versions can forward removed fitting keywords.

- Remove `solver_mode`, kernel `implementation` and kernel `implicit_P` from
  constructors, solver calls and CV grids. Linear `implicit_P` remains supported.
  Former reference-solver users should review the explicit settings that
  reproduce their initialization and solver policies. [Solver migration](docs/kernel_dwd.md#removed-public-solver-controls).
- Replace `initialization='auto'` with `'zero'` and kernel
  `affine_computation='auto'` with `'joint'`. Omit training `sample_weight`,
  even `None`, and pass trailing fit controls by keyword. Weighted accuracy
  remains available via `score`. [Fitting arguments and metadata](docs/fitting_and_conic_persistence.md).
- Start a fresh run directory for source-bound checkpoints. Historical
  estimators retain their documented prediction and refit limits.
  [Caches and saved objects](docs/kernel_dwd.md#paired-releases-caches-and-saved-objects).

The November 19 native-wheel DWD suites passed 6,184 tests on Windows (71 skipped)
and 6,231 on Linux (24 skipped). They used the pre-renumbering DWD 1.3.20 and
Ensemble DWD 0.3.16 artifacts. The 256-by-2,048 dense linear L-BFGS workload
took longer for final feature-based diagnostics. The second fresh fit added a
paired median of 137 ms on Windows and 296 ms on Linux against 1.3.19.
The larger complete CV, final-fit and scoring workflow saved 1.30 s and 3.39 s
respectively. [Comparison and artifact identities](VALIDATION.md#november-19-2026-paired-release-comparison).

## Input ownership

- Document when to retain `KernelScaler.transform`'s default `copy=True` and
  when its in-place option is appropriate. A refused in-place transformation
  may already have changed caller storage. No numerical or default change.

## Integer sparse coordinates

Signed and unsigned integer sparse inputs now sum duplicate coordinates
exactly before converting each sum once to float64. For example,
`[2**53 + 1, -2**53]` at one coordinate represents `1`, rather than losing the
unit during entry-wise conversion. The rule applies at existing sparse input
boundaries, including linear prediction and integer sparse precomputed queries;
precomputed training remains dense-only. Linear prediction avoids an extra
conversion when query storage is verified unique, retaining ordinary
dot-product conversion. Floating and Boolean policies,
callable returned kernels and direct score-helper arithmetic are unchanged.

This strengthens the previous input contract and can change affected fits,
scores and labels. It cannot recover already overflowed or rounded data or
represent every large integer exactly in float64. Historical fits and results
are not rewritten; refit from original integer storage to use the new training
semantics, subject to existing historical-model guards. Exact accumulation
can add work for exceptional duplicate inputs; no universal speed claim is made.
See [input details](docs/kernel_dwd.md#integer-sparse-coordinate-inputs).

## Prediction validation and shared-fold search

Eligible dense batched `KernGDWD` predictions using guarded norm-sum RBF now
reuse the query validation already performed by the public prediction call.
Fresh training norms and checks of storage and helper bindings qualify this
reuse; other cases retain the checked path. The kernel arithmetic, validation
rules and caller error-policy handling are unchanged, and no validation cache
survives the call. See [kernel prediction](docs/kernel_dwd.md#kernels-batching-and-caches).

The explicit `fit_shared_fold_search` helper retains its private fold-token
registry through the ordinary final refit and clears it during helper exit
cleanup. This removes the earlier pre-refit release. Logical cache-use
statistics remain available, and the returned public estimator still owns its
training features. Registry storage can overlap the final owned copy; the
lifetime change is not a universal speedup. See
[shared-fold ownership](docs/shared_fold_search.md#state-and-ownership).

These changes add no public setting and preserve prediction and CV defaults.
The shared-fold route remains opt-in. Performance depends on the workload and
runtime; neither change establishes a universal whole-call speedup. The published
1.3.19 release notes below describe the earlier release, not these source changes.

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
AOCL 5.3; 1,231 passed and 14 skipped on Linux with MKL 2026.2 and AUTO.
Both runs had zero failures and direct exit zero, with unchanged source
readbacks. Platform/runtime skips are retained and are not passes. These are
source-validation results, separate from release-wheel and hosted-CI checks.

Python 3.11 and later Python 3.x releases remain supported. Complete-suite CI is
configured for 3.11 and 3.12; the current-target complete runs above used 3.12.
Focused startup checks on 3.11 and 3.13 have their own recorded scope. Forward
capability checks do not imply testing every future interpreter or arbitrary
asynchronous-interruption window. See [VALIDATION.md](VALIDATION.md) for the
evidence and limits. No numerical-library bug or general performance improvement
is claimed. The remaining resource-benchmark timings were canceled and were not used
in this comparison.

Install `dwd-1.3.19-cp311-abi3-win_amd64.whl` on compatible Windows AMD64 CPython,
or `dwd-1.3.19-py3-none-any.whl` without the optional helper. Wheel loading
compatibility and numerical-path qualification remain separate.

This remains a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license are
retained. See the [changelog](CHANGES.md), [installation](README.md#installation)
and [1.3.18 release notes](docs/history/RELEASE_NOTES-1.3.18.md).
