# Compiled residual arithmetic

The package optionally builds a small C extension that accelerates the existing
compensated kernel residual check. It changes how arithmetic is executed, not
the kernel DWD model, MM equations, free intercept, loss, regularization,
initialization, stopping tolerances, or iteration budget.

## Numerical contract

The core uses the same ordered double/triple-length accumulation as CPython
3.12's `math.sumprod`, with the Dekker product implementation and strict compiler
floating-point flags. Scores and original-equation residuals have separate
accumulators. The residual includes the original right-hand side, intercept and
shift product; it is never formed by subtracting an already-rounded score.

The Python layer retains the previous outward error bounds, score precision
screen, equation checks, coefficient-sum check and RKHS accuracy requirement.
When compiled values are available, the Python layer evaluates those bounds
on arrays with the same outward rounding points. The C core additionally checks
its arithmetic domain and rounding behavior.
Uncertain or unsupported evaluations retain the prior Python/native or portable
fallback. No numerical tolerance is relaxed. The current native-screen runtime
gate remains conventional CPython 3.12. On Windows, CPython 3.11 can install
and load the `cp311-abi3` wheel, but DWD still uses its portable compensated
residual check; the same residual guard excludes CPython 3.13 and later.
On CPython 3.12, omitting the extension can still leave the guarded built-in
`math.sumprod` residual path available. The wheel ABI and DWD's residual
arithmetic guard are separate from NumPy/SciPy BLAS acceleration.

The adapted arithmetic source includes the complete Python Software Foundation
license in `dwd/CPYTHON-LICENSE.txt`. Original DWD attribution and its MIT license
remain intact.

## Threads and memory

Independent row ranges can execute concurrently because the extension releases
the GIL. The worker count does not exceed the smallest active BLAS thread limit,
the logical CPU count, or one worker per 1,024 rows. If no BLAS budget is visible,
it uses one worker. Kernels with fewer than eight rows use the existing Python
path. Compiled checks below 2,048 rows use one worker without inspecting loaded
BLAS libraries; larger checks inspect the caller's active budget.

Thus an external `threadpool_limits(8)` context permits up to eight arithmetic
workers. GridSearchCV processes configured with one BLAS thread remain single
threaded inside this check. This does not introduce another CV engine or change
the ensemble package. The numerical code uses the same read-only kernel and
O(n) additional output storage; workers do not copy the Gram matrix or create
child processes.

## Optional residual profiling

Version 1.3.13 provides an opt-in context for diagnosing which guarded native
residual route is used:

```python
from dwd.profiling import residual_profile

with residual_profile() as profile:
    model.fit(X_train, y_train)

observations = profile.as_dict()
```

The profile records native calls, bounded returns, refusals, exceptions and
scalar `sumprod` calls, including work completed before a refusal. It separates
unsupported arithmetic, exits before dispatch, compiled-helper results and
scalar fallback. `compiled_helper` describes the existing helper call; a refusal
can mean a missing extension, a small input or an unsupported case. Its elapsed
time is not a measurement of raw C arithmetic alone.

A `bounded_return` means residuals and error allowances were returned. The
solver still checks the original equations afterward. Use its existing
`native_residual_acceptances` diagnostic for that later acceptance decision.
The profiler excludes portable residual work after a native refusal and does
not measure prediction scoring or the entire fit. Route times partition the
native-call time; compiled-helper time is nested and must not be added again.

Each context starts a new profile. Nested contexts collect separately, and the
outer context resumes after the inner one exits. Exceptions propagate normally.
Profiles observe only their own thread and asyncio task; child threads, tasks
and process workers need their own contexts and explicit result collection.
An empty parent profile does not establish that parallel workers performed no
accurate checks. `as_dict()` returns a separate JSON-compatible snapshot.

Profiling reads clocks and records counters, so use it for diagnosis rather
than treating its times as an overhead-free benchmark. Outside an active
context, the instrumentation reads no clocks and creates no per-call profile
record. It changes no estimator parameter, arithmetic, tolerance or fitted
state, and replaces no global numerical function. Profile results are not
stored in fitted models or estimator caches.

## Dense accurate scores

The same extension provides a separate `score_rows` entry for bounded dense
tiles already requiring accurate kernel scoring. It uses the previous
`frexp`/Dekker/`ldexp` product order and feeds every high product, followed by
every low product, into the finite `fsum` reduction. It does not replace this
calculation with an ordinary dot product, change a numerical bound or alter the
model. The residual evaluator above remains separate and unchanged.
For normal binary64 values whose scaled result remains normal, exponent-field
changes perform the exact normalization/scaling; zero, subnormal and exceptional
cases retain the library `frexp`/`ldexp` path and its rounding points.

The score path initially requires Windows CPython 3.12 on x86-64 with
binary64 round-to-nearest arithmetic and gradual underflow. Other operating
systems retain the existing Python calculation until their native builds are
qualified. This restriction applies only to the new score entry; the existing
residual runtime guard is unchanged. It handles the
existing C/F-contiguous float64 kernels through bounded contiguous tiles, with
at most eight rows and O(n_training) scratch. It releases the GIL without
starting additional threads. Single-row tails, unsupported layouts/dtypes,
custom arithmetic hooks, active NumPy `call`/`log` handlers and missing or older
extensions keep the original Python path. A native refusal or scratch failure
releases its temporary storage before retrying the original evaluation and
error ordering.

This is an internal execution optimization, with no new estimator setting.
The shared helper is used by prediction and by some fitting/validation score
calculations. Sparse scores and the optional joint-affine calculation retain
their existing arithmetic. Benefits depend on how many dense rows require
accurate scoring; ordinarily accepted dot products do not use this entry.

## Building and installing

An accelerated wheel is platform specific and is tagged
`cp311-abi3-<platform>`, not `py3-none-any`. The stable ABI permits loading on
CPython 3.11 and later; the narrower numerical runtime gate above still applies.
No compiler or tool download is invoked at runtime, and the extension does not
depend on the NumPy C API.

Normal source builds use setuptools and a locally available C compiler. The
extension is optional: a missing compiler can produce a functioning package
using the previous arithmetic. Builds must use strict
floating-point flags. MSVC uses `/fp:strict`; GCC/Clang builds disable implicit
contraction, reassociation and fast-math.

`DWD_BUILD_ACCEL=0` explicitly requests a portable build. Always use a separate
clean source/build directory for portable and compiled wheel builds. For the
audited Windows release build, `DWD_BUILD_ZIG` can name a local Zig executable.
That explicit override fails the build if compilation fails; it never downloads
a compiler or silently produces a wheel labeled as accelerated without the
compiled helper.

The performance improvement is workload dependent. It targets models requiring
compensated residual checks, particularly large ill-conditioned kernel systems.
It is not a claim that every DWD fit, reference eigendecomposition or CV search
becomes faster by the same factor. The default allows at most 100 MM updates per attempt and can stop earlier on
objective change; numerical convergence is reported separately. An eligible
[automatic numerical restart](kernel_dwd.md#automatic-numerical-restart) may add
discarded work, which is included in its timing diagnostics.
