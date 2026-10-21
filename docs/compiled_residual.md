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
fallback. No numerical tolerance is relaxed. The self-contained C arithmetic
does not call the interpreter's `math.sumprod`.

### Runtime eligibility in 1.3.18

The independent compiled residual route accepts Windows x86-64 CPython 3.11
and later 3.x releases. The scalar route accepts CPython 3.12 and later 3.x
releases when `math.sumprod` is a built-in function and Python floats are
IEEE binary64. Neither gate has a Python 3 minor-version ceiling. Both require
a GIL-enabled build (`Py_GIL_DISABLED` absent or zero); rounding, gradual
underflow, operand-range and precision checks still apply.

The scalar route has no operating-system or architecture restriction beyond
its arithmetic requirements, preserving the earlier CPython 3.12 scope. A
compatible residual extension built on another platform can still be tried
through this scalar-eligible dispatcher. The separate dense-score entry below
remains restricted to Windows x86-64. A missing or declining helper retains
the guarded scalar or portable compensated fallback. Python 3.11 has no
`math.sumprod`, so its fallback is portable compensated arithmetic. Wheel ABI
compatibility and these arithmetic gates are separate from NumPy/SciPy BLAS.

The scalar residual bound assumes the built-in continues to use the extended,
triple-length product accumulation described above. The runtime gate does not
inspect a future interpreter's C implementation. Simulated version identities
test dispatch decisions, not arithmetic on those future interpreters.

### Recorded numerical validation

Routine complete-suite validation targets Python 3.11 and 3.12. Python 3.13
and later Python 3.x releases remain supported without dedicated complete-suite
runs. The checks below record earlier numerical qualification.

Released version 1.3.17 qualified the compiled residual and dense-score paths
on conventional Windows x86-64 CPython 3.11 through 3.13. The separate scalar
qualification used CPython 3.13.15 with the extension disabled, exact-rational
residual bounds, unsafe arithmetic-mode refusals and fitting comparisons against
portable compensated evaluation. The relevant extended-arithmetic and
`math_sumprod_impl` source sections in the official CPython 3.12.15 and 3.13.15
sources were identical. Those checks remain evidence for the tested runtimes;
forward eligibility does not relabel later versions or other platforms as tested.

The adapted arithmetic source includes the complete Python Software Foundation
license in `dwd/CPYTHON-LICENSE.txt`. Original DWD attribution and its MIT license
remain intact.

## Shared operand refusals

The compiled evaluator and the scalar `math.sumprod` route use the same finite
operand range: zero, or a nonzero magnitude between `2**-200` and `2**200`,
inclusive. If the original compiled evaluator reports that an operand is outside
this range, the scalar route cannot accept the same values. The dispatcher skips
that redundant scalar attempt and returns control to the existing portable
compensated evaluation.

This automatic shortcut requires the original compiled helper and dispatcher,
an exact NumPy array, the original scalar operand check and unchanged magnitude
limits. Custom hooks, array subclasses, a missing helper and other refusal
reasons retain the existing fallback sequence. Inputs must remain unchanged
during evaluation, as required by the existing numerical routines. Every submitted
worker task completes and its exceptions propagate before a refusal is returned. Error
bounds, solver acceptance checks, tolerances and estimator settings are unchanged.

The saving depends on how much scalar work would otherwise precede the refusal.
This is an internal optimization for that case, with no new estimator option
and no general claim of faster fitting or prediction.

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

The original compiled evaluator prepares the high and low parts of each
coefficient once per native chunk, then uses those parts in the existing
ordered accumulation. Its temporary allocation is 16 bytes per coefficient
per active chunk. Allocation failure uses the previous arithmetic directly;
range checks, rounding checks and acceptance bounds remain unchanged.

Eligible threaded evaluations reuse a lazily created worker pool within one
`solve_kernel` invocation. The pool is owned by the calling process and thread,
and is joined when that invocation exits, including exceptional exits.
Different fits, recovery attempts and ensemble learners do not share it.
Changed helpers, unsupported operand types or unsupported rounding modes keep
the existing dispatch path. There is no new public option for these internal
optimizations.

Long checks divide each worker's rows into bounded native calls. The dispatcher
returns to Python between chunks and polls threaded completion, allowing pending
interruptions to be observed on Windows as well as other platforms. A failure,
refusal or interruption stops remaining chunks; every submitted worker task finishes before
returning, and partial output is discarded. One row remains indivisible, so this
is not a hard wall-clock cancellation guarantee. Small checks retain a single
native call. Chunking preserves the arithmetic order within each row and does
not change the numerical acceptance rules.

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

A `compiled_operand_domain` refusal records this shared-domain shortcut. Its
scalar call count is zero because the redundant scalar attempt was skipped;
portable compensated work still occurs outside the native profile.

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

In version 1.3.18, the score path accepts Windows x86-64 CPython 3.11
and later 3.x releases with a GIL-enabled build, binary64 round-to-nearest
arithmetic and gradual underflow. Other operating systems retain the existing
Python calculation. It handles the
existing C/F-contiguous float64 kernels through bounded contiguous tiles, with
at most eight rows and O(n_training) scratch. It releases the GIL without
starting additional threads. Single-row tails, unsupported layouts/dtypes,
custom arithmetic hooks, active NumPy `call`/`log` handlers and missing or older
extensions keep the original Python path. A native refusal or scratch failure
releases its temporary storage before retrying the original evaluation and
error ordering. Tile and output buffers are allocated lazily and reused within
one eligible call. They are not shared across calls or stored on fitted models,
and are released before scalar tails or fallback. The same selected rows and
product order reach the C core; this reuse changes allocation, not arithmetic.

This is an internal execution optimization, with no new estimator setting.
The shared helper is used by prediction and by some fitting/validation score
calculations. Sparse scores and the joint-affine calculation retain
their existing arithmetic. Benefits depend on how many dense rows require
accurate scoring; ordinarily accepted dot products do not use this entry.

## Building and installing

An accelerated wheel is platform specific and is tagged
`cp311-abi3-<platform>`, not `py3-none-any`. The stable ABI permits loading on
CPython 3.11 and later; the platform, build and arithmetic guards above still apply.
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
