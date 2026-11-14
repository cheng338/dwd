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
and cleanup runs when that invocation exits, including exceptional exits.
Different fits, recovery attempts and ensemble learners do not share it.
Changed helpers, unsupported operand types or unsupported rounding modes keep
the existing dispatch path. There is no new public option for these internal
optimizations.

For an owned default executor, cleanup retries interrupted wake-up requests and
uses checked CPython capabilities to establish worker completion. It retains
ownership until completion is established and preserves the original task
failure, or the first cleanup interruption when no earlier failure exists.

After requesting wake-up, an exactly identified omitted worker that has not
finished starting may receive a private 10 ms readiness wait, timed with
`time.perf_counter()`. The deadline is fixed for that cleanup attempt; repeated
interruptions do not restart it. Readiness and a valid completion token must be
observed before the deadline. This is not a wall-clock limit on the call:
scheduling, completion waits and separate cleanup attempts can take longer.

If that startup remains unresolved, cleanup can still finish registered workers
whose ownership and completion tokens remain proved. Whole-executor completion
remains explicitly unproved and ownership is retained. A wake-up request or a
partial drain is not proof that every worker stopped. Waiting for a known worker
is not capped by the readiness budget and cannot finish if that work never
terminates. Cleanup does not kill threads.

Unsupported or replaced shutdown implementations have no completion guarantee.
Custom executor factories retain their previous lifecycle. There is no new
public option. This does not change signal handlers or guarantee protection
from arbitrary asynchronous interruption. The extra checks add cleanup cost,
particularly for tiny tasks, without changing residual arithmetic or numerical
acceptance.

Long checks divide each worker's rows into bounded native calls. The dispatcher
returns to Python between chunks and polls threaded completion, allowing pending
interruptions to be observed on Windows as well as other platforms. A failure,
refusal or interruption requests cancellation of remaining chunks. The owned
completion checks and explicit cleanup limits above apply; partial output is
discarded. One row remains indivisible, so this
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

## Ordered linear-feature gradient reduction

The current development source also builds a separate private
`_ordered_reduction_accel` extension. Eligible dense named-linear L-BFGS reports
use it automatically for the final sums of already-restored gradient terms.
The six product-expansion planes retain their original order. Product formation,
scaling, restoration checks, objective and gradient acceptance are unchanged.
Beta/coefficient computation and the existing residual and score entries do not
use this new extension. There is no new estimator selector.

The facade admits GIL-enabled Windows/Linux x86-64 CPython 3.11 and later 3.x
runtimes, with the existing binary64 and arithmetic-mode guards. It checks the
package-owned callbacks, array layout and genuine native entry before use.
The native call stages a complete private result and never publishes a partial
tile. Unsupported input, a native refusal or an absent/unloadable binary retains
the original checked sum; allocation failure follows the existing scalar replay.
Unexpected exceptions retain their existing propagation behavior. An unsuccessful
initial optional load is cached for the life of that Python process. Install or
replace package binaries only between processes, then start a fresh interpreter.

This reducer is serial and keeps the GIL: the residual worker-pool description
above does not apply to it. It does not change BLAS/OpenMP thread limits and uses
bounded private packing/output storage. Scalar gradient columns do not import
the new facade or binary. Validation of this route covered CPython 3.12 on the
approved Windows and Linux environments; forward runtime eligibility is not a
claim that every later interpreter was tested. See the
[reporting measurements](kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting)
for the wider-workload gains and ordinary/portable-fallback costs.

## Exact feature-expansion tiles

A separate private `_feature_expansion_accel` extension accelerates qualified
dense named-linear L-BFGS reports. The report dispatcher reuses existing
report-specific score products, including supported native report outputs on
Linux. The expansion helper reuses the existing compensated splits for native
gradient tiles and eligible warm beta calculations. Products, exponent
restoration, plane order, final sums, outward bounds and acceptance remain the
same. It introduces no estimator parameter, solver method or new tolerance.

The helper uses CPython's full ABI. Builds include it only for GIL-enabled
64-bit x86-64 CPython 3.12 on Windows or Linux; its runtime interpreter-version,
array, callback, floating-point mode and source-definition guards must also
pass. It is not an abi3 extension. An unavailable, older or incompatible
extension, active tracing/profiling, or a refused operand/context retains the
original checked implementation. Installing a cp312 wheel does not establish
that this private route qualified on a different interpreter micro version.
Python support outside this native domain is preserved through fallback.

The first eligible tile keeps the existing ordered-reduction path; subsequent
use requires a genuinely qualified context. There is no numerical prewarming
or cross-fit cache used to hide startup. Calls stage complete private output
and retain the existing error and scalar recovery behavior. The tile helper
is serial, holds the GIL, uses bounded scratch, and does not alter native
thread limits. The residual worker-pool description above does not apply.
An unsuccessful optional load is cached within that process; replace binaries
between processes. Package code does not build or download anything at runtime.

Qualified warm gradient/beta calls can combine original groups up to 16,384
entries, with at most 64 feature columns and a sample width of 1,024. The first
original group is unchanged; cold gradient setup continues through original
groups until its context exists. Scalar and tail groups retain their original
sizes. A refused larger attempt replays the original groups in order, with the
same arithmetic, error precedence and outward bounds. This increases the
maximum contiguous float64 feature-input copy from 16 KiB to 128 KiB per call;
outputs, reusable splits and native scratch are separate from that input bound.
Rebuild `_feature_expansion_accel` from matching sources for private
`API_VERSION = 2` and the new capacity. Older binaries decline this route and
retain the checked fallback; there is no new public option.

The reviewed Windows/Linux CPython 3.12 measurements found larger and repeated
fit benefits, plus cold-start and missing-extension costs. The smaller-buffer
recomputing interface and separate native dual-reduction candidate were not
selected. See [the measured scope and limitations](kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting).

## Building and installing

Accelerated wheels are platform specific. A qualified CPython 3.12 build that
includes the full-ABI feature-expansion extension uses
`cp312-cp312-<platform>`, never an abi3 label. Other accelerated builds retain
the residual and ordered-reduction extensions in a
`cp311-abi3-<platform>` wheel. The stable ABI applies to those two extensions,
not to the new expansion helper; runtime arithmetic and build guards still
apply. Portable wheels remain `py3-none-any`. No compiler or tool download is
invoked at runtime, and none of these extensions uses the NumPy C API.

Source builds use setuptools and a locally available C compiler. In the current
source, leaving `DWD_BUILD_ACCEL` unset, or setting it to any value other than `0`,
requires `_residual_accel` and `_ordered_reduction_accel`, plus
`_feature_expansion_accel` on the qualified build domain above.
`DWD_BUILD_ACCEL=1` makes that intent explicit.
Compiler or linker errors stop the build. After compilation, a missing or empty
expected extension file for any required module also stops the build. This file check does not replace
the runtime loading and numerical guards above.

`DWD_BUILD_ACCEL=0` explicitly requests a portable build without compiling any
helper. Users without a compiler can choose that mode or install a published
portable wheel. This changes the earlier automatic source-build fallback;
published artifacts and numerical runtime fallbacks are unchanged.

Builds retain strict floating-point flags. MSVC uses `/fp:strict`; GCC/Clang
builds disable implicit contraction, reassociation and fast-math. For the
audited Windows release build, `DWD_BUILD_ZIG` can name a local Zig executable.
That explicit override remains strict and never downloads a compiler.

The mode setting applies only when a source build actually runs. Installing a
published or cached wheel does not invoke it, and a previously cached portable
wheel may be reused even after a compiler becomes available. When changing
mode, use a separate clean checkout or source-archive extraction and a fresh
wheel output directory. Set the desired mode and build from that source with
`python -m pip wheel --no-cache-dir --no-deps --wheel-dir new-wheelhouse .`.
A clean build tree also avoids reusing earlier compiled products; the mode flag
alone does not establish their freshness.

The performance improvement is workload dependent. It targets models requiring
compensated residual checks, particularly large ill-conditioned kernel systems.
It is not a claim that every DWD fit, reference eigendecomposition or CV search
becomes faster by the same factor. The default allows at most 100 MM updates per attempt and can stop earlier on
objective change; numerical convergence is reported separately. An eligible
[automatic numerical restart](kernel_dwd.md#automatic-numerical-restart) may add
discarded work, which is included in its timing diagnostics.
