# Compiled residual arithmetic

Version 1.3.4 optionally builds a small C extension that accelerates the existing
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
The C core additionally checks its arithmetic domain and rounding behavior.
Uncertain or unsupported evaluations retain the prior Python/native or portable
fallback. No numerical tolerance is relaxed. The current native-screen runtime
gate remains CPython 3.12; other supported Python versions retain the portable
path even if they can load the stable-ABI extension.

The adapted arithmetic source includes the complete Python Software Foundation
license in `dwd/CPYTHON-LICENSE.txt`. Original DWD attribution and its MIT license
remain intact.

## Threads and memory

Independent row ranges can execute concurrently because the extension releases
the GIL. The worker count does not exceed the smallest active BLAS thread limit,
the logical CPU count, or one worker per 1,024 rows. If no BLAS budget is visible,
it uses one worker. Small matrices use the existing Python path.

Thus an external `threadpool_limits(8)` context permits up to eight arithmetic
workers. GridSearchCV processes configured with one BLAS thread remain single
threaded inside this check. This does not introduce another CV engine or change
the ensemble package. The numerical code uses the same read-only kernel and
O(n) additional output storage; workers do not copy the Gram matrix or create
child processes.

## Building and installing

An accelerated wheel is platform specific and is tagged
`cp311-abi3-<platform>`, not `py3-none-any`. The stable ABI permits loading on
CPython 3.11 and later; the narrower numerical runtime gate above still applies.
No compiler or tool download is invoked at runtime, and the extension does not
depend on the NumPy C API.

Normal source builds use setuptools and a locally available C compiler. The
extension is optional: a missing compiler can produce a functioning package
using the previous arithmetic. Compile success does not authorize relaxed
floating-point flags. MSVC uses `/fp:strict`; GCC/Clang builds disable implicit
contraction, reassociation and fast-math.

`DWD_BUILD_ACCEL=0` explicitly requests a portable build. Always use a separate
clean source/build directory for portable and compiled wheel builds. For the
audited Windows release build, `DWD_BUILD_ZIG` can name a local Zig executable.
That explicit override fails the build if compilation fails; it never downloads
a compiler or silently produces a purported accelerated wheel.

The performance improvement is workload dependent. It targets models requiring
compensated residual checks, particularly large ill-conditioned kernel systems.
It is not a claim that every DWD fit, reference eigendecomposition or CV search
becomes faster by the same factor. The default remains 100 MM updates, with
objective-change stopping and convergence reported separately.
