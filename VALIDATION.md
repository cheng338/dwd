# Validation

## Version 1.3.13: optional residual profiling

The `residual_profile()` context passed 16 focused tests and 52 existing
native-residual, bound, compiled-helper and adaptive-order regressions.
The checks cover scalar and compiled routes, strided/read-only inputs,
partial work before refusals, caught and propagated exceptions, nested
contexts, independent threads/tasks and copied contexts after scope exit.
Disabled instrumentation reads no clocks. A static comparison found the
original arithmetic AST unchanged after removing observation-only code.
The compiled sources, native extension and solver files are unchanged.

Four full-kernel MNIST (1,7) fits ran with profiling off/on/on/off at the
saved training rows, parameters and 3,000-iteration limit. All four reproduced
the saved coefficients, intercepts, objective histories and checked training
scores and predictions bit for bit. Each profiled fit recorded 3,000 compiled
helper calls and no scalar `sumprod` calls; native-call counts matched the
solver's existing counters. Bounded native returns remain distinct from the
solver's subsequent acceptance decisions. Two complete 45-pair ensemble fits
also preserved scientific state with profiling off and on.

The sequential Windows CPython 3.12/AOCL hard-case fits used one native thread.
Mean fit times were 42.020 seconds without profiling and 42.126 seconds with
profiling. Whole-interval background CPU observations were retained, but do
not exclude short bursts. These limited observations do not establish a
universal overhead estimate or a speed improvement. The checks did not read
official test arrays or replace dissertation timings, and do not turn an
iteration-limited result into a convergence claim. See the
[profiling scope and usage](docs/compiled_residual.md#optional-residual-profiling).

## Version 1.3.12: dense accurate scoring

The native dense-score entry preserves the previous ordered expanded products
and summation, with exact normal-number exponent operations and a library
fallback for exceptional ranges. The residual evaluator, mathematical model,
score screen and estimator defaults are unchanged. See the
[implementation limits](docs/compiled_residual.md#dense-accurate-scores).

The earlier integration passed 13 new test methods, including 105 exponent-boundary combinations;
88 existing relevant tests passed, with two wider-dtype tests skipped on the
Windows platform. Separate isolated arithmetic checks passed 10,626 scaling,
187 helper and 157 summation cases. All 60 saved DWD/ensemble models reproduced
score and label bytes through the installed fix. Seven fresh fits matched
score/label bytes, 96 checked fitted arrays and recorded top-level stopping
controls; this does not claim equality of every nested diagnostic field.

The following timing used the preceding private build, before the Windows-only
eligibility predicate was added. Windows arithmetic is unchanged; the final
release wheels were not retimed. At unchanged selected parameters, the affected
MNIST 2/3 ensemble prediction
mean fell from 96.707 to 63.084 ms (34.8%) across ten saved training draws. The
other five full-kernel/ensemble cases showed only small timing fluctuations.
The matched Windows Ryzen 9 9955HX/AOCL comparison used one native thread,
prediction batches of 1,024, balanced repeated warm calls and whole-block
background screening. All 30 final blocks were accepted from 32 attempts;
rejected blocks were retained. Observed background thresholds were at most one
aggregate CPU core and 0.05 core for other detected numerical processes.
Those counters do not prove exclusive CPU use or quantify interference in a
single-thread call. No Intel or Ubuntu performance result is claimed.

The main release remains Python 3.11+. CPython 3.10 support is confined to the
separate portable wheel and its reproducible compatibility source bundle.
The new native score entry is limited to conventional Windows CPython 3.12/x86-64;
unsupported runtimes and inputs retain the prior Python calculation.

The final 1.3.12 installed-artifact checks ran 115 targeted test methods per
wheel: 113 passed with the Windows CPython 3.12 native wheel (two wider-dtype
skips), and 99 passed with both the Python 3.12 portable and CPython 3.10
companion wheels (16 native/wider-dtype skips each). This includes the final
14-method dense-score module, with its explicit operating-system guard.
Three public-fit smoke checks passed per wheel. The rebuilt extension is
byte-identical to the validated private extension. These are targeted release
checks, separate from the earlier integration and historical full-suite runs.

Offline selection with all three DWD wheels available selected the CPython
3.10 companion on Python 3.10 and the standard ABI wheel on Windows Python
3.11/3.12. The simulated Linux Python 3.11 target selected the standard
portable wheel. The Python 3.11 target checks do not constitute execution on
a Python 3.11 interpreter or on Linux.

## Version 1.3.11 validation record

Version 1.3.11 packages the changes below, developed and validated individually
after 1.3.10. The per-change records retain their original scope and timing
limitations. For 1.3.10 evidence, see the separate
[validation scope and timing follow-up](docs/validation-1.3.10.md).

## 1.3.11 main release verification

The main release requires Python 3.11 or later. It retains the original
stable-ABI build policy and standard-library TOML test import. Python 3.10
compatibility is provided only by a separate CPython 3.10 wheel and its
associated compatibility source bundle; it is not enabled in the main source.

The numerical payload in these release wheels is byte-identical to the
payload exercised by the complete 684-test suite on Windows with Python
3.12.14 and AOCL: 678 tests passed with the native wheel and 667 passed with
the portable wheel, with six and 17 declared skips respectively. No executed
test failed. That preliminary packaging candidate also supported Python 3.10;
its compatibility-only build/metadata/test changes are absent from main.
Final main artifact checks separately verify Python >=3.11 metadata,
installed source provenance and the restored build/test contracts.

Skips cover unavailable wider-than-float64 arithmetic, optional compiled
features and optional frozen-binary comparisons. The three frozen-binary
comparison methods subsequently passed on the native wheel; the rebuilt
extension is byte-identical to the earlier validated extension. Independent
rational-oracle checks passed in the full native suite.

The AOCL environment uses NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0,
joblib 1.5.3, threadpoolctl 3.5.0 and CVXPY 1.8.2. NumPy and SciPy load AOCL-BLAS
5.2.0. Broad optional SOCP/CVXPY imports additionally load MKL 2025.3 and Intel
OpenMP alongside AOCL's LLVM OpenMP and scikit-learn's Microsoft OpenMP.
`threadpoolctl` reports the multiple-OpenMP warning. Correctness checks
completed successfully; this is not evidence of a pure AOCL process or a
performance comparison. Loaded library paths, versions and hashes were saved
with the validation receipts. Windows checks on the AMD host do not establish
Ubuntu behavior or Intel-target performance.

<a id="final-refit-thread-control-in-local-source"></a>

## Final-refit thread control

The standalone kernel tuning example now accepts an explicit positive integer
`final_native_threads`, with default `1`, also exposed as
`--final-native-threads`. The CV worker default remains `jobs=1`; larger worker
counts are capped by unfinished folds. Each worker keeps one native thread per
detected numerical-library pool. The final limit applies only to full-data
scaling, fresh fitting and diagnostics, and is not a machine-wide CPU limit.
Core estimator source, native binaries, solver equations and defaults are
unchanged. The adopted example files are byte-identical to the validated
prototype.

The isolated example passed 11 focused test methods and 55 existing workflow
tests, with no skips in the completed checks. Coverage included final fitting
with two native threads under a three-thread caller, two-worker CV with
one-thread identities, invalid API/CLI counts, runtime attestation, skipped
refits, strict old-source rejection, summary provenance and resume with final
counts changing from one to two and back. RuntimeError and KeyboardInterrupt
during final scaling, fitting and runtime checks restored caller limits and
allowed a fresh retry using unchanged CV receipts. SystemExit was not directly
injected. An initial test-launcher TEMP permission failure was resolved by
using a permitted workspace temporary directory, without changing package code.

Default source comparisons retained the original guard/context call sequence.
Matched execution sequences produced 40 byte-identical recorded arrays across
the old and updated examples. Earlier instrumented fits and later repeated
fits showed last-bit differences in both unchanged and updated source: maximum
coefficient difference 9.992e-16 and score difference 4.164e-16, with unchanged
labels. The initial comparison failures and exact matched-sequence checks were
retained; numerical tolerances were not relaxed. This evidence does not promise
universal repeatability or bitwise equality across thread counts.

After adoption, 65 canonical workflow tests passed: ten new portable
final-thread regression methods and 55 existing checkpoint/CV methods, with
no failures, errors or skips. These checks used the canonical Python sources
and their portable residual fallback; no native extension was added to the
checkout. The earlier isolated validation separately exercised the matched
compiled runtime. The durable tests depend only on files bundled with the
package and retain strict runtime guards when other suites load historical
fixtures. The ten adopted methods are distinct from the eleven external trial
methods; their counts are not added together.

The option permits measuring an appropriate final-fit limit for a workload;
it does not select one automatically or guarantee a speedup. Correctness checks
could overlap other workers and provide no timing evidence. Increasing native
threads can also increase CPU time without improving elapsed time.

<a id="prediction-kernel-block-lifetime-in-local-source"></a>

## Prediction kernel block lifetime

Multi-batch prediction releases its local reference to each kernel block after
copying the completed scores into the result, before constructing the next
block. Kernel calls, batch shapes, arithmetic, fitted policies and the solver
are unchanged. This is automatic internal behavior with no new public option;
it does not reuse or overwrite kernel storage.

The isolated implementation passed 334 paired numerical cases, six
construction/scoring failure-and-retry cases and five allocation controls.
Coverage included dense layouts, CSR, read-only inputs, named and callable
kernels, precomputed input, fitted RBF and scoring policies, and short final
batches. Each source version also passed 64 selected repository tests, with
two wider-precision tests skipped on this Windows NumPy runtime. Ten fresh
ensemble cases retained 460 recorded arrays per version byte for byte; three
saved MNIST models matched across nine paired prediction cases, including the
historical predictions at their original batch size. Timing fields were
excluded from fitted-state comparisons.

For a 4 MiB owned float64 kernel block, traced peak allocation fell from about
12 MiB to 8 MiB. Weak references independently confirmed that the preceding
block was no longer live at the next construction. Single-batch, precomputed
view and externally retained-array controls showed no material saving. This
is one less overlapping block, not a claim about total process memory or RSS.
Ensemble consensus and binary prediction inherit the improvement when they
use multiple owned blocks. OvO's outer batching can leave each child with only
one batch, in which case this change gives no additional saving.

Six prediction workloads were timed in five alternating fresh-process pairs
each, with one native thread and equal work. Median paired changes ranged from
0.28% slower to 0.62% faster, and every case's range crossed zero. Only one of
30 pairs met the predeclared background-CPU screen; the available process
inventory could not identify all wider-host activity. These timings establish
no meaningful speed effect and cannot exclude small regressions. The memory
evidence is independent of those timings. The adopted numerical source is
byte-identical to the tested candidate.

After adoption, 68 selected package tests passed, with the same two
wider-precision platform skips and no failures or errors. The standalone
lifetime regression tests verify collection before the next allocation, copied
score views, externally retained arrays, and failure followed by retry. These
checks introduce no new timing claim.

<a id="linear-gradient-reuse-in-local-source"></a>

## Linear-gradient reuse

Eligible built-in float64 `GenDWD` iterations with `stopping='optimality'`
reuse the gradient already evaluated at the current coefficients and intercept
for the next MM step. The step formulas and arithmetic order, initialization,
loss, regularization, free intercept, stopping thresholds and iteration budget
are unchanged. Final diagnostics still recompute the gradient at the returned
iterate. This is an internal optimization within the existing optional stopping
mode, not a new public option or a change to default objective stopping.

Reuse requires the original gradient, derivative and selected step hooks,
checked around evaluation and again before dispatch, and an actual float64
coefficient gradient. Substituted or self-restoring hooks and other gradient
dtypes retain the previous computation. Active NumPy `call` and `log` error
handlers also bypass reuse because they can raise or change state. Removing
duplicate arithmetic can reduce ordinary warning counts; the numerical
result is unchanged in the tested warning controls.

Before adoption, the final isolated implementation passed 166 focused checks:
152 core comparisons and controls, plus 14 error-handler checks. Coverage
included dense/CSR inputs, implicit/explicit updates, Schur/legacy modes,
all three stopping modes, initialization, zero/one update, cached preparation,
parameter and scaling extremes, public CV, invalid fits and cleanup. The
comparable numerical states, histories, scores and labels matched bitwise.
Independent scalar and finite-difference checks supported the gradient
interpretation. The 1,110 observed reused step entries used the same iterate,
avoiding 1,110 derivative evaluations and transposed feature products.
Separately, 63 relevant repository tests passed for each frozen source version,
with no failures, errors or skips. These were selected linear/CV modules, not
the full package suite.

After adoption, 76 selected package tests passed and one wider-precision test
was skipped on this platform, with no failures or errors. The new regression
module uses independent original step formulas and covers public fits, CV,
same-iterate reuse, final diagnostics and compatibility fallbacks without
depending on external trial files. The adopted numerical source is
byte-identical to the final validated and benchmarked candidate. These
integration checks address correctness; they are not new timing evidence.

The Windows runtime's `longdouble` had the same mantissa width as float64, so
an actual wider-precision legacy fit was not executed. Dtype-fallback controls
and source review support that guard; they do not replace testing on a platform
with wider `longdouble` arithmetic.

The final guarded source was timed in five alternating fresh-process pairs
for each of four representative workloads. On AOCL BLAS with one native thread,
the observed median paired fit-time reductions were 26.85% for cached dense
implicit updates and 27.88% for CSR explicit updates. Both optimality workloads
used `tol=0` and exactly 80 updates, so these are equal-work timings rather
than times to convergence. Cached eigen-preparation was outside the dense
timer. Each timed batch contained the same number of fits in both versions
and a small shared copy of fitted state; hashing and predictions were outside
the timer. All 20 pairs matched their data and model signatures.

All final timing pairs exceeded the declared background-CPU threshold. No
other coordinated numerical task was running, but the available process
inventory could not identify the wider host contributors. The reductions
therefore describe the recorded workload and background activity, not a
verified idle machine or a universal speedup. Earlier timings from a source
without the final error-handler guard remain separate. Objective and fixed
stopping were controls, with no claimed speed benefit or exactly zero overhead.

Linear `GenDWDCV` fits inherit reuse when configured for optimality stopping.
Kernel DWD, ensemble-DWD and the kernel-based resumable example receive no
direct benefit. Their defaults and the linear examples' defaults are unchanged.

<a id="optional-joint-affine-scoring-in-local-source"></a>

## Optional joint affine scoring

`affine_computation='joint'` retains the DWD methodology while including the
intercept in accurate score evaluation. The default remains `'standard'`.
Before adoption, the numerical implementation passed 1,152 independent fixed
and extended checks across AOCL and OpenBLAS, including exact rational oracles,
CSR duplicates, subnormals, overflow, allocation failures and strict NumPy
error settings. Another 38,034 comparisons verified the optimized scalar
error-bound calculation against its preceding implementation. These are scoped
checks, not a proof of correctness for arbitrary numerical backends.

Eleven paired integration cases included 34 validation observations. Joint
scoring matched the exact represented-input class rule in all 34, correcting
six standard-scoring mismatches in deliberately sensitive cases. Fixed-fit
states and common callback states remained bitwise identical. Forty-seven
existing focused tests passed against the isolated integration.

Twelve saved MNIST models retained all test labels. Twelve fresh ensemble fits
across six locked pair/RBF-policy configurations (72 base/final fits) retained
coefficients, intercepts, objective histories, non-timing diagnostics, consensus
values, full rankings, selected rows/order, labels and accuracy. Two ensembles'
test-score arrays changed by at most `1.31e-9`; recorded durations naturally
differed. No tuning was repeated and no general accuracy gain is claimed.

Ten counterbalanced fresh-process prediction pairs per workload found
approximately unchanged full-kernel pipeline time and about 9% extra time for
the difficult ensemble. Other numerical test workers had exited first. Three
of twenty pairs had background-CPU flags, with no external Python worker
observed; excluding flagged pairs preserved the conclusion. Constructed exact
ties were substantially slower. These timings concern prediction, not training
speed or a guarantee of zero overhead. The [kernel guide](docs/kernel_dwd.md#optional-joint-affine-scoring)
describes the option's precision limits and validation-stopping effects.

After integration into the public API, 112 selected DWD tests and all 26
resumable-CV example tests passed. The latter include joint-policy interruption
and resume, and rejection of checkpoints created with a different affine
policy. Eleven integration cases per policy reproduced the corresponding
frozen standard and joint trial records exactly, including fitted states,
callback values, validation histories and public scores. The adopted helper's
executable content is unchanged from the validated prototype. These adoption
checks ran with concurrent correctness work; they are not timing benchmarks.

<a id="optional-extended-exact-recovery-in-local-source"></a>

## Optional extended exact recovery

`exact_recovery='extended'` selects rank-32 and 8192-bit limits for both exact
factor certification and MM recovery. The standard rank-16/4096-bit setting
remains the default; entry and operation caps and acceptance checks are
unchanged. Exact preparation remains conditional on an ordinary MM failure.

The isolated trial recovered the three 30-row cases from nine previously
rejected nearly constant RBF fits. They completed both five fixed updates and
100 updates with default objective stopping. The longer runs reached the cap
without meeting the convergence or stopping criterion. Independent Fraction
calculations verified 315 recovered actions and their original-kernel functions
and DWD objectives. Forty focused tests covered rank boundaries, separate
resource limits, invalid matrices, ordinary-path bypass and interruption cleanup.

The other six cases still refuse at the profile's rank limit. Independent exact
quadratic witnesses for their 75- and 120-row stored kernels establish
indefiniteness; a larger budget cannot certify them as exactly PSD unchanged.
This is a property of those represented matrices, not of the mathematical
Gaussian kernel. The historical rank-budget refusal was the first encountered
limit, not the complete diagnosis.

After public API integration, 112 selected DWD tests and six checkpoint/resume
controls passed. Twenty-one historical fit attempts preserved all expected
refusals and successful results: all nine standard fits refused, six extended
cases refused, and the three eligible cases returned in both the five- and
100-update controls. Their 315 recovered actions, coefficients, intercepts,
objective histories and decision scores matched the isolated trial exactly.
Eleven ordinary cases per affine policy also retained their pre-adoption
records. The exact-factor and MM arithmetic modules are byte-unchanged.
These are correctness checks with concurrent test work, not timing evidence.

Three separate fresh-process observations measured median fit times of about
0.066 seconds for default refusal and 2.744 seconds for five successful updates.
The outcomes differ, so this is descriptive cost rather than a speedup. Other
known trial workers had stopped; systemwide background activity and peak memory
were not measured. The [kernel guide](docs/kernel_dwd.md#optional-extended-exact-recovery)
explains the unchanged small-matrix cap, per-stage work limits and conditional
benefit. Neither successful return nor a larger budget proves convergence.

## Earlier validation records

The older records below retain their original versions and scope.

For 1.3.9, see the [changes and validation scope](docs/candidate_changes.md)
and [release notes](RELEASE_NOTES.md). Development-build evidence and stable
artifact checks are distinct; a source pass does not certify a different binary.
The [1.3.8 change-by-change validation](docs/validation-1.3.8.md) remains the
historical record for that release, including its measured runtime costs.

# DWD 1.3.5 source validation

The final source passes 412 tests in py12_df, including 22 new regression
methods for failed-fit cleanup. These cover all seven public classifier
interfaces, changed labels and features, early and late failures, partial
result handling, failed CV scoring and final refits, interruptions, successful
recovery, fit signatures, and preserved private precomputation caches.

An AST comparison against the numerical repair used in the MNIST experiment
confirms that existing numerical functions and method bodies are unchanged,
apart from the fit wrappers and replacement of incomplete kernel state cleanup.
The free intercept, regularization, loss, MM and SOCP equations, and stopping
defaults remain unchanged. No MNIST tuning or test evaluation was repeated for
the failed-refit repair. See [the fitted-state contract](docs/failed_refit_state.md).

The preceding numerical repair passed 390 source tests. Separate MNIST first-fold
screens complete all 270 learner fits across 45 pairs at each of 10% and 20%
base sampling, using lambd=1e-12, gamma=1e-5/784 and selection ratio 0.3.
The 10% screen uses the final anchor candidate; the 20% screen uses the preceding
midpoint candidate and never reaches the added anchor path. These are scoped
numerical diagnostics, not final tuned results. Those checks did not access the
official test set.

The final candidate's historical replay retains all 21 valid returned models
and the same nine extreme synthetic numerical rejections. All returned states
are bitwise identical to the midpoint candidate; against 1.3.4, 20 are bitwise
identical and one has decision differences below 7.6e-10 with unchanged labels.
The unchanged acceptance thresholds still reject unchecked or inaccurate states.

Built-wheel validation and exact artifact hashes are recorded separately; a
source pass alone does not certify a different binary artifact.

# DWD 1.3.4 validation (preserved history)

The source was validated in py12_df: CPython 3.12.14, NumPy 2.5.2, SciPy 1.18.0,
scikit-learn 1.9.0 and MKL 2025.3.

| Check | Result |
|---|---|
| Complete source suite with CVXPY | 367 passed |
| Complete source suite with CVXPY blocked | 361 passed; six expected optional-solver skips |
| Published ensemble-dwd 0.1.1 | 85 passed; parent and worker imports verified |
| Exact original full-training 2/3 fit | 100 updates; all stored states and probes bitwise identical |
| Frozen public synthetic replay | Same 21 returned models and nine rejected case IDs |

Source, extension, test and runtime hashes were checked before and after suite
execution. Actual-wheel, portable-wheel and additional-runtime results are in
the accompanying build and validation receipts. A source pass alone does not
certify a different binary artifact.

## Independent arithmetic checks

The five compiled accumulation helper bodies retain CPython's order of operations.
The core is built without reassociation, fast-math or implicit contraction.
An independent core oracle checked 153 cases with 2,444 exact-Fraction enclosures
and 768 Decimal crosschecks. The actual-extension oracle checked 139 cases and
compared all six outputs against disabled compilation: scores, expanded residuals,
coefficient-sum residual and all three bounds matched bitwise wherever returned.
Acceptance/declines, thread budgets and partial-output fallback also matched.

Eleven new bundled tests cover arithmetic, unusual strides and unaligned buffers,
readonly inputs, output validation, thread budgeting and optional-extension
fallback. A separate compiled-core test verifies that unsupported rounding and
underflow modes are declined and restores the floating-point state afterward.

## Fixed full-MNIST regression

The fit uses all 12,089 official training images, the same StandardScaler,
gamma=1e-4, package lambd=2^-31, q=1, zero initialization, optimized Cholesky MM,
eight native threads and the unchanged 100-update/objective-change defaults.
No parameter search or official test-label evaluation was performed for this repair.

The 47.12-second fit retained all saved coefficients, intercept, objective history,
class mapping, preprocessing and scores/predictions on 257 fixed training probes
bit for bit. All stopping/convergence flags matched. It still reached 100 updates
without declaring convergence, as the saved model did. This verifies the same
finite-iteration classifier, not a newly converged optimum.

The same 105 compensated checks and five numerical refinements occurred. Their
elapsed component fell from 352.11 seconds in the saved primary fit to 20.03 seconds.
Native and compensated timer fields overlap and must not be added. The previous
377.39-second figure is a three-fit median; 47.12 seconds is the initial fresh
repaired fit. Accompanying receipts record actual-wheel repetitions separately.

## Preserved limitations

All 21 returned public synthetic models passed independent Decimal100, finite-output,
classification and serialization checks, retaining saved coefficients, intercept,
decisions and objective histories. The nine extreme nearly constant RBF failures
retain their original case identities. No additional rejection was introduced.

The accelerator is optional. Unsupported runtimes/arithmetic retain the previous
checker. Benefits are largest when compensated checking dominates; ordinary fits,
eigendecomposition and kernel construction have different costs. These checks do
not establish universal speed or accuracy improvements.
