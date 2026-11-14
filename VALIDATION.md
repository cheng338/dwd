# Validation

## Native-loader initialization-order correction (2026-11-10)

The complete-suite follow-up identified a package acceleration lifecycle defect.
The November 9 fused-gradient path could capture the ordered reducer's cold
loader state before that reducer finished loading. A subsequent legitimate
initialization then invalidated the immutable context, and its intentionally
sticky mutation guard kept refusing fusion. DWD 1.3.19 has no fused context or
ordered native reducer, so this specific defect is absent from the release.
The November 8 lazy loader alone did not introduce it; the November 9 context
capture introduced the interaction. The earlier fixture prewarming preserved
checks but did not repair the package lifecycle, as clarified below.

The correction adds a definition-owned loader-completion state. Only normal
terminal loading outcomes, including expected negative caches, publish exact
True. Cold, reentrant, concurrent in-flight and unexpectedly interrupted loading
cannot qualify a first context. The original path runs while loading is
incomplete; a later eligible call captures the completed state. The completion
binding joins the unchanged final native guard. Existing contexts are never
rebased, and later mutations still refuse. No C source, binary interface,
arithmetic order, numerical acceptance, public option or solver default changed.
No environment or dependency was modified, and pytest/psutil remain validation
tools rather than new runtime dependencies.

Isolated Python 3.12 validation used Windows dwd_aocl and Robust Linux mkl202602:

| Selection | Windows | Robust |
| --- | ---: | ---: |
| Full DWD suite, 5,377 collected | 5,307 passed; 70 skipped | 5,354 passed; 23 skipped |
| Focused native, 845 selected | 845 passed | 843 passed; 2 skipped |

Both full suites and both focused native runs exited zero. The full suites include all 20 new fresh-process
cases covering genuine gradient/beta acceptance, cold/warm order, expected
negative caches, reentrancy, blocked concurrent loading, unexpected exceptions,
malformed completion flags and post-capture mutation refusal. Relevant numerical
checks retain exact Fraction-reference comparisons. The previous 268 blocked
nodes now all pass on Windows; on Linux 267 pass and one reaches its pre-existing
platform skip for an unavailable audited flush-mode control. All other earlier
skip nodes and reasons are unchanged. Focused and full counts overlap and must
not be added as unique tests.

The local portable selection completed 420 passes and 349 test skips among 769
selected nodes. Its additional native-core module collection skip accounts for
76 unavailable native-only selections; pytest's console counts that module as
one additional skip. The initial selection guard refusal occurred before any
test execution and remains recorded. An earlier restricted-permission local
full attempt and its sandbox-specific hard-link/socket failures also remain
recorded separately; the complete normal-user run above passed. Isolated
source snapshots and recorded dependency-file hashes were unchanged during the runs. Captured
proven process owners were closed; transient unproven discoveries are retained,
so these receipts do not assert universal descendant coverage.

A separate finite cost panel compared the frozen pre-fix development source
with the correction, using 96 fresh children per host: six ABBA blocks for each
of two shapes and cold/warm modes. It used one native thread and serial children
because these small private-helper measurements require isolation, not parallel
throughput. Cold timing includes the first and second complete gradient calls;
their sum excludes the untimed state-observation gap. Warm gradient and beta
batches contain 256 calls each. Oracles, process observations and genuine native
qualification are outside timers. Both hosts completed all children with exit
zero, exact checked outputs and unchanged relevant source/artifact hashes.

| First two cold helper calls | Pre-fix median | Corrected median | Median paired corrected/pre-fix |
| --- | ---: | ---: | ---: |
| Windows, 2 x 8 | 1.063 ms | 1.414 ms | 1.367 |
| Windows, 32 x 63 | 1.279 ms | 1.697 ms | 1.360 |
| Robust, 2 x 8 | 2.541 ms | 3.751 ms | 1.464 |
| Robust, 32 x 63 | 2.974 ms | 4.392 ms | 1.451 |

This exposes a one-time initialization cost of about 0.35-0.42 ms on Windows
and 1.21-1.42 ms on Robust in these cases; deferral does not eliminate that cost.
Warm paired median ratios range from 0.995 to 1.008 on Windows and 0.978 to 1.036
on Robust, with variation between blocks. These are descriptive helper timings,
not a universal no-slowdown, statistical precision, total-fit or vendor/version
comparison. Untimed qualification is not a per-timed-call native-entry counter.
The measured one-time startup cost is accepted to avoid prematurely capturing mutable
loader state and permanently disabling the eligible fused path after legitimate
initialization. This correction does not close the separate accumulated
wide-feature reporting-cost question.

## Shared-fold performance decision (2026-11-09)

A direct follow-up compared the current explicit shared-fold helper, current
ordinary `GridSearchCV.fit`, and the released ordinary implementation. It used
Windows `dwd_aocl` with 16 native threads and Robust Linux `mkl202602` with 20,
with serial searches (`n_jobs=1`). The baseline requested training scores;
separate candidate controls covered `return_train_score=False` for two cases.
Within-host inputs, folds and numerical oracles were shared, while current
ownership and route-specific checks remained in force. This was not a causal
comparison of numerical-library vendors or versions.

Each host completed 21 profiled qualification children and 252 unprofiled
baseline timing children containing 756 searches. Separate comparisons of three
private training-identity candidates completed 54 six-search children and 54
four-search children per host. All scientific checks passed. Focused baseline
and candidate selections totaled 776 passing test executions per host, plus
three local measurement-harness checks. These counts include tests repeated
across different candidates; they are not counts of unique tests.

The original timing policy admitted 247/54/54 baseline/first-candidate/final-
candidate children locally and 227/52/52 on Robust. Excluded observations were
preserved, with no replacement runs. Saved-data audits and independent reviews
verified the supplied artifacts, scientific joins, known process closures and
paired statistics. Source-graph entries whose bytes were absent from the audit
archive were checked through retained origin and before/after hash records;
this is not a claim that reviewers physically rehashed those unavailable files.

The accepted decision retains the existing helper as an explicit,
workload-specific option. Ordinary search is preferred in the measured
configurations where it is faster; fitted ownership and numerical guards stay
in place. The candidate shortcuts added no identified correctness improvement
and did not establish reliable whole-search benefit sufficient to justify their
extra state or dispatch complexity. No candidate was integrated, and no package
algorithm, default, dependency or environment was changed. See the
[direct comparisons and use guidance](docs/shared_fold_search.md#when-to-use-it).

## Exact feature-report expansion integration (2026-11-09)

The reviewed B + G-S + warm-beta Q implementation is integrated into DWD's
private dense named-linear L-BFGS reporting path. It reuses existing score
products and compensated splits, then uses guarded native gradient tiles and
eligible warm-context beta calculations. Optimizer work, model mathematics,
ordered sums, outward bounds, callbacks, acceptance thresholds and error
recovery are unchanged. There is no new public selector or default solver
change. C-dual and the smaller-buffer G-R alternative were not selected.
No ensemble-dwd source change was needed for this integration.

Native and portable wheels were built in isolated directories on Windows
`dwd_aocl` and Linux Robust `mkl202602`. Each native wheel contains the
original two accelerators plus the full-ABI feature-expansion extension and
has the `cp312-cp312` tag. Each portable wheel has no native extensions and
uses `py3-none-any`. Wheel records, all 67 package payload files, source
and license inclusion, extension linkage and loaded module origins were
checked against the frozen source. These are internal development artifacts,
not a published release or an installation into the working environments.

All four configurations passed the focused integration selection:

| Environment | Wheel | Passed tests | Skipped records | Passed subtests |
| --- | --- | ---: | ---: | ---: |
| Windows dwd_aocl | Native | 984 | 2 | 75 |
| Robust mkl202602 | Native | 983 | 3 | 75 |
| Windows dwd_aocl | Portable | 579 | 332 | 29 |
| Robust mkl202602 | Portable | 580 | 331 | 29 |

There were no failures or errors in these accepted runs. The native selection
collected 986 cases from 19 modules. Portable wheels collected 910 cases from
18 modules; the existing native-only ordered-core module contributes one
additional collection-skip record to the 911-record JUnit report. Subtests
are listed separately and are not additional top-level cases. This is a
focused regression selection, not a complete package suite or a count of
unique cases across environments.

Portable skips principally cover calls requiring absent compiled extensions.
The native skips cover an inapplicable interpreter-refusal test, two Linux
floating-point controls without an existing audited platform control, and
Windows' denied test-symlink creation. The latter is also skipped in the
Windows portable run; Windows settings were left unchanged. Independent
preflight processes verified genuine unprofiled native gradient/beta acceptance
and the portable absence/fallback routes before the pytest runs.

Coverage includes independent scalar arithmetic, exponent restoration,
cancellation and underflow, complete-result publication, refused operands,
array layouts, callback and context changes, stale-context refusal without
rebasing, missing/old extensions, warm beta and reporting dispatch, exact
fallback, build-domain and ABI rules, CI routes and source-archive contracts.
All runs used Python 3.12.15, pytest 9.1.1 and psutil 7.2.2; NumPy 2.5.4,
SciPy 1.18.1 and scikit-learn 1.9.1 were recorded. Native limits were 16 on
Windows and 20 on Robust, with existing `MKL_CBWR=AUTO` unchanged. The local
runtime census contained both AOCL-BLAS 5.3.0 and MKL 2026.2; Robust contained
MKL 2026.2. Loaded-library presence alone does not attribute each operation
to a provider. No numerical-library version cause is claimed.

The accepted child runs exited zero, with psutil verifying closure of the
direct child and descendants admitted by its identity/ancestry checks. Three
local transient discovery observations were not admitted as owned; their
identity and exit remain unverified. No forced cleanup or monitor errors
occurred in the accepted phases. This does not claim visibility of every
short-lived process. No working-package installation, dependency edit,
environment-file change or OS-setting change was made.

Earlier unsuccessful validation attempts remain recorded. The fixture was
changed to initialize the ordered helper before constructing a native context,
and an explicit stale-context refusal regression was added. Those changes
preserved mutation checks but did not repair premature context construction in
the package. The November 10 full-suite investigation identified that separate
initialization-order defect; see the correction above. Later harness checks
were narrowed to recognize one byte-pinned scalar test oracle and the portable
native-only module skip. No arithmetic mismatch was demonstrated by these
failures. The accepted native tests and portable preflight were retained when
only the portable test continuation remained. Production source and built wheel
payloads were unchanged by those earlier fixture/harness corrections.

These integration checks add no performance measurements. The completed
600 fixed-work comparison fits and 144 separate precompiled-control fits
remain the evidence for selection, including ordinary cold-start and
missing-extension costs. Larger/repeated-fit improvements are not a universal
speedup, a release-versus-release comparison, or a total-CV/convergence claim.
See [the measured scope and adverse cases](docs/kernel_dwd.md#dense-named-linear-l-bfgs-final-reporting)
and [the native build/fallback limits](docs/compiled_residual.md#exact-feature-expansion-tiles).
The new full-ABI accelerator is qualified only within its stated CPython 3.12
build/runtime domain; these runs do not establish execution on other Python
versions. Unsupported native domains retain the checked portable path.

## Shared-fold contiguous-row integration (2026-11-06)

The reviewed contiguous-row optimization is now the internal standard within
`fit_shared_fold_search`. It shares a private immutable view only for an exact
increasing contiguous row selection; other selections retain the original
gather-and-snapshot path. The helper remains explicitly selected. Ordinary
`GridSearchCV.fit`, `KernMD.fit`, and the separate `KernGDWD` CV defaults,
public eligibility rules, numerical operations and final-refit ownership are
unchanged. No ensemble package source change was needed.

The integrated implementation is byte-identical to the E candidate reviewed
in the completed comparison studies. The integration added three tested
regression modules alongside the existing shared-search and fitted-ownership
modules. Each of four isolated runs passed the same 164 pytest cases with no
skips, xfails or unrun items: Windows `dwd_aocl`, Windows `dwd_mkl`,
WSL `dwd_mkl_wsl` and WSL `dwd_aocl_wsl`. This is 656 test executions,
not 656 distinct tests or a complete package-suite run. All used Python 3.12.15,
pytest 9.1.1 and 16 native threads, with psutil monitoring of owned processes.
The runs completed with actual exit zero and captured process closure; isolated
source origins, native limits and environment restoration were verified.

Coverage includes an independent exact-arithmetic linear oracle, complete score
tables and candidate selection, ordered and duplicate indices, exact endpoint
checks and fallback, C/Fortran/strided/read-only inputs, private-view lifetime,
caller mutation, cloning, pickling, final public ownership, repeated use, and
cleanup after errors and interruptions. The endpoint metadata-width case was
validated on these 64-bit hosts; it makes no separate 32-bit validation claim.

The first WSL AOCL attempt stopped before test collection because the saved
inventory checker expected to add an AOCL descriptor that the current adapter
already reports. The AOCL library hash, direct getter version and 16-thread
setting matched the saved admission. Only the isolated test harness was adapted
to verify the single recognized descriptor against the direct getters and exact
metadata, while continuing to reject duplicates or conflicts. The failed
admission was retained, and the subsequent focused run passed. No dependency,
installed package, environment file or OS setting was changed.

These integration checks do not add performance evidence. The completed
comparisons, including measured adverse cases and the distinction between
predecessor/ordinary-search and updated-helper/predecessor comparisons, remain
as documented in [the shared-fold guide](docs/shared_fold_search.md). They do not
establish a universal speedup, a numerical-library causal effect or an updated
helper/ordinary-search ratio.

## CV default grids and estimator display (2026-11-04)

The CV representation defect and the HTML-resource decoding defect have
different causes. Tuple defaults in `GenDWDCV` and `KernGDWDCV` eliminate
scikit-learn's ambiguous comparison of an ndarray signature default and remove
shared mutable default grids. The same generated float64 values and candidate
order are retained; caller-supplied arrays are not converted or copied by this
change. The existing invalid-name `set_params` preflight remains in place.

The retained fix is limited to package source, tests and documentation. The
temporary installed sklearn backport was outside the authorized package-only
scope and has been reverted on both validation hosts. Their original sklearn
1.9.1 renderers still have the non-UTF-8 HTML limitation described below.
The passing backport checks are historical evidence for a temporary
configuration, not a claim that the current environments have been repaired.

In the isolated investigation, the tuple candidate preserved each host's
baseline results for five bounded CV workflows: default-grid dense linear and
RBF fits, custom-grid sparse linear and precomputed-kernel fits, and a kernel
keyword grid. Each variant used 229 base fits and 12 preparations. Candidate
order, folds, scores, selected parameters, coefficients, intercepts and
predictions matched its own host baseline exactly in the saved non-timing
records. These were two-update regression workloads, not convergence or
performance studies; cross-host coefficient equality is not claimed. The tuple
candidate also passed 122 selected regression tests and 22 subtests per host.

The original local HTML checks exposed a second problem in sklearn 1.9.1:
its estimator renderer opened the UTF-8 `estimator.js` asset without an encoding.
GBK/cp936 on Windows and ASCII under Linux's C locale cannot decode that asset.
Adding `encoding="utf-8"` to that one read was tested on isolated copies of
the complete installed sklearn package, with no other dependency-file delta.

On each approved host, four separate process configurations tested the
original and corrected renderer with UTF-8 mode both disabled and enabled.
Each configuration passed 35 pytest cases covering direct, inherited and MIME
HTML display across ten estimator/container shapes, plain-text display,
Unicode escaping, native/CSS read controls and bounded concurrent rendering.
The original non-UTF-8 configurations reproduced 90 expected decoding failures
per host; these are successful failure-reproduction checks, not successful
renders. Both corrected configurations rendered all 90 attempted outputs per
host/configuration. Their full HTML digests and lengths matched that host's
working original UTF-8 reference after normalizing only generated identifiers.
Text results and CSS hashes also matched. No fits were run in this HTML panel.

The eight completed HTML-stage receipts record actual exit zero and captured
owned-process closure without timeout, forced cleanup, monitor errors or
unproven discovery. The tests used Python 3.12.15, sklearn 1.9.1, pytest 9.1.1
and psutil 7.2.2, with native pool limits of 16 locally and 20 on Robust. The
local dwd_aocl environment loaded both AOCL and MKL; neither defect is attributed
to either numerical library. Earlier failures and the precollection harness
failure remain preserved. Completed numerical work was not repeated.

The isolated investigation did not alter installed dependencies or locale settings.
The HTML backport changes resource decoding, not script/CSS content or solver
arithmetic. No numerical overhead benchmark or new browser-execution claim is
made. See [dependency limitation](docs/dependency_display.md) for the current
package boundary. The later temporary-dependency checks below are separate
from this isolated evidence.


### Historical integration checks with a temporary dependency backport

Before the dependency restoration, the package source passed 172 focused tests
and 22 subtests on each approved host. The temporarily patched installed renderer
passed 35 display checks in each of two encoding modes per host. Native
GBK/cp936 on Windows and ASCII under Linux's C locale remained unchanged;
all corrected HTML outputs matched the working UTF-8 references. These
historical checks are separate from the investigation counts above and do not
establish successful non-UTF-8 rendering with the restored original renderer.

The temporary dependency bytes and distribution metadata were verified, with
original-byte backups retained. Windows owner/group restoration was needed
after replacement; the original owner, group and DACL were restored and
verified before testing. The Linux mode/ownership checks passed. Captured owned
test processes closed; one local API-stage transient discovery was not admitted
as owned, and its identity/exit remains unverified. No new package wheel or
release was published by this source integration.

The original renderer bytes have since been restored on both hosts. Restoration
verified the original 1,071-file Windows and 1,002-file Linux sklearn inventories
and seven distribution-metadata files per host. The Windows file's four hard
links and original owner, group and DACL were also verified. The tuple package
changes remain; the earlier dependency-application and test receipts are
preserved as history. No dependency backport or locale change is retained as
part of this package fix.

## Kernel fitted-state detection (2026-11-04)

`KernGDWD.cv_init()` prepares a reusable kernel cache but does not fit a
predictor. Generic scikit-learn fitted detection now uses the same four state
attributes already required by prediction. Preparation metadata alone cannot
make the estimator fitted; a successfully fitted but nonconverged model still
qualifies. Explicit attribute queries retain scikit-learn's usual semantics.

The isolated investigation reproduced the previous mismatch on local Windows
`dwd_aocl` and Robust Linux `mkl202602`. Each host passed 51 focused tests and
80 existing regression tests with 73 subtests. Coverage included preparation,
cache reuse, failure and interruption cleanup, cloning, serialization,
pipelines, subclasses, four solver backends and dense, sparse, precomputed and
rank-deficient inputs. Real baseline fitted objects retained exactly equal
saved predictions under the candidate.

The integrated regression module contains the same 50 self-contained cases;
the investigation-only baseline-file check remains separate saved evidence.
Those 50 portable cases passed on both hosts using the exact integrated source,
Python 3.12.15, pytest and psutil monitoring, with native pool limits of 16
locally and 20 on Robust. The named local environment loaded both AOCL and MKL;
this is not an AOCL-only attribution claim. No full-suite or additional
Python-version validation is claimed for this change.

A targeted test verified that direct preparation, fitting and prediction do
not invoke the new generic hook. Generic fitted-status queries perform up to
four attribute checks; no separate timing benchmark or universal zero-overhead
claim is made. Numerical code, convergence criteria and defaults are unchanged.

## Guarded native affine reporting: isolated evidence (2026-11-02)

Final kernel stationarity reporting can use the optional native evaluator for
joint affine scores on qualified dense binary64 inputs. Each row retains the
original order: expanded high products, expanded low products, then the
intercept. The original allowance, exact-zero and rational-recovery checks
still follow. Runtime, arithmetic-hook, buffer and alias guards preserve the
Python fallback when the native entry is unavailable or refuses. Public affine
prediction defaults, optimizer steps, stopping and acceptance tolerances are
unchanged. Dense reporting still requires quadratic work; this does not remove
compensated reporting or substitute separately rounded kernel scores plus an
intercept.

The isolated candidate completed 116 focused tests on local Windows dwd_aocl
and 108 passed plus eight expected prior Windows-only skips on Robust Linux
mkl202602. All 39 new native-affine tests passed on both hosts. These checks
covered exact native/Python output and reporting-count parity, cancellation,
subnormal and overflowing intermediates, exact-zero/rational recovery, buffer
ownership, custom hooks, caller rounding state and complete report/solver
outputs. These are prior isolated candidate results.

Subsequent integrated-source validation completed with 112 passed and no skips
on local Windows, and 104 passed plus eight expected prior Windows-only skips
on Robust Linux; neither run had a failure. All 39 new native-affine tests and
both public-consumer tests passed on both hosts. The Python 3.12.15 harnesses
used native thread limits of 16 and 20 respectively, with MKL_CBWR=AUTO. The
retained two-row public fixtures temporarily used one native thread and restored
the caller setting. Saved execution receipts verified state restoration and
closure of captured processes. The separate performance evidence below remains
from the isolated candidate; it was not rerun as an integration benchmark.

A separate balanced helper panel compared the frozen original Python helper
(A), the candidate portable helper (B) and the candidate native reporting path
(C). Five sizes, 32, 128, 512, 1,500 and 3,000, used the same represented dyadic
Gram matrix, coefficients and intercept within each host. Each size had two
mirrored six-arm blocks, A-B-C-C-B-A and C-B-A-A-B-C, with fixed repetitions.
Each host completed five cost tests, 336 measured helper calls, 15 warmups and
one untimed native-route probe, with zero fits. Every measured output and
reporting-count dictionary matched exactly across arms. Construction, hashes
and qualification assertions were outside the timers.

For 3,000 rows, the ranges of the two block means per helper call were:

| Target | Original Python (A) | Candidate Python (B) | Native reporting (C) |
|---|---:|---:|---:|
| Local Windows, 16 native threads | 364.243-366.743 ms | 364.546-367.072 ms | 136.906-136.995 ms |
| Robust Linux, 20 native threads | 1,066.946-1,070.489 ms | 1,080.380-1,105.657 ms | 198.546-202.424 ms |

The native path was faster in every measured block. The portable helper
timings varied: at this size B/A was about
1.00083-1.00090 locally and 1.01259-1.03285 on Robust; smaller cells had mixed
differences. B used the portable helper with the reporting flag disabled, so
this is not a measurement of an unavailable-native guard followed by fallback.
These are whole-helper measurements, not the isolated cost of one added Python
function call. Two blocks do not establish a universal speedup or a no-slowdown
claim. Process CPU includes all process threads during each interval and is
not exclusive function CPU.

Separate prepared-kernel profiles used eight fixed solver updates and two
repetitions per size, with original and candidate runs in separate sessions.
They excluded kernel construction and estimator preprocessing; they are
attribution evidence, not a balanced complete-fit benchmark or time to
convergence. Comparisons remain within each host, and no numerical-library
cause or cross-host speed claim follows. No complete suite, installed-wheel
validation or additional Python-version qualification is claimed here.

## Bounded exact linear recovery in local source

The corrected built-in dense float64 implicit Schur path now attempts bounded
exact recovery when its shifted augmented-Gram spectrum is nonpositive or its
free-intercept correction denominator is invalid. Eligibility requires a
positive represented penalty shift. The exact system is
`P = Z.T @ Z + s * diag(0, I)`, with `Z = [1, X]`, the original represented
features and represented positive shift `s`. Its next-iterate right-hand side
is `Z.T @ (Z @ theta_old - h)`, where `h` is the represented computed
loss-gradient vector used by the MM update. A cached exact rational LDL
factorization supplies each next iterate; exact adjacent rounding cells verify
nearest-even binary64 coordinates, including the intercept, before assignment.

This preserves the MM target and free intercept without eigenvalue clipping,
added regularization, feature transformation, rank truncation or an empirical
accuracy threshold. Successful ordinary paths, supplied-eigensystem validation,
the existing zero-shift rank guard, legacy/custom-hook behavior and stopping
criteria remain unchanged. The exact coordinate contract concerns the computed
`h`; it does not make nonlinear gradient evaluation or public score arithmetic
exact, certify already-accepted positive-spectrum solves, or guarantee convergence.

The recovery defaults admit at most 16 features and 65,536 feature entries.
The cumulative managed exact-work allowance is 10,000,000 units per fit, with
32,768-bit managed integer bounds and a 1 GiB modeled-workspace cap. These are
availability limits: unsupported or exhausted cases refuse rather than change
the model or weaken accuracy. Source-defined work and storage models are not
hard wall-time or RSS guarantees; conversion/serialization temporaries and
caller-retained diagnostics are separately outside those models.

### Prior isolated investigation

The isolated implementation completed 25 unique pytest items per environment:
16 exact-helper, eight routing and one cost item. Both Python 3.12 targets used
MKL_CBWR=AUTO, with 16 native threads on local Windows dwd_aocl and 20 on Robust
Linux mkl202602; the rational arithmetic itself is serial. All final items
passed without skips. Local evidence retains 16 helper and five unaffected
routing passes, plus three corrected test-plumbing cases and the cost item;
the initial failed test receipts remain preserved. Source, environment state
and captured-process closure checks passed. This is focused validation, not a
new complete package suite or installed-wheel claim.

Both natural failure cases returned on both hosts. The original far-outlier
case reached 100 updates and stopped at `max_iter`; the structural case reached
29 updates and stopped at `objective_tolerance`. Both retained
`converged_ = False`. Every applied next vector matched its exact fixed-`h`
coordinate oracle, and no increase in the separately evaluated exact original
objective was observed on those trajectories. This observation is not a new
acceptance criterion or a convergence guarantee. Native rounding can change
`h` across hosts, so complete trajectories are not claimed bit-identical.

Each environment also completed eight instrumented public fits for absolute
cost: one warmup and three measured fits per case, each with ten fixed updates.
Preparation, exact factorization/solves, rounding checks and in-fit observation
were timed; model/observer construction and independent post-fit oracle,
serialization and analysis were outside the timer.

| Case | Shape | Local median | Robust median |
|---|---:|---:|---:|
| Original far outlier | 10 x 3 | 8.9571 ms | 22.304146 ms |
| Structural intercept/feature direction | 128 x 8 | 191.6515 ms | 452.825068 ms |

These are three-observation medians for ten updates, not time to convergence,
percentage overhead against a successful baseline, a comparison with earlier
noncontemporaneous research timings, or a universal latency bound. The
structural cost is material and is justified by correctness and availability,
not memory savings. No numerical-library version regression is inferred.

### Integration validation

The integrated source passed 410 focused pytest items in each approved
environment: local Windows dwd_aocl and Robust Linux mkl202602. Both used
Python 3.12.15 and pytest 9.1.1, with MKL_CBWR=AUTO and native thread limits of
16 and 20 respectively. The selection comprises 25 maintained recovery tests
and 385 neighboring regressions: 94 zero-shift rank, 88 gradient/hook
compatibility, 152 explicit-system and 51 factor-fault tests. Every item passed;
there were no skips or expected failures.

Tests use an independent exact Gaussian-elimination and binary64-cell oracle
to check the returned vectors, each applied next state and final coefficients.
They also cover malformed supplied pairs, both recovery branches, late hook
changes, forged proof data, changed inputs, resource refusal, ordinary paths,
caching and cross-validation. The test observer wraps the actual return value;
the package contains no investigation observer or test-only logging API.

Both hosts verified frozen source hashes, isolated package imports, restored
environment/thread settings and successful actual exits. The psutil monitor
confirmed closure of captured processes; unobserved transient descendants are
outside that claim. Independent reviews cleared the source extraction, tests
and both-host results. Earlier investigation and cost receipts remain intact.
These were focused source-integration checks, not a complete suite or wheel
installation test. No benchmark panel was repeated for this integration.

## Zero-shift linear rank admission in local source

The implicit and explicit linear solvers now share the existing column-scaled
SVD rank check when the represented MM penalty shift is zero. Focused tests
cover exact augmented-design dependencies, supplied/cached eigenpairs, numerical
rank resolution, feature scaling, positive-shift preservation, public cleanup
and both package and scikit-learn cross-validation. Exact small-system oracles
check accepted steps independently of the production eigendecomposition.

Paired measurements of the isolated candidate used dense full-rank synthetic
designs of 128 x 8, 1,024 x 32 and 4,096 x 64, with 25 fixed MM updates per fit.
Each environment completed 198 fits: 18 warmups and 180 measured fits across nine
cells (zero penalty with and without a cached eigensystem, plus a positive-
penalty control). Each cell had five ABBA/BAAB blocks. Native thread limits
were 16 in the local Windows dwd_aocl environment and 20 in Robust mkl202602;
both used Python 3.12 and MKL_CBWR=AUTO. Comparisons were within each environment.

At 4,096 x 64, median paired additional time for zero-penalty fits was about
2.88 ms uncached / 3.00 ms cached locally (+71% / +90%), and 4.28 ms uncached /
4.30 ms cached on Robust (+34% / +40%). Positive-penalty controls had mixed
timing differences and skip the added SVD. Accepted scientific outputs matched
within each comparison cell. These are short, fixed-work, single-process
measurements, not time to convergence, full-CV timing, significance estimates
or universal overhead bounds. Coarse local process-CPU ticks often read zero;
the reported comparisons use wall time.

The rank check requires a dense augmented copy, including for sparse inputs,
and its cost grows with problem size. It is justified by correctness rather
than memory savings. It does not introduce a minimum-norm coefficient policy,
prove exact rank at floating-point boundaries, guarantee reliable inversion
of every ill-conditioned Gram matrix, or establish a finite unregularized
optimum for separable data. Other numerical checks remain in force.

## Version 1.3.19: current-target source validation

The pre-release source snapshot completed the same 1,245 DWD pytest items on
both current Python 3.12.15 targets:

| Target | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Windows AOCL 5.3 | 1,240 | 5 | 0 |
| Robust Linux MKL 2026.2 | 1,231 | 14 | 0 |

Both direct process exits and pytest exits were zero. All 112 collected DWD
test modules and their bound source files were checked; source readbacks were
unchanged and monitoring reported no failure or cleanup error. These are
unique pytest parameter items, not counts inflated by unittest subreports.
The saved 239-file DWD source/test/documentation manifest matched the canonical
checkout before the 1.3.19 version and release-documentation edits.

The Linux run used MKL_CBWR=AUTO from startup, with getter checks before tests,
after optional imports and at completion. Both targets began with 16 native
threads; explicit test-specific thread limits remained in effect. Platform-
specific native helpers were checked separately. The Linux compiled K*w route
is qualified only for private final reporting; Windows-only public compiled
score paths remain outside that Linux qualification.

Skips remain skips. They cover runtime/platform alternatives, wider floating-
point arithmetic, Windows-specific checks and unavailable file-symlink
privileges. The focused sparse, stationarity and startup checks below provide
additional original-equation, ownership and actual interrupted-worker evidence.
Earlier numerical-test corrections also completed source-isolated checks with
deliberate invalid-basis, retry and workflow mutations. They separate exact
workflow assertions from independent native preparations without changing
production arithmetic or acceptance.

These results validate the stated source snapshots, not newly built 1.3.19
wheels, installed-wheel imports, minimum dependencies or every Python version.
Complete-suite CI remains configured for Python 3.11 and 3.12; hosted results
for the new release are separate evidence. Focused 3.11 and 3.13 startup checks
below retain their narrower scope. No numerical-library defect, universal
performance gain, necessity of every historical optimization, or resumed
Resource18 benchmark is inferred. Release-artifact verification is separate
from this completed source validation.

## 2026-10-26 follow-up: startup recovery across CPython versions

Startup recovery no longer requires CPython 3.12 specifically. It recognizes
the standard sentinel implementation and newer native thread handles, including
the default WorkerContext argument layout. Exact post-launch traceback, worker
ownership and completion checks remain required. Compatible later CPython
versions are not rejected solely because their minor version is new.

Focused pytest checks passed on Windows CPython 3.11.15 (18 items plus 18
subtests), AOCL CPython 3.12.15 (18 items plus 18 subtests), and CPython 3.13.15
(12 startup items). Robust MKL CPython 3.12.15 passed all 18 startup/cancellation
items. Real native workers were interrupted and drained; psutil checked process
identity and disappearance. The tests retained the original exception and
cancellation behavior and checked unrelated worker survival. Native helpers and
installed environments were unchanged. Pytest and psutil remain test-only.

CPython 3.14's mechanism was source-reviewed, not runtime-tested here. This is
forward capability support, not a claim that every future interpreter or
arbitrary asynchronous-interruption window has been tested. Earlier Windows
redirector identity failures and execution-boundary denials are retained in the
validation provenance; the corrected harness and actual 3.11 run passed.

## 2026-10-26: sparse promotion, final stationarity and startup cleanup

The final isolated source artifact passed 304 pytest test items on local
Python 3.12.15/AOCL 5.3 and 295 items on Robust Python 3.12.15/MKL 2026.2.
Robust skipped nine Windows-specific cases; local skipped none. Both had zero
failures or errors. Existing unittest cases ran through pytest 9.1.1, alongside
new pytest cases. JUnit also records subtests, which are not independent test
items. A deliberate-failure control on each host confirmed that failed unittest
subtests and setup exceptions are reported rather than silently accepted.

Checks cover literal sparse duplicate arithmetic and ownership, optional conic
boundaries, independent Decimal stationarity references, unchanged solver paths
and prior refusals, actual compiled reporting versus portable reporting, and
real interrupted native-worker startup. Psutil 7.2.2 observed owned validation
processes, CPU/RSS samples and process identities; direct waits recorded actual
exit codes. Source and native-helper hashes remained unchanged during runs.
Pytest and psutil are test dependencies only. Test allocations used one native
thread by default to avoid nested oversubscription; explicit worker tests kept
their own thread controls. These are correctness checks, not speed measurements.

The paired ensemble selection passed 130 items locally and 122 on Robust, with
one and nine platform-specific skips respectively. This is focused integration
validation, not a fresh complete suite, wheel-release matrix or Python 3.11/3.13
run. This initial artifact covered the proved CPython 3.12 startup window; the
follow-up above extends it using thread capabilities. The final gradient is
a floating-point estimate, not an interval certificate. The discarded Resource18
benchmark was not resumed. Pre-existing exact-dispatch test edits were preserved.

## Version 1.3.18: automatic acceleration and optional inverse action

The applied changes passed 1,069 DWD and 947 ensemble-DWD test methods under
Windows CPython 3.12 with AOCL, with zero failures or errors. Five DWD methods
were skipped for unavailable wider floating-point arithmetic, a qualified
runtime's inapplicable refusal case, and file-symlink privileges. The ensemble
suite had one POSIX-only path skip. These are recorded application checks before
the final version metadata and forward-runtime changes, not a claim that the
release artifacts passed another full suite.

The checks cover automatic acceleration dispatch, explicit ordinary MM, saved
policies, numerical recovery, custom dense/sparse prediction helpers and the
explicit inverse action. `acceleration='auto'` selects restarted optimized
Schur MM, including checked spectral solves, and ordinary updates on reference,
legacy and L-BFGS routes. The objective, free intercept, tolerances and numerical
acceptance rules remain unchanged; finite stopping results can differ from
ordinary MM. `linear_action='factor'` remains the default. The explicit inverse
action retains the original-equation checks and recovery, owns its temporary
state per solve, and documents its setup and storage costs.

The focused forward-runtime checks below supplement these application results.
Release artifact verification and hosted CI have separate records. Timing
measurements in this document concern their named fixtures and establish no
universal speedup.

## Version 1.3.18: forward Python 3 runtime support

The forward-eligibility change passed a focused 48-method selection on Windows
using isolated source copies. Python 3.12 used AOCL; Python 3.11 used OpenBLAS.
Every configuration had zero failures or errors and included one 48-row
ensemble-DWD integration fit.

| Python | Arithmetic configuration | Passed | Skipped |
|---|---|---:|---:|
| 3.12 | Compiled helper available | 48 | 0 |
| 3.11 | Compiled helper available | 46 | 2 |
| 3.12 | Portable, without the helper | 32 | 16 |
| 3.11 | Portable, without the helper | 27 | 21 |

The two Python 3.11 compiled-configuration skips require `math.sumprod`.
Portable skips concern unavailable compiled paths and, on Python 3.11,
unavailable scalar `sumprod`. These are repeated configurations of the same
selection, not 192 distinct test methods. The unchanged compiled helper was
reused after verifying the current C and build-source hashes.

Simulated Python 3.14, 3.15 and 3.99 identities exercised dispatch on the actual
3.11/3.12 interpreters; they do not validate future interpreter arithmetic.
This change received focused checks, not complete suites or large benchmarks.
That focused check performed no new Python 3.13 run, installation, build or
release. Routine complete-suite validation targets 3.11/3.12; Python 3.13 and
later Python 3.x releases remain supported.

## Version 1.3.18: guard scanning and sparse prediction reuse

The follow-up optimization comparison holds the preceding sparse geometry and
selective RBF correctness fixes constant in both arms. It changes only guard
scanning and per-call training-norm reuse. Differential checks cover 214 flag
and kernel cases, including nonfinite distances, envelope overflow, cancellation
and self/cross inputs. New regressions also cover canonical CSR matrices and
arrays, stale ordering flags, unsafe historical bases, helper replacement,
read-only calls and the lifetime of reused norms. These are automatic internal
optimizations on qualified inputs, with the checked fallbacks retained.

Local Python 3.12/AOCL measurements used six alternating fresh-process pairs
per case, one process with 16 native threads at a time. The 98 records comprise
60 fits, two sparse-model preparations and 36 prediction trials. All saved
numerical arrays matched exactly in the 48 pairs, including ensemble selection,
disjoint subsets, coefficients, objective histories and predictions where
applicable. One additional untimed replay of the frozen MNIST ensemble case
also matched exactly, preserving its original preprocessing and parameters.

The median paired fit-time change was -2.55% for the base-heavy ensemble case;
ordinary and larger ensemble controls were approximately unchanged. Warm CSR
prediction changed by -2.20% with batches of 128 and -3.29% with batches of 32.
The small-gamma hard fit changed by +1.15%. Full-DWD fit measurements drifted
across the session in both arms and showed no clear total benefit. Unbatched
CSR timings were noisy and are not evidence of an optimization gain. These
figures describe the tested cases, not a universal speedup or elimination of
the cost of the retained correctness checks. Percentages are medians of paired
ratios, not ratios of separate timing medians. All formal trials were retained.

The final source passed the full local suites: 1,035 DWD test methods and 938
ensemble-DWD methods collected, with zero failures or errors and five and one
skips, respectively. Focused Python 3.11 and 3.13 checks each collected 130 methods with zero
failures or errors and eight skips. Skips concern platform/runtime-specific
paths, unavailable optional oracle dependencies in the focused environments,
and Windows file-symlink privileges. The actual symlink-retarget regression
was rerun with administrator privileges on the final source in Python 3.12,
3.11 and 3.13. All three runs passed without skips. These supplemental checks resolve the permission-blocked
case; the original full-suite skip counts above remain historical records.
The primary 3.12 suite retains the optional conic/oracle coverage.
Validation did not replace existing package installations or released artifacts.

## Version 1.3.18: sparse feature geometry and selective RBF correction

Sparse regression checks cover CSR/CSC/COO matrices and arrays, duplicate and
unsorted coordinates, stale ordering flags, read-only input buffers, dtype
promotion before summation, overflow, automatic SOCP penalty selection, CV
preparation and batched prediction. A genuine model fitted by released 1.3.17
with duplicate training coordinates reproduces the historical incorrect Gram
matrix. Prediction now requires refitting that model; valid canonical and
direct-distance saved models retain their behavior. Independent represented-input
oracles and the original failing input distinguish the correction from a change
in solver tolerances or acceptance.

Selective RBF checks include large common offsets, identical and nearly identical
rows, positive but ill-conditioned distances, underflow/overflow, dense/sparse
cross inputs, construction hooks, fitted-policy serialization, CV generations
and changes to NumPy error handling. An exact-fraction/Decimal oracle is
independent of the production distance formula. The global shortcut agrees
with a forced full flag scan; the self-row shortcut preserves the corrected
kernel while avoiding unnecessary diagonal work. Focused checks cover local
Python 3.11, 3.12 and 3.13, with Python 3.12/AOCL as the primary environment.

Serial paired measurements retain all trials and observe external CPU activity.
The source-path initialization optimization, combined with the sparse fix,
preserved all scientific outputs in 16 ordinary-data pairs against released
1.3.17/ensemble-DWD 0.3.13. Small full-DWD first prediction had medians
9.42 ms before and 5.51 ms after. Larger timings were mixed: one three-pair
warm-prediction comparison had a median paired increase of 7.6%, despite
unchanged prepared-kernel arithmetic. It is not attributed to a proven cause
or presented as a universal speed improvement.

The separate RBF comparison used the same sparse/initialization fixes in both
arms. With the self-row shortcut, median complete fit plus first prediction
changed from 1.109 to 1.122 seconds for 4,096 dense training rows and from
0.257 to 0.291 seconds for 2,048 sparse rows. All saved ordinary-case numerical
arrays were identical. Pathological offset cases needed substantially more
direct-distance work and corrected large kernel errors. These are diagnostic
cost measurements with fixed settings, not new classifier-accuracy studies or
proof of stationarity. They do not establish universal timing, PSD or decision
sign guarantees; see the [construction contract](docs/kernel_dwd.md#rbf-construction-and-query-consistency).

A subsequent 24-trial integrated comparison preserved all saved final numerical
arrays in 12 paired fits, but exposed a consistent larger-ensemble fitting cost
before scan optimization (median paired increase 37.8%). Profiling attributed
most of that cost to base-score flag scanning and visiting empty packed rows;
that ordinary fixture required no distinct-pair direct-distance repairs.
Intermediate consensus hashes differed from the released implementation, so
equality of final outputs is not a claim that every intermediate value stayed
identical. The scan optimization retains the flag criterion and arithmetic,
uses bounded 256-by-256 tiles and visits active rows in their original order.
Independent equivalence checks cover 251 controls on each of Python 3.11, 3.12
and 3.13. The earlier timing panel describes the implementation before this
scan optimization; it is retained as diagnostic evidence, not presented as a
measurement of the optimized implementation. Exact artifact identities and
subsequent measurements are recorded with the local validation results.

## Version 1.3.17: recovery, literal diagnostics and legacy warnings

Eligible typed failures in restarted kernel MM can take one fresh spectral
retry from the original initialization. The retry preserves the objective,
solver tolerances and numerical acceptance rules. Restart acceleration remains
explicit: objective-change stopping depends on the optimization trajectory,
so agreement with ordinary MM at a finite stopping threshold is not promised.

The explicit `literal_objective_diagnostics` function evaluates the represented
stored kernel, using `K` for scores and `(K + K.T) / 2` for the quadratic penalty.
It encloses the objective and coefficient/intercept gradients and reports a
dual lower bound within its declared arithmetic contract. It is limited to
small inputs, with at most 16 rows and bounded precision and operation counts.
It does not alter fitting or certify the analytic Gaussian kernel independently
of its stored representation. See the [kernel guide](docs/kernel_dwd.md).

New valid legacy fits emit a `FutureWarning`; invalid inputs retain their
validation errors. The historical update equations and prediction from saved
models remain available. Branch-only push CI avoids duplicate release-tag
runs without removing test jobs, pull-request checks or manual runs.

Application checks on 2026-10-19 passed 57 focused tests on each of Python 3.11,
3.12 and 3.13, using both isolated native and portable wheels. Every configuration
had zero failures, errors and skips. The exact installed runtime payloads were
verified. Checks used Windows CPython 3.12 with AOCL as the primary runtime and
isolated Python 3.11/3.13 environments for focused validation. These checks did
not update existing package installations.

The prior full DWD run executed 960 test methods with no failures or errors and
four platform/runtime skips. That full run was reused, not repeated: all 43
Python runtime files at the application stage had identical syntax trees to
the validated sources, and the compiled helper is byte-identical. The new focused checks
supplement this existing coverage. Strict Sphinx documentation checks also
passed. These are correctness and integration checks, not new performance
benchmarks or hosted CI results.

## Version 1.3.17: automatic policies and bounded internal reuse

Version 1.3.17 defaults new kernel estimators to automatic
affine scoring and residual ordering. Corrected Schur fits resolve affine
scoring to joint evaluation; eligible optimized Cholesky fits resolve residual
ordering to adaptive checks. Explicit historical policies remain available,
and older fitted objects retain their recorded behavior. The objectives,
free intercept, solver tolerances and stopping defaults are unchanged.
Restarted MM and L-BFGS remain explicit options. Earlier sections describe
their dated validation scope and do not override these current defaults.

Local Windows checks in Python 3.12 covered the combined policies, cloning,
CV forwarding, supplied eigensystems, spectral recovery, failed refits and
exact-zero affine certification against independent rational arithmetic.
Saved pre-change model predictions were rechecked on actual Python 3.11,
3.12 and 3.13. Unsupported gradual-underflow modes remain rejected by the
shared input guard; the zero shortcut does not bypass that rejection.

The native coefficient preparation and solver-scoped worker changes passed
81 discovered tests on each of those three runtimes: 79 passed with two
runtime-qualification skips on Python 3.11, and 80 passed with one skip on
Python 3.12 and 3.13. Separate allocator fixtures covered allocation refusal
and cleanup. The 14 new RBF norm-reuse tests passed on each runtime, including
fresh call-local state, original-route score parity, custom-hook fallback and
concurrent read-only prediction. No persistent norm cache was introduced.

These checks used Windows CPython 3.12 with AOCL as the primary runtime, with
isolated Python 3.11/3.13 environments for focused qualification. They are
correctness checks, not new performance measurements or a universal speedup
claim. These dated checks precede the release version and metadata changes;
release-artifact verification and hosted CI have separate records. Installation
into application environments is also separate from validation.

## Version 1.3.15: development validation

Before release packaging, local Windows checks under conventional CPython 3.12
passed 141 tests for complex-input rejection, existing numerical methods,
eigendecomposition validation, API behavior, failed refits, compatibility and
cross-validation. There were no failures, errors or skips. A separate set of
16 valid-real comparisons exactly matched pre-change coefficients, intercepts,
objective histories, scores and predictions. These comparisons covered linear
and kernel DWD, Schur and legacy modes, float32 and float64 inputs, lists and
real object-array initial coefficients. The baseline source identity was checked
before the changes.

The applied finite-grid reporting changes passed 71 tests: 62 reporting,
resumable-CV and exact-accuracy checks, plus 9 standalone checkpoint and CLI
checks. There were no failures, errors or skips. This is a separate test scope
with overlap in the earlier 141-test run, not an additional count of unique
tests. Reporting preserves duplicate candidate labels, exact fold accuracy,
first-maximum selection and final refitting. Compact diagnostics retain existing
counts without performing extra solver checks; unavailable counts remain
unknown rather than becoming zero.

These Windows CPython 3.12 checks used one native thread for small fixtures;
the CV integration checks also included their existing two-worker regression.
These completed runs are correctness checks, not a performance benchmark or
evidence of zero overhead on all inputs.

The development results do not establish that earlier real-data studies used
complex inputs or that historical results were corrupted. They also do not
certify newly built release artifacts or other Python, operating-system or
hardware configurations. Built-wheel and CI validation are separate checks.

## Version 1.3.14: development validation

Before release packaging, the frozen development sources passed 59 DWD tests
covering restarted/adaptive MM, residual ordering, CV native threads, memory
planning, resume identity and restoration of caller thread limits. The paired
ensemble-DWD sources passed 90 integration tests with this DWD source. There
were no failures, errors or skips in those focused runs. They used Windows,
conventional CPython 3.12.14 and AOCL; these source checks are separate from
built-wheel and cross-version release checks.

The resource tests cover successful, failed and interrupted work, final-refit
failures, complete checkpoint replay, exact memory-ratio flooring, and mutation
of thread limits inside callbacks. Defaults remain one CV worker and one native
thread. Memory values are planning allowances supplied by the caller, not peak
memory measurements or operating-system limits. CV native threads enter resume
identity; final-fit threads retain their independent scope.

Adaptive-order confirmation used four saved TCGA cases at both one and 16
native threads, with two balanced default/adaptive pairs per case and thread
count: 32 complete pipeline fits. Median paired adaptive/default fit-time ratios
were:

| Saved case | One native thread | 16 native threads |
|---|---:|---:|
| Ensemble DWD, difficult split 3 | 0.716 | 0.707 |
| Ensemble DWD, difficult split 9 | 0.711 | 0.715 |
| Ensemble DWD, healthy split 7 | 0.996 | 1.054 |
| Full-kernel DWD, split 10 | 0.854 | 0.854 |

These observations support retaining an option, not changing the default. The
healthy 16-thread case was about 7 ms slower. All compared saved arrays matched:
final coefficients, intercepts, histories, decisions, predictions, selected
rows and subsets. Every learner retained its iteration count and final objective
and passed the unchanged numerical requirements. This is not a direct comparison
of every base learner's coefficient array or a general guarantee of identical
finite-iteration states under both orders.

For difficult ensemble split 3, native refinement trials fell from 1,155 to 46
while compensated checks remained 1,177. For split 9, trials fell from 735 to 12
while compensated checks remained 735. Required accurate checks were preserved;
the avoided trials were speculative corrections that would have been discarded.

An earlier 64-fit engineering comparison encountered concurrent numerical work
and is excluded from the timing conclusions above. The 32 confirmation fits had
no competing Python process or observer error in 50 ms observations; this does
not exclude all OS or non-Python activity. Historical experiment timings were
not replaced. No Intel or Ubuntu performance conclusion follows from these runs.

The release CI configuration covers conventional Python 3.11, 3.12 and 3.13
on Windows and Ubuntu with base and optional SOCP dependencies. A configured
matrix is not itself evidence of a passed run, and the source results above do
not certify a rebuilt wheel. The compiled helper retains its existing narrower
numerical runtime guards. Python 3.10 companion maintenance has ended; references
to its checks below describe historical releases only.

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
injected.

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

The final source passes 412 tests, including 22 new regression
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

The source was validated with CPython 3.12.14, NumPy 2.5.2, SciPy 1.18.0,
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
