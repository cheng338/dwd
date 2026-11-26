# Resumable kernel DWD cross-validation

The examples directory provides fixed-grid cross-validation for binary
`KernGDWD` with recovery after interruption. Completed candidate/fold evaluations
are saved separately. A resumed run reuses compatible completed evaluations and
reruns unfinished ones. The existing `dwd.cv.run_cv` API is unchanged.

The example uses the estimator's existing fitting, scoring and matrix
preparation. It retains the supplied solver settings, candidate ordering,
equal weighting of fold scores, and selection of the first candidate at an
exact accuracy tie. It does not extend or refine the grid.

## Command-line example

Run from the repository checkout with the package and its dependencies installed.
Provide a local NumPy `.npz` file containing dense feature array `X` and binary
label array `y`. These are training data for cross-validation. Keep any final
test set separate from the search.

```shell
python examples/tune_kernel_dwd.py --data training.npz --run-dir runs/kernel-dwd --lambd 0.01 0.1 --gamma 0.1 1.0 --folds 5 --seed 42 --jobs 2
```

The command-line example uses an RBF kernel and requires explicit lambda and
gamma grids. It does not download data. The example values above illustrate the
syntax; suitable values depend on the data and preprocessing.

Repeat the same command with `--resume` to continue. Use `--scale` to fit a
`StandardScaler` on each fold's training rows and apply it to that fold's
validation rows. The final scaler is fitted on all supplied training rows.
Keep the scaling choice unchanged when resuming.

The defaults are five folds, seed 42, one fold worker, one native thread per
detected numerical-library pool, and no scaling. The CLI
uses shuffled stratified folds; `--seed` controls both their construction and
the estimator seed.
`--max-iter` defaults to the estimator's 100-update cap; other solver defaults
remain those of `KernGDWD`. Pass `--cv-only` to select parameters without a final
fit. Without it, the selected model is fitted afresh on all supplied training
rows, including when every CV evaluation was loaded from a checkpoint.

## Python interface

From the examples directory, or with that directory on the Python import path:

```python
from dwd.gen_kern_dwd import KernGDWD
from resumable_kernel_cv import run_resumable_cv

result = run_resumable_cv(
    KernGDWD(kernel='rbf', random_state=42),
    X, y,
    params={
        'lambd': [0.01, 0.1],
        'kernel_kws': [{'gamma': 0.1}, {'gamma': 1.0}],
    },
    run_dir='runs/kernel-dwd',
    cv=5,
    scoring='accuracy',
    jobs=1,
    resume=False,
    scale=False,
    refit_best=True,
    final_native_threads=1,
    cv_native_threads=1,
)
```

An integer `cv` follows scikit-learn's default unshuffled stratified splitting
for classifiers. To use shuffled folds, supply a `StratifiedKFold` with an
explicit integer `random_state`. The realized train/validation indices are
included in the checkpoint identity.

New estimators default to `affine_computation='joint'`. The former `'auto'`
alias is no longer accepted; replace it with `'joint'` in estimator settings
and parameter grids. Explicit `'standard'` selects separate dot-product and
intercept arithmetic; see [choosing numerical options](choosing_options.md)
before overriding the precision-oriented joint default. The policy
is recorded with estimator and candidate parameters; changing it requires a
new run directory. Completed evaluations retain the chosen arithmetic when
resumed, and the selected estimator uses it in the final refit. The source
identity also includes the affine scoring helper. See
[affine scoring](kernel_dwd.md#optional-joint-affine-scoring) for costs
and numerical scope.

`KernGDWD(..., exact_recovery='extended')` selects the optional larger recovery
capacity for each fit. It is also part of estimator, candidate and source
identity, so a checkpoint cannot be reused after changing this setting. The
default is unchanged. See [extended exact recovery](kernel_dwd.md#optional-extended-exact-recovery)
for its small-kernel scope and work limits.

The return dictionary contains `best_params`, `best_score`, `best_clf`, `scaler`,
`agg_results`, `all_cv_results`, `work_summary` and `finite_grid_report`.
`best_clf` is `None` when `refit_best=False`; `scaler` is also `None` in that case.
When scaling is enabled and the model is refitted, apply the returned `scaler` to new feature
rows before calling the returned model's prediction methods. With scaling
disabled, `scaler` is `None`. No fitted model is serialized in the checkpoint.

## Recovery and compatibility

Receipt schema 2 records accuracy counts and an explicit aggregation policy.
Checkpoints from earlier source versions require a new run directory; they
are not upgraded by inferring missing counts or diagnostic facts. Existing
checkpoint files remain untouched.

Each saved evaluation belongs to a specific candidate and fold. Compatibility
checks cover the data, materialized folds, candidate settings, scaling, source
and numerical runtime. A changed search requires a new run directory. Existing
checkpoints are not silently treated as a new run. You may change `jobs` and
`refit_best` when resuming with the same source and runtime. The selected
model is fitted afresh with one estimator worker and one native thread.
Changing `cv_native_threads` requires a new run directory, including when all
folds are saved.
A corrupt or incompatible receipt stops the run
before new CV fitting; it is not silently dropped. Source and runtime files
must remain unchanged while the process is running.

Resume diagnostics distinguish a checksum failure from an identity mismatch.
A checksum failure means the saved payload does not match its checksum; keep
the evidence and investigate the file's integrity instead of editing its hash.
For an authenticated run manifest with changed settings, the error requests
a new run directory and names a bounded set of changed top-level identity
fields when available. Field values are not printed. An identity mismatch in
another receipt means its stored run identity does not match the expected one;
it does not by itself prove
that the invocation changed. Malformed or internally inconsistent saved
metadata remains an error, and none of these diagnostics repairs or permits
reuse of incompatible work.

The runtime identity records the compiled accelerator's availability at startup,
native-screen support and, when loaded, the extension's path and file hash.
Different startup states cannot share checkpoints even when the package files
are identical. The extension must belong to the identified DWD installation.
Runs without the optional compiled accelerator can still resume when their
recorded runtime matches.

When joblib runs folds in separate processes, the example starts workers for
that search and closes them when it finishes. Other joblib calls in the same
process can continue using their workers. Each new process-based search pays
the worker startup cost. Use `jobs=1` to run folds serially. An enclosing
joblib configuration that requires shared memory can select threads, and
joblib's usual restrictions on nested parallelism still apply.
If a threaded search is interrupted or a fold fails, the call waits for active
folds to release their checkpoint files and native thread settings before it
returns the error. A fit already in progress can delay that return. Completed
evaluation receipts remain available on resume.

If the calling process has already imported CVXPY or SCS, parallel fold workers
import the same solver modules before checking their native libraries and thread
counts. Both modules are optional for this example. For a new process resuming
a run, use the example source and loaded numerical runtime recorded for that
run. Load optional solvers before the first checkpointed search in the process.
Loading additional native libraries after the runtime has been recorded can
invalidate subsequent checkpoint operations.

Checkpoints created before these accelerator fields were added cannot be resumed
with the updated example. Start a new run directory, or use the matching old
example and runtime to resume the earlier run; missing fields are not inferred
or migrated.

The single-thread final-refit update changes the example source identity.
Use the matching source and runtime to resume an earlier checkpoint, or
start a new run directory with the updated example.

The finite-grid reporting update changes `resumable_kernel_cv.py`, whose source
hash is part of checkpoint identity. A checkpoint created before this update
cannot be resumed with the updated example. Use the matching earlier source
and runtime to continue that run, or start a new run directory. Existing
receipts are preserved and are not migrated to the new source identity.

## Allocating CV workers and native threads

`--cv-native-threads N` optionally changes the numerical-library thread limit
inside each fold worker; the default remains one. Candidates within each fold
still run sequentially so they can reuse the existing matrix preparation.
For example, `--jobs 4 --cv-native-threads 4` allows four simultaneous folds
with four native threads each. Five folds cannot occupy more than five fold
workers. Fewer unfinished folds also reduce the active worker count on resume.
Choose the product of workers and native threads to fit the CPU budget;
additional threads do not guarantee a faster solver.
These are numerical-library limits and nominal allocations, not a limit on all
operating-system threads. In the Python interface, `jobs=-1` divides the detected
logical CPU count by `cv_native_threads`, rounding down but allowing at least one
worker, then caps it by unfinished folds and any supplied memory allowance.
An explicit positive `jobs` value remains the caller's concurrency choice.
Choose a native-thread count that itself fits the available CPU budget.
The checkpoint checks that every detected numerical-library pool reports the
requested count. A loaded library built for one thread can satisfy a request
for one. For a larger count, use a process in which every pool supports that
count; otherwise the run refuses to record results.

Two optional arguments can further cap concurrency:
`--cv-memory-budget-mib B --cv-worker-memory-mib W`. Both are required together.
The worker count is at most `floor(B/W)`, and a budget smaller than one worker
allowance is rejected. These are **caller-supplied planning allowances**, not
automatically measured model peaks or operating-system memory limits. Estimate
the largest fold's data copies, dense matrices, solver workspaces and scoring
buffers, include headroom, and leave room outside `B` for the coordinator and
other applications. A mistaken allowance can still exhaust memory. This cap
applies to CV only; the final full-data fit needs its own memory assessment.

The Python names are `cv_native_threads`, `cv_memory_budget_mib` and
`cv_worker_memory_mib`. Worker and allowance settings may change on resume;
the CV native-thread count may not. Completed receipt timings retain their
original concurrency and should not be pooled as controlled timing benchmarks
after a scheduling change. `work_summary.cv_resource_plan` records the current
request, unfinished folds, effective workers, native-thread count and supplied
memory allowances. It reports zero active workers on complete CV replay.
The example restores the caller's thread limits after success or failure.

Full-data scaling and the timed final fit run with one estimator worker and
one native thread regardless of the CV allocation. The final runtime is
recorded separately. Add the reported final scaling and estimator-fit
intervals to obtain the time for the complete preprocessing-plus-fit procedure.

The run directory contains `identity.json`, `generation.json` and completed
evaluations under `receipts/fold-0000/candidate-000000.json`, with zero-based
fold and candidate indices. Process locks prevent concurrent invocations from
writing the same run. If an earlier invocation or fold worker still owns the
run, a new invocation raises a busy error. The lock files may remain after
processes finish; their presence alone does not indicate an active owner.

An evaluation is saved only after fitting and scoring finish. After an
interruption, completed evaluations remain available and unfinished evaluations
run again. Saved matrix preparation is not serialized: a resumed fold rebuilds
it when an unfinished candidate first needs it. Compatible later candidates in
that worker reuse it through the estimator's existing `cv_init` mechanism.

### Abrupt termination and leftover files

A forced process termination, such as `SIGKILL` on Linux or forced termination
on Windows, bypasses Python exception handling and cleanup. It can leave an
unpublished `<destination>.<uuid>.tmp` file. Such a file is scratch storage,
not a completed evaluation: do not rename it to a receipt or use its contents
to claim completion. Ordinary exception handling attempts to remove the
temporary file but cannot promise cleanup when the process or storage fails.

When the run identity was committed, use the normal `--resume` path with its
matching settings, source and runtime. It validates completed evaluation
receipts and checks ownership before accepting more work. A surviving fold
worker still holding its lock blocks a new coordinator. A process exit does
not prove that all its workers have exited, and the presence or age of a lock
file does not establish a live or dead owner. Keep the stable lock files;
deleting one can let different processes lock different underlying files.

No manual cleanup is required merely because unpublished scratch remains.
If disk reclamation is necessary, first establish that every process using
that run has stopped, then inspect only the recognized unpublished scratch
files. Preserve committed receipts, identity and generation records, and
lock files. Do not use an age-based sweep or delete uncertain files to bypass
a busy, corrupt or incompatible-run refusal. If the initial identity was not
committed, follow the separate startup procedure below. These process-stop
semantics do not guarantee storage durability through a power failure.

### Interrupted initial checkpoint setup

If the process stops before it publishes `identity.json`, no committed run
identity or completed evaluation is available to resume. Retry the same command
without `--resume` (or call `run_resumable_cv(..., resume=False)`). A fresh retry
is allowed when the directory contains only `coordinator.lock` and regular,
single-link, non-reparse files named `identity.json.<32 lowercase hexadecimal
characters>.tmp`. The coordinator lock must be acquired first; another live
coordinator still causes a busy error.

These temporary files are unpublished scratch, not trusted checkpoints. Their
contents are not read or adopted, and their bytes and names are preserved for
diagnosis. The retry creates the identity from the current invocation and
publishes a new generation normally. A complete-looking temporary file does not
authorize resume or reuse of work.

Once `identity.json` is published, a fresh attempt still refuses the directory;
use `--resume` with its matching invocation, even if generation publication was
interrupted next. A generation file, fold lock, receipt directory, unknown file,
symbolic link, hardlinked temporary or other unsupported entry without an
identity is not treated as initial scratch. Diagnose that directory or choose a
new one. Do not remove or replace `coordinator.lock` to force takeover.

This narrow recovery covers process interruption during first identity
publication. It does not claim recovery from arbitrary checkpoint deletion,
filesystem corruption or power-loss durability. Existing source/runtime identity
checks remain strict; updating the example does not make checkpoints created by
different source versions interchangeable.

Timing from saved evaluations describes their original execution. The work
summary separates that history from the current invocation's work. It cannot
recover time spent in an unfinished evaluation or infer time saved by replay.
A final fit is always fresh when requested; replaying all CV evaluations does
not reload an earlier model.

## Reading the results

After a successful invocation, `cv-results.json` records the selected parameters,
CV scores, `finite_grid_report` and final-fit diagnostics. `work-summary.json`
records the current work summary. These files are conveniences for inspection;
resume uses the checked evaluation receipts. A summary-file write failure produces a warning
and does not discard the returned fitting result. If work-summary calculation
fails, its coverage is marked unavailable while the numerical result is
returned. The CLI also prints the selected parameters, a readable finite-grid
report and the work summary.

`best_score` is the largest mean validation accuracy. Candidates remain in the
same order as `dwd.cv.run_cv`, including duplicate settings; an exact tie selects
the first candidate. Termination diagnostics are saved for inspection and do
not silently exclude a candidate with valid finite scores.

### Interpreting the finite-grid winner

The winner is best among the supplied candidates. The report does not establish
an optimum outside that grid or claim that a search has converged. It does not
add candidates, change selection, trigger extra fits or alter final refitting.
The first candidate attaining the largest exact, equally weighted fold accuracy
remains selected; duplicate parameter labels retain their original indices.

`finite_grid_report` records the exact number of tied best candidates and two
score gaps. The gap to the second candidate is zero when the top score is tied,
including repeated candidate labels. The gap to the next strictly lower
distinct score describes separation from the next lower score; it is `None`
when no such score exists. A singleton grid has neither gap. The `all_flat`
flag requires at least two candidates, all with identical mean accuracy. Exact
numerator/denominator fields accompany the reported accuracy and gaps.

Numeric positions are assessed within conditional slices that hold every other
supplied parameter exactly fixed. For example, a `kernel_kws.gamma` endpoint
is relative to candidates with the same lambda and other kernel settings,
not every gamma appearing anywhere in the grid. Distinct floating-point values
are not rounded together, and candidate labels are not merged. A singleton
slice means only one distinct numeric value was supplied in that slice; it is
not evidence of an endpoint optimum. Missing, categorical
or unsupported coordinates are reported as unknown rather than replaced by
assumed defaults.

Tie and coordinate detail lists are limited to ten entries, with total counts
and truncation flags retained. Full candidate labels and fold results remain
in `all_cv_results`. If reporting fails, the report is marked unavailable and
the valid numerical result is still returned. The report is observational;
completed evaluation receipts remain the authority for checkpoint recovery.

## Accuracy counts and diagnostics

Both this example and `dwd.cv.run_cv(..., scoring='accuracy')` compare the
exact mean of each fold's correct/sample-count proportion. Folds retain equal
weight even when they have unequal sizes; scores are not pooled across rows.
The correct count is recovered from the known unweighted accuracy and sample
count, with an exact round-trip check. This requires no second prediction.
The generic `run_cv` behavior for custom or other named scorers is unchanged.

Fold results include `train_correct`, `test_correct`, `train_count` and
`test_count`. Aggregate results include reduced numerator/denominator pairs in
`mean_train_score_fraction` and `mean_test_score_fraction`; ordinary mean fields
contain the float conversion of those exact means. Standard deviations retain
their existing floating-point calculation. Mathematical ties select the first
candidate, even if a floating-point average would differ by one rounding unit.

Diagnostics separately record the objective, iteration work, gradient and
stationarity residuals, and the reported objective/optimality/stopping flags.
`solver_summary` retains bounded scalar timing, recovery and certificate facts
already reported by the solver. It does not save coefficient arrays or
iteration histories. Missing attributes remain `None` or absent, never a
fabricated zero or a convergence claim. Nested solver timing categories can
overlap and must not be summed as independent costs. A zero dual endpoint
retains its recorded kernel assumption and is not a PSD certificate.

The `work_summary` describes only the current invocation:

- `executed_evaluations` and `replayed_evaluations` count candidate/fold results.
- `current_seconds` separates preparation, fitting, scoring and scaling for new
  evaluations, plus final fitting, final scaling and elapsed wall time.
- `historical_seconds` contains the saved preparation, fitting, scoring and
  scaling durations of replayed evaluations.
- `fold_workers` is the number of workers used, which is zero on complete CV
  replay. `final_refit` records whether a fresh final model was fitted and
  includes its termination diagnostics, `estimator_workers` (one when fitted,
  zero when skipped), `native_threads_requested` and the observed `runtime`
  fingerprint. The runtime is `None` when final fitting is skipped.

Preparation, fit, score and scaling totals sum elapsed worker intervals. They
are neither CPU time nor overall wall time. The wall interval begins on entry
to `run_resumable_cv` and ends when its summary is assembled; it excludes caller
data loading, subsequent summary-file writing and context cleanup. `all_cv_results` and `agg_results` retain saved fit durations on
replay. Each fold's `init_time` in `all_cv_results` sums preparation durations
from its receipts, including any matrix rebuilding after an interruption.

## Parallel execution

`jobs=1` and `cv_native_threads=1` are the defaults. Larger positive `jobs`
values permit separate fold worker processes, bounded by unfinished folds and
any supplied memory allowance. In the Python interface, `jobs=-1` divides the
logical CPU count by the requested native threads per worker, with at least
one worker before applying those bounds; the CLI accepts positive `--jobs`
values. See [allocating CV resources](#allocating-cv-workers-and-native-threads)
for the native-thread option, allowance calculation and limits of this plan.
Candidates within each fold remain serial to share compatible matrix preparation.

Each active fold may require its own dense kernel and solver workspace. Choose
`jobs` according to available memory as well as processors. A timing comparison
on a 20-row dataset used two workers in both versions. It timed complete
searches with a reusable worker pool and with a fresh pool per search. To
measure speedup on your data, compare `jobs=1` with
your chosen worker count using the same folds, candidate grid and native-thread
limit. Use a fresh checkpoint directory for each run so both perform the same work.

The compatibility argument `final_native_threads` (CLI `--final-native-threads`)
accepts only `1`. During the final phase, the sequential joblib backend also
limits named kernels that request workers through `kernel_kws['n_jobs']`.
The example verifies the final native pool counts and restores the caller's
joblib and native thread settings on success or failure. The fitted estimator
retains its selected parameters for later predictions. Resuming an interrupted
final fit reuses compatible CV receipts and fits a fresh full-data model.

## Supported scope

This example is limited to `KernGDWD` itself, with finite dense real features,
exactly two classes, an explicit integer estimator seed and accuracy scoring.
Numeric, Boolean and string labels are supported. Each fold must contain both
classes in its training and validation rows, with no overlap between those
two sets. Callable scorers, callable kernels, callbacks, random number generator
objects, validation stopping and tuning the estimator seed
are not supported by this example. Named feature kernels are supported;
`kernel='precomputed'` is rejected.

Fold-local feature scaling is optional and forms part of the search identity.
It changes the feature representation supplied to DWD. Checkpoint recovery
retains that choice and the DWD loss, regularization, free intercept and MM
update equations.

## Extreme-input prediction qualification

The example owns its training arrays to protect checkpoint consistency from
caller mutation. With the default `rbf_computation='standard'`, on extreme
unscaled RBF data, querying the caller's original
training array can differ from an older model retaining that exact object,
because the standard kernel implementation treats self and copied queries
differently in floating-point arithmetic. See the
[release validation scope](validation-1.3.10.md) for the reproduced case and
matched-query controls. Optional
[direct RBF computation](kernel_dwd.md#optional-direct-rbf-computation) addresses
that demonstrated kernel-construction discrepancy. Joint affine scoring alone
cannot repair kernel-entry errors. Universal bitwise prediction equivalence
is not claimed.
