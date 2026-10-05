# Resumable kernel DWD cross-validation

The examples directory provides fixed-grid cross-validation for binary
`KernGDWD` with recovery after interruption. Completed candidate/fold evaluations
are saved separately. A resumed run reuses compatible completed evaluations and
reruns unfinished ones. The existing `dwd.cv.run_cv` API is unchanged.

The example uses the estimator's existing fitting, scoring and matrix
preparation. It retains the supplied solver settings, candidate ordering,
arithmetic used to aggregate fold scores, and selection of the first candidate
at an exact tie. It does not extend or refine the grid.

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

The defaults are five folds, seed 42, one fold worker and no scaling. The CLI
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
)
```

An integer `cv` follows scikit-learn's default unshuffled stratified splitting
for classifiers. To use shuffled folds, supply a `StratifiedKFold` with an
explicit integer `random_state`. The realized train/validation indices are
included in the checkpoint identity.

The return dictionary contains `best_params`, `best_score`, `best_clf`, `scaler`,
`agg_results`, `all_cv_results` and `work_summary`. `best_clf` is `None` when
`refit_best=False`; `scaler` is also `None` in that case. When scaling is
enabled and the model is refitted, apply the returned `scaler` to new feature
rows before calling the returned model's prediction methods. With scaling
disabled, `scaler` is `None`. No fitted model is serialized in the checkpoint.

## Recovery and compatibility

Each saved evaluation belongs to a specific candidate and fold. Compatibility
checks cover the data, materialized folds, candidate settings, scaling, source
and numerical runtime. A changed search requires a new run directory. Existing
checkpoints are not silently treated as a new run. You may change `jobs` or
`refit_best` when resuming. A corrupt or incompatible receipt stops the run
before new CV fitting; it is not silently dropped. Source and runtime files
must remain unchanged while the process is running.

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

Timing from saved evaluations describes their original execution. The work
summary separates that history from the current invocation's work. It cannot
recover time spent in an unfinished evaluation or infer time saved by replay.
A final fit is always fresh when requested; replaying all CV evaluations does
not reload an earlier model.

## Reading the results

After a successful invocation, `cv-results.json` records the selected parameters,
CV scores and final-fit diagnostics, and `work-summary.json` records the current
work summary. These files are conveniences for inspection; resume uses the
checked evaluation receipts. A summary-file write failure produces a warning
and does not discard the returned fitting result. If work-summary calculation
fails, its coverage is marked unavailable while the numerical result is
returned. The CLI also prints the selected parameters, mean CV accuracy and
work summary.

`best_score` is the largest mean validation accuracy. Candidates remain in the
same order as `dwd.cv.run_cv`, including duplicate settings; an exact tie selects
the first candidate. Termination diagnostics are saved for inspection and do
not silently exclude a candidate with valid finite scores.

The `work_summary` describes only the current invocation:

- `executed_evaluations` and `replayed_evaluations` count candidate/fold results.
- `current_seconds` separates preparation, fitting, scoring and scaling for new
  evaluations, plus final fitting, final scaling and elapsed wall time.
- `historical_seconds` contains the saved preparation, fitting, scoring and
  scaling durations of replayed evaluations.
- `fold_workers` is the number of workers used, which is zero on complete CV
  replay. `final_refit` records whether a fresh final model was fitted and
  includes that model's termination diagnostics.

Preparation, fit, score and scaling totals sum elapsed worker intervals. They
are neither CPU time nor overall wall time. The wall interval begins on entry
to `run_resumable_cv` and ends when its summary is assembled; it excludes caller
data loading, subsequent summary-file writing and context cleanup. `all_cv_results` and `agg_results` retain saved fit durations on
replay. Each fold's `init_time` in `all_cv_results` sums preparation durations
from its receipts, including any matrix rebuilding after an interruption.

## Parallel execution

`jobs=1` runs folds serially. Larger positive values permit separate fold worker
processes, bounded by the number of unfinished folds. In the Python interface,
`jobs=-1` permits all logical processors, subject to that same bound. Each
process uses one native numeric thread. Candidates within a fold remain serial
so compatible candidates can share matrix preparation.

Each active fold may require its own dense kernel and solver workspace. Choose
`jobs` according to available memory as well as processors. Parallel execution
has not been benchmarked for this example. Small folds can spend more time
starting workers than they save through concurrent fitting.

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
caller mutation. On extreme unscaled RBF data, querying the caller's original
training array can differ from an older model retaining that exact object,
because the existing kernel implementation treats self and copied queries
differently in floating-point arithmetic. See the
[release validation scope](validation-1.3.10.md) for the reproduced case and
matched-query controls. Universal bitwise prediction equivalence is not claimed.
