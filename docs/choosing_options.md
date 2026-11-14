# Choosing DWD options

This guide explains when to depart from the current defaults. It covers public
numerical policies and explicit workflow helpers, rather than ordinary model
selection over `lambd`, `q`, kernels and preprocessing. Those choices still need
training-only validation. An optional setting is neither a universal improvement
nor a promise to repair every numerical failure.

For a first generalized DWD fit, retain zero initialization and objective-change
stopping. Kernel DWD additionally defaults to automatic backend, acceleration
and residual ordering, standard RBF construction, joint affine scoring, standard
exact-recovery capacity and factor-based linear action. Linear `GenDWD` defaults
to `implicit_P=True`. These defaults do not certify convergence: inspect the
returned model and its diagnostics. The detailed [kernel guide](kernel_dwd.md)
defines supported combinations and numerical acceptance rules.

## Stopping and observation

| Choice | Use it when | Tradeoff and check |
|---|---|---|
| `stopping='objective'` | Ordinary predictive work is adequately served by a small change in training objective and the returned model passes your validation requirements. | This default is a progress heuristic. A reached objective-change criterion does not imply that `converged_` is true or that the stationarity residual is small. |
| `stopping='optimality'` | A specified stationarity target is part of acceptance, or you need to compare optimizers at a common achieved numerical target. | It can require substantially more updates and complete fit/CV time without improving heldout predictions. Check `converged_`, the final residual and the termination reason; requesting a tolerance does not establish that it was reached. |
| `stopping='fixed'` | You deliberately want a finite update budget, for example a controlled trajectory experiment or a training-validated truncated fit. | A fixed update count is not equal work or equal solution quality across policies. Restart proposals and recovery add work; callbacks can stop early, and L-BFGS can terminate before its cap. |
| `stopping='validation'` on `KernGDWD` | You intend to select an intermediate fitted state using accuracy on an explicitly supplied monitoring set. | This is model selection, not a stationarity test. Monitoring adds prediction/checkpoint work and can select a different state. Keep the final test set untouched; create monitoring splits inside each training fold. `KernGDWDCV` and `run_cv` do not silently borrow their scoring folds for this purpose. |

Choose `tol` to express a numerical requirement in the problem's scale, and
`max_iter` as a work allowance. Increasing either the budget or the strictness of
the target does not guarantee a usable result. Kernel automatic recovery can
make a second attempt, so its cap is per attempt. Keep numerical refusals and
capped outcomes visible. The completed stopping comparisons support retaining
objective stopping as the default: stronger stationarity had a real cost and
did not establish a general predictive improvement. See
[optimality guidance and result checks](kernel_dwd.md#choosing-optimality-stopping).

With validation stopping, `check_interval` is useful when you need to control
monitoring frequency; a larger interval reduces observations but can miss an
intermediate best state. `patience` counts non-improving checkpoints, while
`min_delta` sets the required accuracy improvement. Choose them through the
training procedure, not by repeatedly inspecting the final test set. Changing
them changes selection, not just monitoring cost. See
[validation scheduling and restoration](kernel_dwd.md#explicit-validation-and-callbacks).

Use a kernel `callback` for progress recording, inspecting accepted optimizer
states, or an explicitly designed external stop rule. It is not needed for
ordinary prediction or the built-in stopping rules. Snapshot copies and, on
spectral/L-BFGS routes, additional original-kernel checks add work. A callback
also makes the automatic fresh spectral restart ineligible; it is therefore not
an invisible timing instrument. Returned trace decision values need not be
bitwise equal to public joint-affine prediction scores. For performance
comparison, time the intended full workflow and disclose any callback; see the
[callback contract](kernel_dwd.md#explicit-validation-and-callbacks) and
[automatic-restart restrictions](kernel_dwd.md#automatic-numerical-restart).

## Initialization

| Choice | Use it when | Tradeoff and check |
|---|---|---|
| `initialization='zero'` | You need the deterministic default starting point without initialization RNG consumption. | Zero is a useful default, not a proof that every other starting point needs more work. |
| `initialization='random'` | You are measuring sensitivity to the starting state, or training-only validation supports a random-start policy for your chosen update budget and objective. | Starts can change update counts, finite fitted states and predictions; a flat optimum can also admit different states. Use predeclared seeds, include the cost of all starts, and do not select a seed on final test performance. |
| `fit(..., beta_init=..., offset_init=...)` for linear DWD; `alpha_init` for kernel DWD | You have a deliberate starting state for a controlled comparison or an explicit warm start in the current feature/kernel representation. | This starts a fresh fit; it does not restore optimizer, momentum or recovery state. Supply finite values of the correct shape and label orientation. Kernel coefficients must match the current training-row order. |

An explicit coefficient vector overrides coefficient initialization; the
intercept is controlled independently by `offset_init`. Supplying only an
intercept does not suppress requested random coefficients. Integer
`random_state` values reproduce starts; a mutable RNG can advance between fits.
Kernel random initialization normalizes its Gaussian coefficient vector; linear
`GenDWD` uses Gaussian coefficients without that normalization. Neither estimator
automatically warm-starts from its previous fit. The removed `'auto'` alias should
be replaced with `'zero'`. See [initialization details](kernel_dwd.md#backends-and-initialization).

## Backend and linear-system routes

### Kernel DWD

Keep `backend='auto'` unless you have a reason to select a specific route. It
uses Cholesky without an eigenbasis, spectral computation with supplied/cached
eigenpairs, and has a narrowly qualified fresh spectral recovery for internally
constructed RBF problems. Explicit backends do not acquire that automatic retry.

| Explicit choice | Use it when | Tradeoff and check |
|---|---|---|
| `backend='cholesky'` | You need a controlled original-system MM route, or want to isolate its checked behavior from automatic spectral dispatch. | It retains refinement/recovery checks but fixes the backend choice. A failure is not permission to disable checks; retain it when comparing with another supported backend. |
| `backend='spectral'` | You can amortize a checked full eigensystem across compatible parameter candidates, or want to evaluate spectral MM on a difficult system. | Eigensystem setup, validation and storage are part of the cost. Full spectral coordinates do not mean low-rank truncation, and this route is not universally faster. |
| `backend='lbfgs'` | You want a different optimizer for the same full-kernel objective, especially when an MM trajectory reaches its budget without adequate stationarity. | It still needs full spectral preparation. Bounded comparisons found useful quality differences, not universal superiority. Inspect actual termination and residuals; it can stop before a fixed cap, and explicit restart acceleration is incompatible. |

For numerical comparisons, match the represented input, objective, initial state
and achieved target. For predictive comparisons, assess the whole configured
training procedure inside training-only validation instead. A smaller training
residual alone is not evidence of better classification. See
[backends](kernel_dwd.md#backends-and-initialization) and
[fitted diagnostics](kernel_dwd.md#stopping-and-fitted-diagnostics).

### Linear DWD

`GenDWD(implicit_P=False)` and the corresponding `GenDWDCV` setting choose the
explicit checked linear-system route. Use this as a controlled alternative when
the implicit eigensystem/free-intercept correction cannot resolve a problem, or
when comparing complete preparation-plus-fit cost for your design. A bounded
large-penalty case succeeded with the explicit route after implicit preparation
failed; other difficult cases favor implicit. This is a reason to retain the
override, not a guarantee of recovery or a reason to change the default.

Both choices optimize the same corrected linear objective with a free intercept.
They have distinct preparation, arithmetic and recovery behavior. Do not equate
`implicit_P=False` with kernel `linear_action='inverse'`: the current built-in
explicit linear route uses a checked factor rather than requiring a public dense
inverse. Kernel `implicit_P` has been removed. See
[linear classifiers](../README.md#linear-classifiers-and-socp).

### Acceleration and residual ordering

| Choice | Use it when | Tradeoff and check |
|---|---|---|
| `acceleration=None` | You need ordinary MM as a controlled trajectory, or representative training validation/cost measurements favor it over restarted MM. | Ordinary and restarted MM can reach different finite-budget states. Compare at an achieved numerical target when optimizer accuracy is the goal; compare predictive procedures when prediction is the goal. |
| `acceleration='restart'` | You deliberately require restarted MM on a supported backend and want incompatible configurations to be rejected rather than resolved by `auto`. | Automatic acceleration already selects this on eligible MM routes. Restart can save many updates, but extra proposals/restarts and occasional slower cases remain; it is incompatible with L-BFGS. |
| `residual_check_order='adaptive'` | You deliberately request the accurate-first policy on an eligible Cholesky route, including an explicit Cholesky fit with supplied eigenpairs, or are investigating repeated unproductive refinement. | The adaptive trigger can avoid corrections after accurate assessment shows the untouched state was adequate. A failed probe adds work; ordinary residual acceptance alone does not demonstrate a benefit. |
| `residual_check_order='refinement_first'` | You want the correction-first route for a controlled comparison, or measurements show adaptive probes add unproductive work. | The acceptance criteria remain the same, but ordering can change accepted floating-point states and cost. Spectral/L-BFGS routes do not accept explicit adaptive ordering. |

Retain both automatic defaults for general use. Read the requested, initial and
effective policy diagnostics, particularly after recovery or eigenbasis reuse.
Do not infer a speed advantage solely from iteration counts or a microbenchmark
of one check. See [restarted MM](kernel_dwd.md#restarted-mm) and
[residual assessment and recovery](kernel_dwd.md#numerical-validation-and-recovery).

## Arithmetic and recovery

| Opt-in | Use it when | Tradeoff and check |
|---|---|---|
| `rbf_computation='direct'` | Sensitive feature geometry makes rounded RBF entries important, and you want every supported pair evaluated through direct coordinate differences rather than the standard policy's selective repair. | Standard already repairs identified cancellation cases. Direct has additional precision benefits on some unflagged pairs, but is not uniformly more accurate and can be much slower, especially for CSR. Assess kernel entries and downstream scores/selection as well as complete construction/fit/prediction cost. |
| `affine_computation='standard'` | Representative measurements favor separately rounded scoring and checks establish adequate margins/score accuracy for your application. It is also useful as a controlled arithmetic comparison. | Joint remains the default because cancellation with the intercept can change a sign. Standard is unsuitable as a shortcut for sensitive near-zero decisions; validate score and label effects, including CV or validation-state selection. A cost saving is not proof of acceptable accuracy. |
| `exact_recovery='extended'` | A small difficult MM problem reaches standard exact-recovery rank/bit capacity and you are willing to spend more arithmetic work attempting a checked recovery. | It raises only specified capacities. It cannot repair an indefinite stored kernel, lift the entry/operation limits, guarantee convergence or bound total wall time. Healthy fits do not gain speed, and L-BFGS does not use this recovery. |
| `linear_action='inverse'` | Many updates may amortize construction of a full inverse action and representative end-to-end measurements support using it. It can also serve a controlled factor-versus-inverse comparison. | It adds quadratic workspace and can be slower under actual stopping. Retain `factor` otherwise; smaller iteration counts or cheaper individual actions do not establish a faster fit. Backend restrictions, numerical checks and recovery still apply. |

These settings do not change the intended loss or unpenalized intercept. They can
change rounded states, predictions and selected models, so their identity belongs
in a reported experiment. The direct-RBF and affine policies address different
calculations: accurate summation cannot repair inaccurate kernel entries. Joint
scoring is also not a guarantee of correct rounding for every accepted score.
See [direct RBF](kernel_dwd.md#optional-direct-rbf-computation),
[joint/standard scoring](kernel_dwd.md#joint-affine-scoring),
[extended recovery](kernel_dwd.md#optional-extended-exact-recovery), and
[inverse-action limits and memory](kernel_dwd.md#explicit-full-inverse-action).

## Prediction, preparation and searches

- **Prediction batches:** a positive `prediction_batch_size`, or a per-call
  override, bounds query rows in temporary kernel blocks. Use it to control
  allocation granularity, then measure the full prediction call and check scores
  and labels. Partitioning can change low-order rounding and add overhead. Do not
  recommend a slower setting solely for memory savings; smaller buffers do not
  establish a performance improvement. Batching does not shrink the training
  Gram matrix or caller-owned precomputed inputs. Explicit `None` disables it.
  See [batching](kernel_dwd.md#kernels-batching-and-caches).
- **Prepared kernels/eigensystems:** `cv_init(X)`, supplied kernel `K`/`K_eig`,
  or linear `P0_eig` are useful for repeated compatible fits where preparation
  is significant. Include setup and validation in the workload cost, and reuse
  only the exact geometry, row/feature order and preprocessing. Do not slice a
  full-data eigenbasis into fold eigenpairs or use it to leak validation data.
  Cloning does not transfer hidden caches; kernel `cv_init` invalidates the
  previous predictor. See [preparation and consistency checks](kernel_dwd.md#kernels-batching-and-caches).
- **Shared feature folds:** choose `fit_shared_fold_search` for an eligible serial
  `KernMD` grid where repeated wide-feature copying is a measured cost. Within
  the explicit helper, exact contiguous row selections automatically share a
  private immutable snapshot; other selections keep indexed copies. Ordinary
  or moderate-sized searches and short grids may favor ordinary search.
  The helper retains cached subsets through the final refit and clears its
  private registry when it exits. This can retain more feature storage;
  `shared_fold_info` reports logical payload, not peak memory. Pipelines and
  unsupported configurations use the
  ordinary route. Preserve statistically chosen folds and keep learned
  preprocessing within them. See [supported use and measured tradeoffs](shared_fold_search.md#when-to-use-it).
- **Resumable CV:** choose the example workflow for a fixed kernel-DWD grid whose
  completed fold/candidate work needs interruption recovery. It is an example
  interface, not an estimator setting or an adaptive search. Checkpoint writes,
  hashing and compatibility checks add work. Changed data, policies or runtime
  can require a new run; it is not a general way to reuse old results under new
  settings. Worker/native-thread limits control concurrency, not numerical
  acceptance or guaranteed speed. See [resumption](resumable_cv.md) and
  [resource planning](resumable_cv.md#allocating-cv-workers-and-native-threads).
- **Final refit in that example:** use `refit_best=False` (CLI `--cv-only`)
  when the current task is evaluating the grid and no deployable full-training
  model is needed yet. It skips final fitting, not CV or winner selection.
  With `refit_best=True`, the selected model is fitted afresh even if all CV
  results were resumed. Raise `final_native_threads` above its default of 1
  only when a measured full-data scaling/refit comparison supports useful
  native-library scaling within the available resources. It does not change
  CV worker threads and can change floating-point reductions. Use
  `cv_native_threads` and `jobs` to allocate the separate CV phase, avoiding
  nested oversubscription; more threads do not guarantee a faster fit.

## Optional conic models

Choose `dwd.socp_dwd.DWD` when the original constrained linear SOCP formulation
is the intended model. Choose `dwd.svm.SVM` when its mean-hinge loss with an L1
coefficient penalty is intended. These are distinct mathematical formulations,
not interchangeable numerical backends for generalized kernel DWD or sklearn's
L2-regularized SVC. They require the optional CVXPY dependency. Their `C` parameter
must not be substituted blindly for `lambd` or SVC's `C`.

Use `solver_kws` when you need an explicit conic solver, solver diagnostics, or a
documented solver-specific accuracy/work setting. Both classes forward it to
`cvxpy.Problem.solve`. First inspect `cvxpy.installed_solvers()` and choose an
installed solver supporting the problem's cones. Option names and defaults vary
by solver; follow the [CVXPY solver documentation](https://www.cvxpy.org/tutorial/solvers/index.html)
rather than copying one solver's settings to another. Tighter settings can add
cost without meeting the requested accuracy. Inspect `solver_status_` and the
problem status; `optimal_inaccurate` does not establish the desired accuracy,
and solver failures can raise exceptions. See
[CVXPY status handling](https://www.cvxpy.org/tutorial/intro/index.html).

Nested builtin solver-option containers are copied before a conic solve so
solver adapters cannot edit the caller's dictionaries, lists or tuples in
place. Arrays, callbacks, custom solver objects and container subclasses retain
their identity; mutation inside those objects is not isolated. For saving a
fitted conic model, diagnostic record types, warm-start behavior, solver-limit
refusals and measured fit costs, see
[conic persistence](fitting_and_conic_persistence.md#conic-problems-and-persistence).

For SOCP `DWD` only, `C='auto'` selects the documented distance-based heuristic
`100 / d**2`, where `d` is the median **between-class** training distance. Use it
when that scale-based starting rule is appropriate and an explicitly tuned `C`
has not been selected. It is not a validation search or an optimality guarantee.
It depends on feature scaling, requires a positive finite median, and constructs
the cross-class distance array, whose size is the product of the two class
counts. On larger data, account for that work before choosing it. Compare the
heuristic with explicit `C` values using training-only validation; compute it
inside each training fold. The default remains `C=1.0`, and the SVM class does
not accept this `'auto'` policy. See [conic examples](../README.md#linear-classifiers-and-socp).

## Diagnostics and native execution

Use `literal_objective_diagnostics` when you need a bounded literal check
of a small stored kernel's represented objective, gradients and dual bounds.
Its at-most-16-row and arithmetic limits make it suitable for numerical
investigation, not a large-data fitting policy. Preserve `DiagnosticRefusal` as
an unsupported diagnostic outcome rather than replacing it with a success.
It does not refit or change the model. See
[the literal diagnostic contract](kernel_dwd.md#bounded-literal-operator-diagnostics).

Use `residual_profile()` to investigate which guarded residual routes run and
why they decline. It adds observation overhead, excludes some work and does not
measure a complete fit. Use ordinary wall-clock workflow measurements for the
performance decision. See [profiling](compiled_residual.md#optional-residual-profiling).

The exact feature-expansion helper is also automatic on eligible builds; it
is not an estimator option or a reason to alter numerical thresholds. It can
reduce wide and repeated-fit reporting cost, while first use and missing-binary
fallback can add work. Preserve fallback on unsupported interpreter builds and
measure the complete intended workflow. Active tracing/profiling or replaced
private callbacks can deliberately decline native execution. See
[feature-expansion eligibility](compiled_residual.md#exact-feature-expansion-tiles).

Compiled residual/score dispatch and fitted `prediction_precision_` are automatic
implementation decisions, not public estimator switches. Do not change learned
attributes or private hooks to force them. Choose a portable build with
`DWD_BUILD_ACCEL=0` when compiling the optional helper is unavailable or a
portable distribution is required; choose an accelerated build for supported
workloads that benefit from guarded native arithmetic. This is a build-time
choice, not a promise of whole-fit speedup or a way to relax numerical checks.
See [build modes and runtime eligibility](compiled_residual.md#building-and-installing).

## Assess a choice before adopting it

Change one relevant policy at a time for attribution, using identical input,
preprocessing, objective and appropriate initialization. Preserve failures,
budget exhaustion and observed slow cases. Include preparation, scoring,
checkpointing and final refit when those belong to the intended workflow.
Inspect the actual backend, returned residuals and convergence separately from
predictive validation. A faster fixed-budget run can return a different model;
a tighter residual can leave test accuracy unchanged or make it worse.

Use ordinary and relevant difficult cases, not only a small kernel-operation
benchmark. Evaluate score signs and selected states where arithmetic matters,
and distinguish scientific correctness from runtime observations. A numerical
correctness or robustness benefit can justify additional work, but a smaller
memory footprint alone does not justify recommending a measured slowdown.
