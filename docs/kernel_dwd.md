# Kernel DWD: API and numerical controls

This guide describes the kernel estimator, its objective, parameter conventions,
stopping rules, and numerical diagnostics. For release-specific validation, see
[VALIDATION.md](../VALIDATION.md). Details of the optional arithmetic accelerator
are in [compiled residual arithmetic](compiled_residual.md).
For practical reasons to select each nondefault policy and how to assess its
tradeoffs, see [Choosing DWD options](choosing_options.md).

`dwd.gen_kern_dwd.KernGDWD` fits one binary classifier. It does not search for
parameters or create hidden validation splits. `KernGDWDCV` provides
cross-validation; sklearn GridSearchCV can also wrap the estimator or a complete
external training procedure.

## Objective and parameter names

The fitted function is `f(x) = sum_i alpha_i K(x_i, x) + b`, with unregularized
intercept `b`. The objective is

```text
mean_i V_q(y_i * f(x_i)) + lambd * alpha.T @ K @ alpha
```

Labels are mapped to -1/+1 using their sorted training classes. Positive decision
values predict `classes_[1]`; nonpositive values predict `classes_[0]`. Predictions
retain the original labels.

`lambd` is the package regularization coefficient, `q` is the generalized-loss
exponent, and `kernel_kws` contains kernel parameters such as RBF gamma. These
names have not changed. The choice `q=1` gives DWD. Chang Cheng's dissertation
writes the penalty as `(lambda_dissertation / 2) * norm_squared`. For the same
objective, set `lambd = lambda_dissertation / 2`. When reproducing a historical
fit, check the convention in its own script before converting a stored value.
[Dissertation, version 1](https://doi.org/10.25394/PGS.34548831.v1).

The optional SOCP classifier uses a different constrained linear formulation and
parameter `C`. Passing a Gram matrix as ordinary linear features changes the
regularization geometry; it is not an equivalent kernel SOCP conversion.

### The free intercept need not be unique

For both generalized linear `GenDWD` and kernel `KernGDWD`, the loss is
`V_q(u) = 1-u` when `u <= q/(q+1)`. With equally many positive and negative
training labels, a range of intercepts can leave every margin in this linear
part of the loss. Over that range, the mean loss is constant in the intercept;
the norm penalty does not involve it. When the coefficients also minimize the
joint objective, this gives several globally optimal intercepts. Balanced
labels alone do not guarantee such a range, and a flat intercept interval at
arbitrary fixed coefficients does not establish joint optimality.

For an exact linear example, take `X = [[-1/2], [1/2]]`, `y = [-1, 1]`,
`q = 1` and `lambd = 1`. The unique slope is `beta = 1/4`, while every
`b` in `[-3/8, 3/8]` attains the global minimum `15/16`. Indeed,
`V_1(u) >= 1-u` gives the lower bound
`objective >= 15/16 + (beta - 1/4)**2`, with equality throughout that interval.
At a query `x = 0`, the optimal intercepts `b = -1/4` and `b = 1/4` therefore
predict different classes. The stated nonpositive-score rule still applies at
`b = 0`. This effect occurs in exact arithmetic, without a singular design,
an ill-conditioned kernel or a change of numerical library.

For positive `lambd`, the regularized slope or RKHS function is unique at a
global optimum; the unpenalized intercept need not be. With a singular kernel,
different coefficient representations of the same RKHS function are a separate
issue. These statements concern the generalized mean-loss objective above,
not an asserted equivalence with the optional constrained SOCP formulation.

Consequently, equal objective values alone do not establish identical decision
scores, labels or cross-validation scores across solver routes or starting
states. Diagnose observed differences using the objective, stationarity and
achieved stopping target as well as predictions; do not assume every discrepancy
is a flat optimum. Keep input, initialization and solver settings fixed for
reproducibility, and retain the existing score, label and numerical checks.
The package does not choose a canonical intercept from an optimal interval or
add an intercept penalty. The bounded MM equation-residual correction described
in [intercept refinement](intercept_midpoint_refinement.md) is a different
operation, not a tie-breaker among global optima.

## Backends and initialization

The public estimator uses the corrected optimized solver. Initialization now
defaults explicitly to `'zero'`, preserving the previous default's starting
state; the other default policies are retained. The repaired coefficient-MM
reference route remains private validation machinery; `implementation` is no
longer a public parameter.
The supported backends solve the same full-kernel DWD objective with a free
intercept, while their finite-iteration behavior and numerical costs can differ.

| Parameter | Behavior |
|---|---|
| `backend='auto'` | Cholesky unless eigenpairs are supplied/cached; eligible internal RBF fits may restart once through the spectral backend after constrained/MM recovery is exhausted. |
| `backend='cholesky'` | Solve the original constrained shifted-kernel equations with residual checks, bounded refinement, and centered factor recovery. Unknown/precomputed kernels still require PSD validation. |
| `backend='spectral'` | Use the full checked eigensystem and spectral coordinates; the eigensystem can be reused across lambda/q values. |
| `backend='lbfgs'` | Optimize the same generalized objective in full spectral RKHS coordinates with a free intercept. It requires spectral preparation. |
| `acceleration='auto'` | Default: restarted MM, including checked spectral solves; ordinary updates for L-BFGS. Resolution does not depend on stopping policy or data. |
| `acceleration=None` | Explicit ordinary MM. A missing requested setting uses the current `'auto'` default. |
| `acceleration='restart'` | Extrapolate decision values and restart momentum using the original objective. Available with Cholesky or a checked full spectral solve; incompatible with L-BFGS. |
| `initialization='zero'` | Default: deterministic zero coefficients and default zero intercept. |
| `initialization='random'` | Random unit coefficients controlled by `random_state`. |

The former `initialization='auto'` alias has been removed. Replace that value
with `'zero'` in explicit constructor settings, `set_params` requests and CV
grids; omitting the parameter now selects `'zero'` directly. This spelling change
applies to both linear and kernel generalized DWD estimators and their CV
wrappers. The initialization control remains available with `'zero'` and
`'random'`; other controls such as `backend`, `acceleration` and
`residual_check_order` keep their supported `'auto'` values. The objective,
free intercept and stopping defaults are unchanged.

The corrected kernel backends implement generalized q, not a different loss
for each backend. See the release's [validation record](../VALIDATION.md) for the
tested q values and numerical cases. No backend is universally fastest: caching,
kernel size, conditioning, and number of parameter candidates matter.

`fit(..., alpha_init=..., offset_init=...)` overrides the corresponding initial
values. Linear `GenDWD.fit` similarly accepts `beta_init` and `offset_init`.
An explicit alpha vector must match the current training rows. Fits do
not warm start from previously fitted coefficients. `initialization='zero'`
ignores `random_state`; external subsampling or CV still needs its own seed.

Use an integer `random_state` when reproducible random initialization is needed.
Comparing finite-update policies also requires the same kernel, regularization,
initial values and stopping settings. Reusing a mutable random generator can
consume its state; a fixed integer seed makes the requested initialization clear.

<a id="migration-from-legacy-fitting"></a>

### Removed public solver controls

All new linear and kernel fitting, CV preparation, low-level solves and kernel
steps use corrected update algebra with a free intercept. `solver_mode` and
kernel `implementation` are no longer public parameters. Remove them from
constructor/solver calls, `set_params` requests and parameter grids, even when
their values were `'schur'` or `'optimized'`.
Kernel-only `implicit_P` has also been removed; it did not select a supported
alternative. Removed controls are not silently ignored. Built-in DWD CV
rejects a grid containing a removed control before preparing folds or fitting
another candidate.

The corrected reference route is retained privately for tests, not as a public
solver or a compatibility route for old configurations. Linear
`GenDWD(implicit_P=False)` and `GenDWDCV(implicit_P=False)` remain supported,
as do their default implicit routes. Kernel inverse action remains an explicit
`linear_action` option, unrelated to the removed kernel `implicit_P` control.

Constructor arguments after `random_state` are keyword-only for `GenDWD`,
`GenDWDCV` and `KernGDWDCV`. Constructor arguments after `kernel_kws` are
keyword-only for `KernGDWD`. `solve_gen_dwd` accepts positional arguments only
through `q`; `solve_gen_kern_dwd` retains its first six positional arguments
through `offset_init`. Their subsequent arguments are keyword-only. Unsupported
`sample_weight` arguments have been removed, including the former `None`
placeholder; see [fitting argument migration](fitting_and_conic_persistence.md#fitting-arguments-and-metadata).
`get_step_implicit_P` no longer accepts a final mode argument. These boundaries
make obsolete positional calls fail rather than silently bind to another
option. Use named arguments when updating those calls. The ignored
`reference` argument to `get_K_eig` has also been removed; the helper computes
the same checked eigensystem without an implementation selector.

#### Migrating former reference requests

Do not interpret removal of `implementation='reference'` as a behavior-preserving
switch to the existing defaults. Reference auto selected random normalized
Gaussian coefficients, spectral solves, ordinary MM and refinement-first residual
checks. The current public defaults retain the former optimized choices: zero
initialization, normally Cholesky, restart acceleration and adaptive residual checks on eligible
Cholesky fits. Supplied/cached eigenpairs or bounded recovery can select spectral.
These choices can change the RNG consumption, path, callbacks, stopping point,
scores and labels at a finite iteration budget.

To request the former reference auto policies through the supported solver, use
explicit settings such as:

```python
model = KernGDWD(
    backend='spectral', initialization='random', random_state=7,
    acceleration=None, residual_check_order='refinement_first',
)
```

This retains the policy choices, not the former internal coefficient-MM
implementation or a promise of bitwise equality. Keep the same data, kernel,
regularization, stopping settings and integer seed when making comparisons.
The independent equation tests and private reference machinery remain available
to maintainers; private paths are not a supported public workaround.

The follow-up investigation used 48 quality fits and 117 cost-panel fits per
host on Windows and Linux. At the quality target of `1e-8`, reference and
optimized matched policies converged on the same eight of twelve geometry/q
cases; both reached the iteration cap on the other four. Reference auto also
met eight targets, while optimized auto met all twelve. The matched converged
results agreed to floating-point rounding. This separates an automatic-policy
advantage in those hard cases from a claim that one update formula is always
more accurate. Independent equation/original-kernel checks remain necessary:
the private reference shares preparation machinery and is not an independent
oracle by itself.

Every fit in the separate repeated cost panel met its independent `1e-6`
target. Optimized spectral with matched reference policies had lower fit cost
in every recorded paired block on the three fixed workloads. Optimized auto
had lower median cost than reference auto, but one Windows block was slower;
optimized spectral also had lower median cost than optimized auto. The fits
were short and all timing spikes were retained. These finite observations
support retaining useful policy controls, not a universal speed guarantee or
a new automatic backend rule.

The earlier default-objective-stopping panel is still relevant to migration:
its eighteen models per host remained nonconverged, one optimized-auto
objective was about `1.3236e-5` higher, and a separate RBF query panel changed
two of 256 predicted labels under optimized auto on both hosts. The new
optimality-stopping study does not retroactively qualify those results or
promise unchanged finite-budget predictions. No accuracy claim follows from
query-label changes without true labels, and neither study attributes these
implementation differences to an AOCL or MKL version upgrade.

#### Paired releases, caches and saved objects

Use DWD 2.0.0 and Ensemble DWD 0.4.0 together. DWD 2.0.0 rejects the
`implementation` keyword forwarded by earlier Ensemble versions. Upgrade both
packages before refitting an ensemble or starting a new parameter search.

Preparation needs the current geometry marker; stale or absent markers cause
rebuilding. Persistent Ensemble cache identity includes package source content,
so changed sources create a cache miss/new entry while preserving old entry
bytes. Checkpoint resume rejects changed source/runtime identity before
new fitting or owner-state rewriting. The example checks installed wheel metadata
as part of that identity, so a same-version rebuild with different metadata can
also require a fresh directory. Start a new run directory for the upgrade;
do not rebase or overwrite the old provenance to force a resume.

Old serialized estimators are outside an arbitrary cross-version compatibility
guarantee. Twelve genuine legacy fixtures retained identical baseline/candidate
predictions within each checked host and produced unfitted current clones,
without source-dependent special cases. That observation does not certify old
fitting algebra or every old object. Tiny cross-host score differences remained,
shared by both source versions; all observed labels agreed. Clone omits the
removed requests; a later requested fit is new corrected computation, not
retroactive repair of the saved model. Keep original artifacts and environments
when exact historical reproduction is required.

To replace a model fitted by an incorrect historical algorithm, construct a
current estimator and refit from the original training data, labels and
settings. Loading an old object does not repair its coefficients, and this
API cleanup does not promise cloning or refitting arbitrary old serialized
states. Historical reproduction belongs to the matching archived release and
environment; it does not select a current public solver.

## Automatic numerical restart

An optimized fit with `backend='auto'` and
`callback=None` can make one fresh spectral attempt when a constrained MM update
and its existing bounded recovery both fail. The trigger is specific: the
original/centered/anchor constrained solve must be exhausted, followed by failure
of its certified MM-function recovery. An unrelated objective-descent rejection,
callback exception, memory error, or cancellation does not trigger this restart.

This applies only to named RBF kernels built internally by the package's unchanged
construction methods, with valid known-PSD parameters. A reused CV kernel also
requires the recorded construction provenance and unchanged preparation methods.
Caller-supplied `K` or eigenpairs, precomputed and callable kernels, overridden
construction methods and explicit backend choices
do not gain this restart. Explicit `acceleration='restart'` is supported: its
ordinary-step retry remains first, and only exhausted constrained/MM recovery
can trigger the fresh spectral attempt. A successful ordinary fit
does not compute an additional eigendecomposition.

The failed attempt's temporary solver arrays and factors are released before
spectral preparation.
The new attempt reuses the exact original kernel, labels, realized initialization,
regularization, generalized-loss exponent, and stopping/validation settings. It
does not draw another initialization or continue from rejected coefficients.
Both attempts optimize the same kernel DWD objective with an unregularized
intercept; no ridge is added and no positive eigendirections are discarded.

The existing spectral acceptance checks remain in force: a checked eigenbasis,
original-kernel scores and RKHS objective for the returned model, and additional
checks before any validation checkpoint is observed. These differ from the
Cholesky route's constrained coefficient/gauge residual checks. A successful
spectral restart does not certify a rejected Cholesky coefficient state. If the
spectral attempt also fails its checks, the error propagates; there is no second
restart or unchecked fitted model.

`max_iter` is the accepted-update cap **per attempt**. The default is at most 100
updates per attempt, with earlier objective-change stopping. After a successful
restart, `n_iter_`, objective/validation histories, and `returned_iteration_`
describe only the successful spectral attempt. Previously completed updates are
reported as discarded work in `diagnostics_['spectral_restart']`; they are not
added to the returned trajectory. Fixed stopping therefore specifies the successful
attempt's update budget, not total computational work across both attempts.

The restart receipt records `discarded_completed_updates`,
`failed_attempted_iteration`, `successful_updates`, `total_completed_updates`, and
`total_attempted_updates`. The last includes the failed update attempt. It also
records failed-attempt details and `successful_attempt_timing`. Top-level
`setup_seconds`, `optimization_seconds`, and `total_seconds` include both attempts.
`backend_` is `'spectral'`, while the requested backend remains `'auto'` in the
diagnostics. This numerical restart is separate from `acceleration='restart'`.
The failed-attempt receipt also preserves acceleration counters, including in
ensemble lean storage and reusable base-cache summaries.

## Bounded literal-operator diagnostics

`dwd.literal_diagnostics.literal_objective_diagnostics` is an explicit diagnostic
for small stored kernels (at most 16 rows). It treats finite binary64 inputs as
exact represented numbers and encloses the objective and coefficient/intercept
gradients for any positive finite generalized-loss exponent within its arithmetic
budgets. For score map `K @ alpha + offset`, the coefficient gradient uses `K.T`
and the penalty uses the symmetric part of `K`; the stored score map is never
silently replaced. Exact PSD and range checks support a valid dual lower bound.
Indefinite penalties and exceeded precision, operation or bit budgets raise
`DiagnosticRefusal`; this routine is not an unbounded large-kernel validator.

This diagnostic never changes estimator state, solver stopping or numerical
acceptance. Its operation counter counts instrumented rational operations,
not total CPU work. Decimal transcendental enclosures assume correctly rounded
`ln` and `exp`; reported intervals are rounded outward to binary64.

## Restarted MM

```python
accelerated = KernGDWD(
    kernel='rbf', kernel_kws={'gamma': 1.0}, lambd=.02,
    acceleration='restart', stopping='optimality', tol=1e-6, max_iter=1000,
).fit(X, y)
```

The acceleration acts on total decision values, including the free intercept.
Each proximal update solves the same original constrained MM system. It neither
changes the kernel nor truncates eigendirections. The momentum schedule begins
with two ordinary MM updates. An extrapolated proposal that increases the
objective triggers one ordinary-step retry; a failed numerical proposal at
nonzero momentum can also trigger that bounded retry. A failed ordinary step
can invoke the narrowly gated fresh spectral attempt described above when its
constrained/MM recovery is exhausted; other unresolved failures raise. The
ordinary-step recovery receives the same full extrapolated right-hand
side, with the corresponding zero prior-offset convention of the proximal step.
Rejected proposals are not passed to callbacks or validation rules.

`n_iter_` counts accepted MM updates. Extra proposals and restarts consume work
without increasing that count; `diagnostics_` records
`acceleration_proposals`, `acceleration_accepted_updates`,
`acceleration_restarts`, and `acceleration_numerical_restarts`. The latter is a
subset of all restarts. Fixed 100 updates therefore does not imply identical
work across ordinary and accelerated MM. A validation-restored model can come
from an earlier accepted iteration, as with ordinary MM.

The automatic default selects restart only on compatible solver routes. Explicit
`None` selects ordinary MM, and explicit `'restart'` rejects incompatible routes.
The constructor parameter remains the request; fitted `requested_acceleration_`
and `acceleration_`, also recorded in `diagnostics_`, distinguish request from
effective policy. Momentum retains one extra n-element float64 score vector
(8n bytes) between updates, plus temporary O(n) work vectors. This is not a
bound on total solver or recovery memory.

The default absolute
successive-objective stopping rule depends on the optimization trajectory and
does not certify stationarity. A faster trajectory can therefore stop at a
different optimization quality under the same objective-change threshold;
fixed update budgets and validation checkpoints also observe different states.
Use `stopping='optimality'` when the aim is a common numerical stationarity
target, and retain any failed or capped result as such. Parameter tuning and
explicit stopping validation belong inside the training fold. The package
does not infer a different optimizer from a stopping or callback setting.

## Stopping and fitted diagnostics

| `stopping` | Rule |
|---|---|
| `'objective'` | Default: stop when absolute successive-objective difference is below `obj_tol=1e-5`, subject to `max_iter=100` per attempt. |
| `'fixed'` | Execute the requested MM update budget; callbacks can stop earlier. L-BFGS can terminate earlier if its optimizer cannot continue. |
| `'optimality'` | Check `max(RKHS gradient L2 norm, absolute intercept gradient) <= tol` against `tol=1e-6`, subject to the budget. |
| `'validation'` | Monitor explicitly supplied data, then return the best checked model as described below. |

`tol` is a numerical scale, not an accuracy target or a universal recommendation.
L-BFGS may stop at an exactly stationary point or when its internal line search
cannot continue, even with `stopping='fixed'`. Its actual accepted updates and
native termination reason are reported; the package does not invent extra
iterations to fill the budget. Exactly fixed update counts apply to MM.

### Choosing optimality stopping

The default remains `stopping='objective'`. Choose the explicit
`stopping='optimality'` option when:

- Your application requires a returned model to meet a specified numerical
  stationarity tolerance, rather than only a small successive-objective change.
- You are comparing solver backends or acceleration settings at a common
  achieved stationarity target. Keep the data, objective and initialization
  comparable, and check that each returned model actually met that target.
- An objective-stopped model's reported residual is too large for the precision
  required by your downstream analysis, so you are willing to spend more work
  pursuing a better optimized solution.

For ordinary predictive tuning, retain the objective default when its model
quality and diagnostics are adequate. Optimality is not an accuracy upgrade:
smaller training residuals can leave predictions unchanged or change them in
either direction. Assess predictive choices using training-fold validation,
without tuning the stopping policy on the heldout test set.

Optimality can substantially increase complete fit and CV time by requiring
more updates; kernel residual checks also add work. Linear built-in
updates already reuse eligible gradient calculations, so the cost is not simply
a duplicated gradient every iteration. Measure representative complete workflows
when this cost matters. Do not infer a negligible cost from a single cheap check.

Choose `tol` as a numerical requirement in the problem's scale, and give
`max_iter` enough room to pursue it. The default budget of 100 can still be
exhausted with optimality selected. A larger budget is not a convergence
guarantee; callbacks, backend termination and numerical failures can also prevent
the requested target. Kernel `max_iter` is per attempt, so an eligible automatic
spectral restart can add work beyond that one-attempt budget. Inspect the actual
returned residual and convergence flag rather than assuming the request succeeded.

For example, when stationarity is a requirement:

```python
from dwd.gen_kern_dwd import KernGDWD

model = KernGDWD(
    stopping='optimality', tol=1e-6, max_iter=1000,
).fit(X_train, y_train)

print(model.termination_reason_, model.n_iter_, model.stationarity_residual_)
if not model.converged_:
    raise RuntimeError('The fitted model did not meet the stationarity target.')
```

Here `1000` is an illustrative explicit budget, not a recommended universal
default. `GenDWD` accepts the same stopping, tolerance and budget controls and
exposes the same diagnostics used in this example. Its residual is the maximum
of the feature-gradient L2 norm and absolute unpenalized-intercept gradient;
the kernel criterion uses the RKHS/intercept residual. These are numerical
checks, not a general certificate of distance to the optimum or predictive error.

`GenDWDCV` and `KernGDWDCV` apply their stopping controls to the candidate fits
and final refit. Inspect `search.best_estimator_.converged_` and
`search.best_estimator_.stationarity_residual_` after `search.fit(...)`.
Those fields describe the selected final refit, not all candidate/fold fits;
the CV score table does not certify their convergence. If every fold must meet
the target, retain and check those fits explicitly in your validation procedure.

### Reading convergence diagnostics

All corrected fits check the returned model's stationarity, whatever their
stopping rule. A fixed cap is not by itself a convergence claim. Likewise, an
objective stop may or may not satisfy the separate numerical check.

- `n_iter_`: accepted updates in the successful attempt. `returned_iteration_`: iteration of the
  returned coefficients, which can be earlier after validation restoration.
- `termination_reason_`: why execution stopped, such as `objective_tolerance`,
  `optimality_tolerance`, `validation_patience`, `callback_stop`, `max_iter`, or
  an L-BFGS stationary/stagnation status.
- `objective_history_` / `obj_vals_`: initial objective followed by executed
  states of the successful attempt. `final_objective_`: objective evaluated at the returned model.
- `objective_tolerance_met_`: whether the final two executed objective values
  met `obj_tol`; this can describe a later state than the restored model.
- `rkhs_gradient_norm_` / `stationarity_residual_`: the maximum RKHS/intercept
  residual above. `stationarity_checked_` records whether it was checked.
  `converged_` / `optimality_met_` reports that residual against `tol`.
- `gradient_inf_norm_`: retained coefficient-coordinate/intercept gradient
  diagnostic. Its scale depends on the kernel coordinates; it is not substituted
  for the RKHS criterion.
- `dual_gap_` and `dual_equality_residual_`: corrected-fit dual diagnostics.
  The slopes use an exact dyadic class-mass balance, and the dual value uses
  an outward quadratic bound for a symmetric PSD kernel. The primal objective
  and gap remain numerical estimates. `diagnostics_['dual_certificate']`
  records the method and any explicit zero-slope fallback when a nonzero bound
  is unavailable or weaker. An asymmetric stored kernel also uses that fallback;
  the symmetric-kernel dual formula is not claimed for that score map.
  If the final certificate calculation establishes a negative dual quadratic
  or a negative constant kernel, its PSD assumption is contradicted. The fit
  retains its completed numerical result, while `dual_gap_` and
  `diagnostics_['dual_certificate']['dual_objective_lower_bound']` are `None`.
  The certificate records `dual_objective_is_lower_bound=False` and the exact
  `fallback_reason`: `certified_negative_dual_quadratic` or
  `negative_constant_kernel`. This is unavailable-bound reporting, distinct
  from the valid zero-slope fallback above. It supplies no certificate of PSD
  or global optimality. Existing handling of other input, reconstruction,
  norm, gradient and arithmetic failures is unchanged; numerical acceptance
  and stopping checks are unchanged.
  These diagnostic choices do not change the fitted state or certify convergence.
  Tiny negative raw gaps from rounding remain visible. They are not the stopping
  tolerance. `backend_` is the effective backend. `diagnostics_` records
  `requested_backend`, `initial_backend`, `effective_backend`,
  `attempted_backends`, `auto_fallback`, `fallback_reason`, and `implementation`.
  `linear_system_diagnostics` records the solve representation, accepted equation
  and constraint residuals, estimated RKHS solve error, refinement steps, and
  recovery counts. Cholesky factorization attempts include `rcond` and time;
  `rcond` alone is not an acceptance rule. `eigendecomposition_validation` records
  attempted native drivers and checks for full eigenpairs. Failed attempts remain
  visible; no provider self-probe or isolated-child recovery is performed.
  Validation is part of spectral setup.
  `eigenvalue_validation`
  identifies the weaker invariants available in an eigenvalues-only PSD check.
  Separate `cholesky_setup_seconds`, `eigenvalue_check_seconds`, and
  `spectral_setup_seconds` report preparation. Full fit timing must also include
  any recovery performed during iteration. These nested fields are present only
  when their corresponding path executes.
- Optimized spectral-coordinate and L-BFGS checkpoints exposed to a callback or
  validation rule are checked against the original kernel scores and RKHS
  objective first. `observed_checkpoint_checks`, `max_observed_score_discrepancy`
  and `max_observed_objective_discrepancy` report these checks. An inconsistent
  checkpoint raises before exposure; it is not silently rewritten. These checks
  add matrix products only when a checkpoint is observed. Unobserved iterations
  retain their fast coordinate path and the separate checks on the returned model.
- `prediction_precision_`: `'adaptive'` or `'compensated'` for corrected
  kernel models; already-fitted legacy models and older serialized models
  without the field retain ordinary evaluation. This is learned state, not a constructor option. It is
  restored with the best validation checkpoint and preserved by serialization;
  cloning does not transfer it. `diagnostics_` can also record
  `compensated_score_reconstructions`,
  `max_adaptive_score_reconstruction_error_before_retry`, and
  `objective_evaluation_retries` when these paths execute.
- `affine_computation_`: fitted `'standard'` or `'joint'` scoring policy. This
  is separate from `prediction_precision_`, which chooses adaptive or fully
  compensated evaluation. Prediction uses the fitted affine policy even if
  the constructor parameter is subsequently changed. Older serialized models
  without this field retain standard scoring; refit to enable joint scoring.
- `diagnostics_['mm_function_recovery']`: lazy exact-certificate preparation,
  accepted recovery actions and their checks. `accepted_actions` counts helper
  actions, including actions inside a subsequently rejected acceleration
  proposal; it is not `n_iter_`. The separate
  `returned_state_uses_certified_columns` field identifies the returned state,
  which can precede a later recovery after validation restoration. When true,
  `coefficient_representation` is `'certified_sparse_kernel_columns'`. An
  attempted recovery or nonzero action count alone does not establish that
  representation for the returned coefficients. `backend_` still identifies
  the numerical backend initially selected for the ordinary route; consult
  these fields for recovery details.
- `exact_recovery_`: requested `'standard'` or `'extended'` capacity for the
  completed fit. It does not establish that exact recovery was used; inspect
  `diagnostics_['mm_function_recovery']` for actual attempts and accepted actions.
  Constructor changes affect the next fit, while serialized fitted state
  retains its provenance. Older objects without the option use the standard
  setting when cloned or refitted.
- `C_`: descriptive conversion from the fitted norm. It does not change `lambd`.
  A zero-norm conversion or an overflowed conversion is not a valid positive
  finite C for a separate SOCP fit. `C_conversion_finite_` identifies whether the
  conversion is finite; a finite zero conversion still is not a valid SOCP C.

### Dense named-linear L-BFGS final reporting

For an ordinary `KernGDWD` with `kernel='linear'` and an explicit
`backend='lbfgs'`, final reporting uses the represented training features when
the estimator itself constructs the kernel from its owned dense float64 array.
The route requires unchanged standard construction methods and kernel builders,
and no kernel keywords, supplied kernel/eigenbasis or reused CV kernel cache.
Subclasses, custom construction, sparse inputs and other kernels or backends
retain their existing reporting routes. The default `backend='auto'` is
unchanged. Callbacks and validation restoration remain supported on the
qualified route; diagnostics describe the state actually returned.

`diagnostics_['rkhs_geometry'] == 'linear_features'` identifies this route;
`rkhs_geometry_info` records its provenance and score/objective compatibility
checks. The norm, objective, stationarity, dual endpoint and `C_` all refer to
the exact real Gram matrix of the represented features. Primal and gradient
values remain numerical estimates. The dual endpoint uses outward arithmetic
bounds; subtracting it from an estimated primal does not certify that primal.
Original kernel reconstruction and optimizer score/objective checks retain
their existing tolerances. Unsupported arithmetic within the selected feature
helper, including erased derivatives or unrepresentable expansion terms, raises
an error rather than silently reporting convergence.

Optimizer updates, returned coefficients and prediction policies are unchanged.
Prediction still evaluates the rounded kernel, so this reporting geometry does
not promise identical signs under independently evaluated near-boundary query
formulas. No sparse or general nonlinear feature backend is introduced.

The extra final checks can be costly. In a fixed Python 3.12 study using 16
native threads, four paired blocks of short fits at shapes 64 by 4096 and 32 by
16384 added about 143 and 352 ms on Windows, and 446 and 830 ms
on Linux. These are workload-specific observations, not a vendor comparison
or an idle-host guarantee; Windows loaded both MKL and AOCL.
This tradeoff addresses numerical reporting correctness, rather than memory
savings or a universal speed improvement.

The preceding timings record the original reporting-correctness change, before
the later reduction of repeated report work. The current helper reuses
compensated operand expansions within each coefficient or gradient calculation
and shares raw/joint-affine score products on the compatible portable path.
Separate ordered sums, existing native scoring, numerical checks and exception
recovery are retained. Reuse is guarded and falls back when its required
dependencies are replaced or unsupported. This is an internal optimization of
the same eligible reporting route; it adds no public option or backend and does
not change optimizer updates or prediction policies.

A subsequent Python 3.12 comparison used matched fixed-work fits, with native
thread limits of 16 on Windows and 20 on Linux. In the primary fresh-process,
repeated-fit comparison, the 64-by-4096 workload saved about 5.38 ms on Windows and
42.69 ms on Linux. The ordinary 128-by-16 Windows report added about 57
microseconds; its whole-fit change was estimated at +49 microseconds, with
uncertainty spanning no change. The same ordinary workload on Linux saved
about 1.75 ms per fit. These are mean paired time changes against the development
implementation immediately before this optimization, not comparisons with a
released package or older numerical-library versions. They do not establish a
uniform speedup, negligible reporting cost, first-fit acceleration or total-CV
savings. The earlier correctness-versus-cost tradeoff therefore still applies.

The preceding implementation additionally introduced a separate native ordered
reducer for the final six-plane gradient sums, after the original products and
restoration checks. That reducer leaves coefficient/beta calculations unchanged. The existing qualified
route selects it automatically; absent, unsupported or replaced native entries
retain the original checked-sum path. The
[native contract](compiled_residual.md#ordered-linear-feature-gradient-reduction)
describes its scope and fallback behavior.

A November 2026 fixed-work comparison against the preceding development code
used the same inputs and model work, with 16 native threads on Windows and 20 on
Linux. For 128-by-512 fits, geometric mean time changes were about -3.8% first
fit/-10.0% repeated on Windows and -6.3%/-11.8% on Linux. For 256-by-2048 fits they
were -7.5%/-11.2% on Windows and -10.5%/-8.5% on Linux. Windows ordinary and
128-by-512 estimates had only two complete qualified blocks; larger Windows and
Linux estimates used five or six. These are workload observations, not a
convergence-rate, numerical-library upgrade or vendor comparison.

Costs remain relevant: the same panel's ordinary 512-by-32 Linux fits were
about 5.2% slower on first use and 2.8% slower on repetition (+12.72/+2.52 ms).
Without the new binary, ordinary Linux costs were about +5.6%/+3.6%
(+14.16/+3.68 ms). Earlier ordinary panels differed, so neither a universal
ordinary speedup nor a fixed penalty is established. First use includes lazy
loading, but source files were read beforehand; it is not cold operating-system
cache timing. No overall workload-weighted speedup is claimed, and smaller
reports or portable builds need not benefit. The earlier beta-extension
candidate was excluded after its component savings failed to provide a useful
incremental full-fit result across both hosts. This historical result is
separate from the subsequent warm-context candidate described below.

The November 9 follow-up selected three changes to final reporting. It reuses
report-specific scores, computes native gradient tiles from existing operand
splits, and uses the same expansion machinery for eligible coefficient
calculations once the context exists. The first call starts without a warm-up,
and unsupported calls follow the checked fallback. The model, optimizer work,
error bounds, callbacks and numerical thresholds are unchanged. The
[expansion contract](compiled_residual.md#exact-feature-expansion-tiles) gives
the full-ABI build and runtime limits.

The 600 fixed-work comparison fits used Python 3.12. Windows loaded AOCL-BLAS
5.3.0 and MKL 2026.2 with 16 native threads. Linux used MKL 2026.2 with 20.
For the full-native 128-by-512 workload, median paired time changes were +4.2% first/-13.5% repeat on Windows and -10.9%/-20.7% on
Linux. The 256-by-2048 changes were -10.7%/-14.6% and -27.2%/-37.0%, with all
six larger-case blocks faster on each host. Wide Windows/Linux results used
four/five eligible blocks. These compare with the preceding development code,
not the released package or old numerical-library versions. Fixed work does
not establish a convergence-rate or total-CV gain.

Cold and fallback costs remain: Windows ordinary 512-by-32 first fits were
+4.472 ms (all six slower); Linux wide first fits with the older residual
extension absent had median +14.517 ms and mixed repeat results. With the
new expansion binary absent, the first ordinary report cost +3.348 ms on Windows
and +9.639 ms on Linux relative to the simpler score-reuse candidate.
Correct fallback does not imply cost-free startup.

A separate 144-fit control used source-identical isolated copies with verified
checked-hash bytecode available, without importing numerical modules or
prewarming the fits. Ordinary first-fit medians were near neutral against
the simpler score-reuse candidate: +0.515 ms on Windows (four of five slower) and
-0.779 ms on Linux (three of six faster). Repeats saved median 1.538/5.712 ms,
with four of five/five of six faster. Windows used full native artifacts; Linux
omitted the old residual extension while retaining the expansion helper.
This is not an actual installation benchmark, a proof that every import used
bytecode, or a causal subtraction of compilation time from earlier cohorts.
All original exclusions and adverse blocks remain; no uniform speedup or
complete removal of the feature-geometry reporting cost is claimed.

The November 10 follow-up enlarges qualified warm gradient/beta batches from
2,048 to 16,384 entries, retaining the original first groups and checked
fallback described in the [expansion contract](compiled_residual.md#exact-feature-expansion-tiles).
A reporting-only comparison used saved fitted states with a common reconstructed
score input, without running new fits. Six balanced blocks per shape compared
this change with the preceding 2,048-entry development implementation. Each
process made one first report and 12 warm reports; warm comparisons use each
process's mean, not 12 independent observations. All valid blocks, including
adverse results, were retained regardless of background CPU load.

Median paired report-time changes were:

| Samples by features | Windows first | Windows warm | Linux first | Linux warm |
| --- | ---: | ---: | ---: | ---: |
| 512 by 32 | -3.1% | -11.0% | -12.3% | -16.2% |
| 128 by 512 | -6.8% | -14.1% | -8.2% | -15.9% |
| 256 by 2,048 | -9.2% | -16.8% | -13.2% | -24.1% |

These are descriptive reporting results on the two tested hosts, not a uniform
speedup: some first-report blocks and one ordinary Linux warm block were
slower. Imports and saved-state preparation were outside the clocks, and the
reconstructed score inputs do not replay missing historical compatibility
arguments. The results do not establish cold-import, whole-fit, convergence or
cross-validation gains, or erase the earlier startup and fallback costs.

## Explicit validation and callbacks

```python
from sklearn.model_selection import train_test_split

X_fit, X_monitor, y_fit, y_monitor = train_test_split(
    X, y, test_size=.25, stratify=y, random_state=5,
)
monitored = KernGDWD(
    kernel='rbf', kernel_kws={'gamma': 1.0}, lambd=.02,
    stopping='validation', max_iter=100, patience=3,
    min_delta=0., check_interval=1,
).fit(X_fit, y_fit, validation_data=(X_monitor, y_monitor))
print(monitored.n_iter_, monitored.returned_iteration_, monitored.validation_history_)
```

At each checkpoint, accuracy must exceed the previous best by more than
`min_delta` to count as an improvement. `patience` counts non-improving checkpoints,
not necessarily iterations. Checkpoints include iteration 1, subsequent positive
multiples of `check_interval`, and the maximum-iteration endpoint. The
zero-iteration case checks the initial state. The best checked state is restored
on return, including when the cap is reached. Its coefficients, intercept,
prediction precision and representation status are restored together. `validation_history_` records
checked iteration numbers and scores. Labels must belong to the training classes;
a one-class monitoring set is allowed.

Validation data is accepted only with `stopping='validation'`. Neither `run_cv`
nor `KernGDWDCV` secretly takes its scoring fold for online stopping; their direct
validation-stopping setting is rejected. Construct training/monitoring splits
inside each CV training fold when that extra procedure is intended. A final test
set must remain outside this selection process.

Callbacks are optional and receive a mapping with `iteration`, `alpha`, `offset`
(also `intercept`), `objective`, `training_scores`, `decision_values`, and
`elapsed_seconds`. The initial state is included. `training_scores` means
`K @ alpha`; `decision_values` includes the intercept. These are optimizer
trace values, not a recomputation using joint affine scoring, so their
rounded margins may differ from public decision scores. Arrays are protected
copies, so callbacks cannot modify the numerical state. Return `True`, or raise
`StopIteration`, to stop; return `False` or `None` to continue. Other return types
raise.

For optimized spectral and L-BFGS fits, observing a checkpoint also verifies its
scores and objective against the original kernel representation. This can reject
a highly cancellation-sensitive state before a callback or validation rule sees
it. The numerical tolerances are consistency estimates, not exact arithmetic
certificates. Callback and validation timing includes this additional work.

```python
def report_and_stop(state):
    if state['iteration'] == 5:
        return True

controlled = KernGDWD(stopping='fixed', max_iter=20, callback=report_and_stop)
```

A callback stop is reported explicitly. Combining callback and validation
stopping returns the best validation state already checked, if one exists.
Validation monitoring allocates its query-by-training kernel; prediction batching
does not currently batch this monitoring allocation.

## Kernels, batching, and caches

Named kernels use sklearn pairwise-kernel parameters. Known-PSD auto selection
covers linear and nonnegative-gamma RBF kernels, plus polynomial kernels with a
nonnegative integer degree and nonnegative gamma/coef0. Sigmoid, callable, and
precomputed kernels do not receive that automatic PSD assumption.

A matrix-level callable receives `(X_train, X_query, **kernel_kws)` and must
return a training-by-query matrix. `kernel='precomputed'` instead follows the
public sklearn input convention: square training Gram matrix in `fit`, then
query-by-training values in `predict` or `decision_function`. Standard sklearn
pairwise tags and built-in CV slicing preserve this orientation.

`KernGDWD` and `KernMD` retain their own feature copies and fitted `kernel_`
and `kernel_kws_`. Changing the caller's features or kernel-keyword dictionary
after fitting does not change predictions. Constructor kernel changes apply
on the next fit. Nested keyword values are copied, but a callable kernel remains
the same callable: its external state and behavior must remain stable.
Precomputed mode retains a shape-stable view without copying the square matrix,
because prediction uses the supplied query kernel rather than training values.
`KernGDWD.cv_init` does own its precomputed matrix, because later fits reuse
those values. Older serialized models without these fitted attributes retain
their historical parameter lookup until refitted.

For `KernMD`, copying the feature basis and nested keywords adds work during fit,
with no new prediction-stage copy. Keyword values must support deep copying;
if capture fails, the original exception is raised and learned state is cleared,
including after an attempted refit.

Set `prediction_batch_size` to a positive integer to bound the number of query
rows used by each prediction kernel calculation. This supports named, callable,
and precomputed kernels. Callable kernels must compute consistent pairwise values
when their query rows are partitioned. The training Gram matrix is still dense.
The current setting applies after fitting, including in `KernGDWDCV`. A call to
`decision_function(X, prediction_batch_size=...)` overrides the limit for that
call only; explicit `None` disables batching. The CV wrapper forwards this limit
without changing its fitted child's parameter. Invalid limits fail before kernel
construction. Partitioning can change rounding at the last few bits in numerical
library operations; it does not change the prediction formula.

Completed scores are copied into the result before releasing each temporary
kernel block. When the package owns those blocks, the previous block no longer
overlaps the next allocation. This does not free caller-owned precomputed
matrices or arrays retained by a callable, and single-batch prediction receives
no such saving. Kernel and scoring arithmetic are unchanged. See the
[allocation checks and timing limits](../VALIDATION.md#prediction-kernel-block-lifetime-in-local-source).

Eligible built-in standard RBF prediction reuses training norms within a batched
call for supported float64 dense arrays and canonical, native float64 CSR
matrices or arrays. CSR canonicality is checked from the current coordinates,
without trusting mutable ordering flags. The training norms and validation
results are discarded when the call returns. Custom numerical hooks,
unsupported representations and unsuitable arithmetic retain the checked
fallback. This does not make an unsafe historical fitted basis valid or support
concurrent mutation of fitted data.

For an exact `KernGDWD` using `kernel_computation_='guarded_norm_sum_rel_v2'`,
eligible dense batches also reuse the public call's query validation. The
query must retain the checked native float64 ndarray storage and C-contiguous
layout, and each batch must be a contained view of that query. Fresh finite
training norms establish training-basis finiteness for the current call.
Unchanged validation and numerical helper bindings, their supported defaults,
and the caller's NumPy error policy are checked before this route is used.
Unsupported inputs, policies or replaced hooks retain the original checked
fallback. Later public calls validate afresh; no validity state is stored on
the estimator. The guarded kernel arithmetic and repair checks are unchanged.
This internal reduction in repeated validation adds no public option and does
not imply that batching is faster for every workload.

### Integer sparse coordinate inputs

At supported sparse input boundaries, duplicate signed and unsigned integer
entries at the same coordinate are summed exactly, then each sum is converted
once to float64. For example, `[2**53 + 1, -2**53]` represents the coordinate `1`.
Summation does not wrap in the source integer dtype. A large final coordinate
can still round on conversion to float64; this is an input-representation rule,
not exact arithmetic throughout the solver. Already overflowed or rounded
input cannot be repaired.

The rule applies wherever sparse input promotion already occurs, including
linear and kernel fitting, CV preparation, validation and prediction, and to
integer sparse queries for linear classifiers. Linear prediction skips this
extra conversion when CSR, CSC or BSR storage is verified canonical, COO
coordinates are verified sorted and unique, or the input is valid DOK or DIA.
Those queries retain ordinary validation and dot-product dtype conversion,
including the existing behavior with custom float32 or extended-precision
coefficients. Unsorted compressed/COO storage and LIL integer queries use the
normalization path.

Integer sparse precomputed queries use the same rule; precomputed training
still requires a dense Gram
matrix. Callable kernels receive normalized integer feature inputs where
supported. Their returned kernel matrices retain the existing callback/output
contract. Direct kernel/affine score helpers still interpret each stored
contribution through binary64.
Floating input dtype and summation policies are unchanged. Boolean entries
retain cast-first counting behavior, and complex input validation is unchanged.

Normalization preserves caller-owned storage and cached ordering flags.
Ordinary inputs use guarded conversion paths; exceptional integer duplicates
can require sorting and arbitrary-precision sums. This does not extend support
to storage or layouts otherwise refused by the dependency or public validator,
or establish a speed guarantee.

This strengthens the earlier entry-wise conversion policy. Affected coordinates,
kernels, fitted coefficients, scores and labels can change. Existing fitted
parameters and historical results are not rewritten. To use the new training
semantics, refit from the original integer storage; new queries use the current
input policy even for an older fitted estimator. Existing historical-basis and
source/cache guards still apply and are not bypassed.

### Sparse Laplacian input ownership

Named `kernel='laplacian'` calculations preserve caller-owned CSR feature
buffers and ordering flags, including read-only storage. The actual CSR
structure is checked without trusting cached ordering flags. Noncanonical
query or fitted-basis storage is copied privately before sklearn sorts or
combines coordinates; canonical CSR and dense inputs retain their existing
representation. After the integer input normalization described above, the
Laplacian kernel arithmetic and floating duplicate aggregation are unchanged.
Sparse training support remains estimator-specific.

The same protection applies to validation and prediction, including batched
prediction and historical fitted bases. Structure checks and private copies
can add runtime cost. A noncanonical fitted basis may require this work again
for each prediction batch; this is an ownership guarantee, not a universal
speed improvement.

### RBF construction and query consistency

Sparse feature coordinates are combined before named RBF row norms are
calculated. Repeated CSR entries therefore have the same meaning during fitting,
CV preparation, validation and prediction. Normalization uses private storage;
caller buffers and ordering flags are preserved, and nonfinite duplicate sums
raise an error. Corrected Schur fits promote floating sparse values to float64
before a format conversion can sum duplicates in a narrower dtype. Integer
inputs follow the exact-coordinate rule above. Callable returned kernels and
precomputed training matrices retain their separate contracts; supported
integer sparse precomputed queries follow the same input normalization.

An old standard-RBF CV cache containing noncanonical CSR features is rebuilt.
Its stored Gram matrix may have squared individual storage entries instead of
the combined coordinate. An older fitted norm/dot RBF model with duplicate
training coordinates must be refitted from its original data, labels and
settings: changing prediction distances cannot repair its fitted coefficients.
Prediction raises an explicit error for this case. Unique unsorted coordinates,
duplicate query coordinates and saved direct-distance models remain supported.
Canonicalization addresses sparse representation; it does not eliminate
floating-point cancellation between large feature vectors.

For new corrected named RBF fits, the default `rbf_computation='standard'`
selectively recomputes cancellation-sensitive squared distances using coordinate
differences. Eligibility requires the original construction hooks, supported
scalar gamma without extra kernel keywords, and verified numerical helper
provenance. NumPy error-policy selection does not change this eligibility.
Parameters may use an ordinary `dict` or `collections.OrderedDict`; custom
mapping or scalar subclasses retain their original routes. Supported scalar
representations include ordinary Python scalars, NumPy binary16/binary32/binary64
floats and ordinary integer scalar types. Newly admitted integer types must be
exactly representable in binary64; historical Python integer and `np.int64`
conversion behavior is preserved. True extended-precision scalars retain the
original route. Gamma must remain finite and nonnegative, or `None` for the
default. Other standard-policy routes retain their existing construction.
The Gaussian definition `exp(-gamma * ||x-y||**2)`,
regularization, loss and free intercept are unchanged.

The existing internally masked arithmetic is retained when underflow is ignored
and all NumPy error modes are `warn` or `ignore`. For other caller policies,
actual norm/distance, scaling and exponential operations honor those policies,
including the exponential evaluation of repaired pairs. Genuine events can
therefore warn, raise or invoke the caller's handler; handler exceptions propagate.
Internal error-bound bookkeeping retains its scoped masks. Exact-distance
arithmetic does not manufacture incidental floating-point overflow events.
An exponent beyond binary64 range is saturated to its largest finite value
before the policy-aware exponential: both values round to a zero kernel entry,
and the actual exponential can report underflow. Historical event counts/order
and bitwise equality between scalar exponential implementations are not promised.
This construction policy does not establish strict-error-policy compatibility
for every separate solver or scoring operation.

The trigger compares the computed distance with a conservative forward-error
envelope divided by `sqrt(eps)`. It catches positive but poorly conditioned
distances as well as nonfinite or nearly cancelled results. This is a numerical
conditioning rule under ordinary floating-point assumptions, not a universal
interval certificate. It does not guarantee every unflagged kernel bit, every
prediction sign, or a positive-semidefinite floating-point Gram matrix. Existing
PSD, residual and objective checks remain in force; no jitter or eigenvalue
clipping is introduced.

Ordinary unflagged pairs use the selected evaluator's multiplication and exponential.
A global bound can skip the flag scan when every pair is safely outside the
trigger. When the norm and distance bounds establish the required finite
envelope, the tiled scan omits redundant finite-value masks. In the self-norm
case, a tile whose minimum distance exceeds that bound can be skipped. These
shortcuts preserve the same flagged pairs; uncertain inputs retain the full
predicate. Flag storage requires `n_train * ceil(n_query / 8)` bytes, plus
256-by-256 temporary tiles and one unpacked flag row. A conservative budget for
the visible tile arrays is 4 MiB, separate from the retained flags, kernel
matrix, numerical-library allocation overhead and direct-repair feature scratch.
Selecting rows with flags also uses O(n_train) Boolean and integer storage.
Only actual row identity in a
self-kernel permits setting a diagonal entry directly to one. Off-diagonal
duplicates and cross queries retain the checks. A large fraction of flagged
pairs can require substantial direct-distance work; bounded scratch space does
not imply a small running time.

For an exact `KernGDWD` instance using standard named RBF, Schur mode, a
privately captured float64 dense or CSR basis, and unchanged supported fit,
construction and cache hooks, self-kernel construction now selects the guarded
norm-sum evaluator upfront. It adds the two squared norms before subtracting
the dot-product term, applies the same selective repairs, mirrors one computed
triangle and sets the self-diagonal to one. This avoids first constructing a
matrix that would then need replacement. It also changes arithmetic for healthy
already-symmetric matrices and subsequent cross queries; it is not merely a
postprocessing change to diagnostics. The Gaussian kernel definition is unchanged.

Other internally constructed standard RBF routes retain the existing rebuild
when asymmetry exceeds `100 * eps * max(1, max(abs(K)))`. Custom or supplied
matrices, precomputed kernels, direct-RBF mode and unqualified construction
hooks do not acquire the new self-construction policy. Exact
symmetry enables applicable certificate/recovery checks but does not prove that
the represented matrix is PSD. Spectrum, exact-factor and negative-quadratic
checks remain unchanged; no eigenvalue clipping, jitter or relaxed threshold is
introduced.

New qualified fits record `kernel_self_construction_='norm_first_v1'` and
`kernel_computation_='guarded_norm_sum_rel_v2'`. The self marker identifies
construction, not certification. Other guarded routes can record
`'guarded_sklearn_rel_v2'`; other standard routes record `'sklearn'` or
`'norm_sum'`. `kernel_symmetry_correction_` records observed asymmetry that was
corrected, not a comparison with an unevaluated alternative construction.
Prediction and explicit validation retain the fitted construction, including
when NumPy error handling is later changed; invalid feature inputs are still
rejected. Deliberately replaced numerical hooks remain observable. Copied or
partitioned queries are not promised bitwise identity because BLAS/batch rounding
and self-kernel mirroring can differ.

Older serialized models, including guarded v1 models, retain their recorded
construction and arithmetic behavior; refit to use the current correction.
The duplicate-training-coordinate exception described above
requires refitting. New CV preparation records its construction generation;
older incompatible generations are rebuilt before a new fit, and unversioned
old caches are rebuilt when a new fit would select the corrected policy. The
self-construction generation is checked separately from the fitted cross-query
mode; changing a label never makes a stale matrix reusable.
Loading models with new construction modes in older packages is
unsupported. Example checkpoint identities include the runtime modules.

Bounded two-environment measurements found construction savings and mixed
whole-fit/CV costs; they do not establish a universal speedup. Small score changes
can change labels near zero. No universal sign parity across old and new fits,
row orders or independently evaluated query batches is promised. The existing
score and objective checks retain their original tolerances.

Supported public kernel fits share this construction policy.
The numerical-library matrix products use configured native thread limits;
the reconstruction does not forward `kernel_kws['n_jobs']` into pairwise-kernel
workers. Automatic solver restart has the narrower eligibility
described above.

### Optional direct RBF computation

`rbf_computation='direct'` is an explicit option for numerical accuracy on
`KernGDWD` and `KernGDWDCV`. It evaluates every pair directly rather than
selecting pairs with the standard policy's conditioning trigger. The direct option is included in DWD
1.3.11; the earlier 1.3.10 wheels do not include it.

```python
model = KernGDWD(
    kernel='rbf', kernel_kws={'gamma': .1}, lambd=.02,
    rbf_computation='direct',
)
```

Direct computation requires the named `'rbf'` kernel,
the package's original construction hooks, and `kernel_kws` containing only
`gamma` or no entries. Gamma must be finite and nonnegative; `None` keeps
`1 / n_features`. It supports public kernel fits with
validated float64 dense or CSR features. Other kernel families,
callable/precomputed kernels, extra kernel keywords and caller-supplied
training `K` raise when direct computation is requested; they retain their
existing behavior under the standard policy.

The calculation uses coordinate differences in bounded dense tiles or a
sparse row merge. Exceptional overflow and very small squared distances use
exact arithmetic on the represented inputs before rounding the gamma-weighted
exponent. Gamma zero returns ones. This addresses the demonstrated equal-row
and common-offset cancellation failures without changing
`exp(-gamma * ||x-y||**2)`, adding jitter, or modifying a caller's kernel.
The dense output still requires one value per training/query pair, and the
usual PSD, residual and objective checks remain in force.

The fitted `kernel_computation_='direct_v1'` is used for training, prediction
and explicit validation. Changing the constructor option takes effect on the
next fit. Existing serialized models retain their fitted standard policy;
CV caches and resumable checkpoints distinguish the requested computation.
Loading a new direct model in an older package that lacks `direct_v1` is
unsupported. Use `rbf_computation='direct'` on `KernGDWDCV`, or
`kerngdwd__rbf_computation='direct'` on the `make_pipeline` example in the
[README](../README.md#explicit-parameter-search). The standalone resumable-CV Python
function accepts a `KernGDWD` configured this way; no separate CLI flag is added.

Eligible built-in direct-RBF prediction validates the retained training basis
once per public call, while still checking each query batch. Sparse basis
canonicalization is private to that call; no validity cache or temporary buffer
is stored on the model. Later calls validate afresh, and custom construction or
scoring methods retain their existing behavior. This internal reuse is automatic
within the direct path and introduces no additional numerical option.

Direct computation can be substantially slower. In ten paired MNIST 2-vs-3
prediction measurements per estimator with one native thread, it took about
18 times as long for full-kernel DWD and 6 times as long for ensemble DWD.
Four pairs had background-CPU flags; the unflagged measurements led to the
same conclusion. These are workload-specific costs, not universal ratios.
Six saved MNIST fits retained their test labels and accuracies, and all three
ensembles retained their selected rows. More sensitive fixtures changed
coefficients, row rankings and labels, so matching results are not guaranteed.

Direct distances are not universally correctly rounded or compensated sums.
On an adversarial 4,097-feature test, sequential summation gave relative kernel
errors of about `2.27e-13` at gamma 1 and `1.59e-10` at gamma 700. Correcting
kernel construction also does not eliminate rounding in `K @ alpha + b` or
promise identical scores for every batch shape. The separate prediction
precision checks below still apply; numerical recovery limits are unchanged.

### Prediction precision

With `affine_computation='standard'`, corrected models evaluate dense and CSR
query matrices adaptively. The ordinary
product is accepted per row only when a conservative estimate is no larger than
`5e-7 * max(1, abs(ordinary_score))`. The estimate includes positive-dot rounding
and gradual underflow; an overflowing estimate sends that row to expanded
products instead of rejecting a potentially finite cancellation result.
Uncertain rows use high/low products with accurate summation. CSR duplicate
entries count as their separate stored contributions. Expanded-product storage
is O(n_training); adaptive screening retains its existing block of at most 128
query rows.

Dense score evaluation prepares the coefficients' high/low split once per
helper call, when the first expanded row is needed. The preparation is reused
only within that call; it is not stored on the fitted model or shared between
prediction batches. Product arithmetic, summation order and per-row numerical
validation remain unchanged. Empty queries and adaptive calls whose ordinary
scores all pass need no split; sparse rows retain their existing evaluation.

Dense rows requiring expanded evaluation are automatically grouped in batches
of at most eight. Each row retains the same elementwise product arithmetic and
high-then-low `math.fsum` term order. A conservative 1 MiB allowance sizes the
product-array batches; it is not a hard cap on temporary memory or process RSS.
Batch size decreases as the number of training coefficients grows; above 4,096
coefficients, evaluation remains row by row. Sparse inputs, single-row
evaluations and floating dtypes wider than 64 bits also retain the row-wise path.
If allocation fails, batch arrays are released before retrying row by row, and
batching is disabled for the rest of that helper call. An allocation failure in
the row-wise retry still propagates. This adds no option and leaves precision
selection, estimator defaults and DWD mathematics unchanged.

If the fit requires fully compensated original-kernel reconstruction, all later
query rows use that policy. The ordinary/adaptive product and the fully
compensated alternative face the same fit score and objective checks. An
objective-only mismatch also permits one compensated reevaluation. Validation
uses the same query policy as the corresponding public model, so a sensitive
state cannot be accepted for stopping using one precision policy and exported
under another. No inconsistent callback state is emitted. These are numerical
estimates under the stated floating-point model, not universal interval proofs.

<a id="optional-joint-affine-scoring"></a>

### Joint affine scoring

New `KernGDWD` and `KernGDWDCV` models default to `affine_computation='joint'`.
The former `'auto'` alias already selected joint scoring and is now rejected
during fitting or CV preparation. Replace it with `'joint'` in constructor
settings, `set_params` requests and parameter grids. This is an API spelling
change, not a new scoring algorithm or additional numerical work. Other
controls' supported `'auto'` settings remain unchanged.
Already-fitted legacy models retain their recorded standard scoring; they are
not refitted or upgraded during prediction. Explicit `'joint'` and `'standard'`
remain available for new fits. It supports the same
named, callable and precomputed kernels as corrected kernel DWD. It is
independent of `rbf_computation`: joint scoring evaluates the supplied kernel
entries more accurately but cannot repair inaccurate entries themselves.

```python
model = KernGDWD(
    kernel='rbf', kernel_kws={'gamma': 1.0}, lambd=.02,
    affine_computation='joint',
).fit(X, y)
```

The expression remains `K @ alpha + b`. Adaptive screening includes both the
dot product and intercept addition. It accepts an ordinary row only when its
error estimate is at most `5e-7 * max(1, abs(score))` and is smaller than the
absolute score. Uncertain rows sum expanded high/low products together with
the intercept before rounding. Models requiring compensated prediction skip
the ordinary screen. Exceptional range or unresolved cancellation can require
an exact rational sum of the represented inputs. Product temporaries are
processed in bounded groups, with row-wise retry after an allocation failure;
the exact fallback has no fixed wall-time limit. A guarded exact-zero shortcut
can avoid rational summation when the product expansions preserve every bit.
It requires the trusted helpers, round-to-nearest arithmetic and sufficient
product exponent range; underflow or uncertainty retains the exact fallback.

The DWD loss, regularization, unregularized intercept, solver updates and
acceptance thresholds are unchanged. Binary classification still chooses the
second class only for a strictly positive returned score. A tiny positive exact
value can round to zero when it is below binary64 range; the existing zero tie
rule still applies. Ordinary accepted scores are not guaranteed to be correctly
rounded, and the bounds rely on the stated binary64 arithmetic model.

Explicit validation stopping uses the same joint scoring as public prediction.
Correcting a rounded score may change a label, the best validation checkpoint
or the stopping iteration. CV candidate scoring likewise uses the selected
policy; neither the fold split nor its scoring rule changes. Constructor
cloning copies the requested option; serialization preserves the fitted
`affine_computation_`. Changing a parameter after fitting does not change the
stored scoring policy. Missing requested fields use current defaults for
cloning and refitting; explicit stored policies are preserved.

The repair was reproduced with a genuine two-observation fitted linear-kernel
model: the standard score was zero while the exact represented affine value
was positive, and joint scoring recovered its sign. Locked MNIST controls
retained all tested labels, ensemble rankings and selected samples. This is a
conditional precision benefit, not evidence of a general accuracy increase.

The original opt-in study recorded ten matched prediction pairs per workload:
full-kernel pipeline timing was approximately unchanged, while the difficult
ensemble pipeline cost about 9% more. Other numerical test workers had exited
before timing. Three of twenty pairs had background-CPU flags; excluding them
gave the same conclusion. Constructed exact-tie rows were substantially slower
in that implementation because every row required exact arithmetic. These are
historical observations; its initial recommendation to keep standard scoring
as the default was superseded. The subsequent guarded exact-zero shortcut also
means those original tie timings do not predict every current exact-tie cost.

Current joint scoring is the default because the combined-expression check
protects against intercept cancellation that can change a decision sign.
This correctness benefit does not imply a universal accuracy increase or a
uniform speed improvement. Standard remains an explicit, separately rounded
alternative when representative score checks, margins and complete prediction
timing support its accuracy/cost tradeoff; it is not a shortcut for sensitive
near-zero decisions. See [when to choose arithmetic policies](choosing_options.md#arithmetic-and-recovery)
and [validation scope](../VALIDATION.md) for the evidence and limits.

### Fitted state, cross-validation and caches

### Default CV grids and display

`GenDWDCV` and `KernGDWDCV` use immutable tuple defaults for `lambd_vals`
and `q_vals`: respectively `tuple(np.logspace(-2, 2, 10))` and
`tuple(np.logspace(-2, 2, 5))`. The values and ordering are unchanged; the two
default grids still form 50 combinations before any kernel-keyword expansion.
The container change fixes parameter representation and shared-default
mutability; it does not change fitting, scoring or the winner rule.

Replace a grid instead of mutating the default tuple in place. Explicit lists,
tuples and NumPy arrays remain supported, including callers that need their
own mutable array:

```python
import numpy as np
from dwd.gen_dwd import GenDWDCV

model = GenDWDCV(lambd_vals=np.array([0.01, 0.1, 1.0]), q_vals=[1.0, 2.0])
model.set_params(lambd_vals=np.array([0.05, 0.2]))
```

Code that relied on the omitted default's ndarray methods such as `.shape`,
`.dtype` or `.copy()` must explicitly create its required array, rather than
assuming every grid is an ndarray. This does not require changing an already
explicit array-valued grid.

Plain-text estimator representation and notebook HTML also involve
scikit-learn. The separate affected-version HTML encoding issue is described
in [estimator display and dependency maintenance](dependency_display.md).

Starting a new fit clears learned state, including the kernel prediction-precision
flag and fitted affine policy. If fitting or refitting fails, learned state is cleared before the exception
propagates; prediction then raises `NotFittedError`. This contract applies to the
package's linear and kernel classifiers, their CV wrappers, optional conic
classifiers, and `KernMD`. Constructor parameters and validated private
precomputation caches are preserved. See [fitted-state behavior](failed_refit_state.md).

`cv_init(X)` explicitly caches the exact training kernel and, where needed, its
eigensystem. `run_cv` sets candidate kernel parameters before preparing that
candidate's fold cache. Default CV caches only K and keeps Cholesky;
explicit spectral
or L-BFGS CV also prepares eigenpairs for reuse. Unknown kernels may still need
an eigenvalues-only PSD check at fit time. Cloning clears learned state and does
not transfer hidden caches.

Generic `run_cv`, `GenDWDCV` and `KernGDWDCV` require finite real numeric scalar
train/test scorer outputs. Invalid outputs raise before choosing/refitting a
winner rather than silently excluding a candidate. Numeric Python/NumPy scalars
and zero-dimensional numeric arrays are supported; object-dtype scorer results
are outside this contract. Valid-score aggregation and first-tie ordering are unchanged.

Cache equality accounts for kernel parameters (including nested arrays), data
values/order after the solver's dtype conversion, sparse versus dense representation,
corrected solver geometry, backend, and requested `rbf_computation` and
`affine_computation`, plus the requested `exact_recovery` capacity. Corrected fits
compare the promoted float64 features they actually use, allowing identical
float32 inputs to reuse preparation. Preparation without the current geometry
marker is rebuilt rather than reused. The RBF construction
policy is cached with K; automatic restart also checks construction provenance. A
lambda/q change can reuse the same eigensystem. Different subsets or feature
sets cannot generally share one. Stateful callable kernels must be treated as
immutable while cached. Updating an opaque callable's captured state is not a
reliable cache-key change.

Advanced `fit(..., K=..., K_eig=(U, eigenvalues))` inputs must correspond to the
exact training data, order, preprocessing, and kernel. The package validates
supplied eigenpairs against K: every column norm, deterministic orthogonality
actions, and eigen-equation actions are checked, along with spectral invariants.
Large-matrix validation uses a bounded set of O(n^2) probes; small matrices receive
full checks. Valid supplied ordering and signs are preserved. Invalid supplied
pairs raise without automatic recomputation. This establishes numerical matrix
consistency, not the provenance of externally provided K. Never slice full-data
eigenvectors to manufacture fold/subset eigenpairs. These parameters are
preparation inputs, not alternative prediction-kernel definitions.

## Numerical validation and recovery

Corrected feature paths promote to float64 before constructing their Gram
matrices. Casting an already rounded float32 external kernel cannot repair its
lost precision. Matrices must be finite and symmetric. Spectral admission rejects
eigenvalues that are materially negative relative to the spectrum's own scale,
as detailed below. The initial quadratic-form guard covers both native and
explicit coefficients. If the ordinary length-n bound is overconservative,
1.2.1 uses bounded-memory compensated accumulation to check the same represented
quadratic. Acceptance includes the actual observed quadratic's discrepancy from
the compensated result plus a forward-error bound for products, summation and
gradual underflow. The precision threshold is unchanged. The optimizer reuses
the checked scores and scalar quadratic without changing alpha, b or the model.
Unsafe states, overflow and unsupported flush-to-zero arithmetic still raise.

Diagnostics record `initial_quadratic_method`, ordinary and compensated bounds,
the observed discrepancy, `initial_quadratic_acceptance_threshold`, and
`initial_state_check_seconds`. This initial check does not replace later solve,
score, objective or stationarity safeguards. Its bounds assume the documented
floating-point behavior, not exact originating Gram measurements. Tiny
negative spectral values consistent with rounding may become
zero, but positive eigenvalues are never truncated and no extra ridge or
intercept penalty is silently added.

For a validated finite spectrum with `n` eigenvalues, let
`s = max(abs(eigenvalues))` and let `eps` be binary64 machine epsilon. When
`s > 0`, the numerical PSD check accepts only if
`min(eigenvalues) / s >= -100 * eps * n`. An all-zero spectrum passes this
PSD check, subject to the existing arithmetic guards. There is no unit-scale
floor: shrinking an otherwise materially indefinite spectrum does not make
its negative directions count as roundoff. This is a floating-point roundoff
policy, not an exact PSD certificate, and it never discards positive modes.

The normalized comparison determines admission. The diagnostic `psd_tolerance`
reports the rounded value of `(100 * eps * n) * s` in the original eigenvalue
units; this displayed value can round to zero near underflow. The tiny-scale
path requires supported gradual underflow and does not change the CPU mode.
Initial spectral admission and fresh original-kernel spectral recovery use the
same policy. Known-family optimized auto/Cholesky routes that already omit a
spectral check keep that routing; this change does not add universal spectral
validation. Symmetry, shifted-system positivity, residual, reconstruction and
optimizer checks retain their separate rules.

Passing numerical PSD admission does not guarantee that a dual bound can be
reported. A later contradiction found by the final dual calculation still
produces an unavailable bound and gap under the reporting policy above; it does
not establish exact PSD or global optimality.

### Eigensystem validation

Computed full eigensystems must pass numerical validation before use. The native
sequence is EVD, EVR, then EVX. Each candidate is checked against the same unchanged
matrix; a failed basis is released before another attempt. EVD uses divide and
conquer, the algorithm family used by upstream NumPy eigendecomposition. This
driver choice changes numerical preparation, not the kernel objective, full-rank
representation or free intercept.
Different valid bases can differ in signs, repeated-eigenvalue coordinates and
roundoff; matching a historical buggy trajectory is not an acceptance criterion
for the supported corrected solver.

The earlier MKL-specific analytic self-probe and compatibility-mode child process
have been removed. Per-result checks remain generic and also validate supplied
bases. No subprocess, temporary matrix exchange, dependency installation,
provider-mode change or global NumPy/SciPy monkeypatch occurs. If all native
drivers fail validation, the error identifies the failed checks and recommends
updating or replacing the numerical libraries and BLAS/LAPACK runtime. Memory
failures and cancellation propagate immediately rather than trigger more attempts.
The documented runtime update resolved the failures reproduced in that environment;
other installations still require per-result numerical checks. No blanket vendor-version restriction substitutes for
checking the actual result.

Native EVD was faster than EVX in bounded tests on the updated host, but it can
use more dense workspace. Driver performance depends on matrix size, spectrum
and numerical library. Multiple failed attempts can also be expensive. These are
preparation costs and must be included in reported fit times; they do not establish
a universal whole-fit speed advantage. Release-specific validation is separate
from the historical 1.2.1 release.

The generic signed-matrix helper does not impose a PSD policy; kernel consumers
apply their own checks. The linear augmented-system helper uses the same
validation machinery. Eigenvalues-only checks use bounds, trace and the sum of
squared eigenvalues; they cannot certify eigenvector correctness and report
that limitation. Historical wheel and training-fit checks are documented in the
[release record](../VALIDATION.md); they are not relabeled as tests of this release.

### Constrained solves and residual checks

Optimized Cholesky uses a reciprocal-condition estimate as diagnostic information,
then checks each candidate solve against the original constrained MM equations.
If necessary, bounded residual refinement or a centered factorization solves
those same equations. The centered constant direction is an internal solve
device, not extra regularization or a penalty on the fitted intercept. Checked
spectral inverse actions retain their validated eigenbasis and bounded refinement.
If an MM proposal still
fails, the bounded exact-function recovery below can be attempted before the fit
raises. A failure in a callback or validation routine is not intercepted as an
MM numerical failure.
Initial coefficients, objective, lambda, and stopping policy are not changed by
these numerical repairs.

Ordinary residual acceptance includes forward-error allowances for the matrix
product, scalar operations and coefficient sum, together with a lower bound
for its relative RKHS scale. An inconclusive check can recompute the same product
in short BLAS dot products with bounded accumulation; it uses at most 128 partial
vectors. Unsupported arithmetic or unresolved cancellation retains the existing
compensated check. These estimates require the documented floating-point and
BLAS assumptions; they are not interval certificates for arbitrary runtimes.

With explicit `residual_check_order='refinement_first'`, a failed bounded ordinary
residual check first triggers one inexpensive native correction using the existing
factor or basis. The solver accepts that trial only
after fresh ordinary checks of the corrected state and scores. If the trial
fails or produces a numerical error, it is discarded and the original state is
reassessed using compensated float64 products and summation, with allowances for
rounding and gradual underflow. This preflight is bounded to one trial per
factor representation or checked spectral candidate. The original and centered
Cholesky forms provide at most two native-trial representations; anchor LU
recovery uses compensated residual measurement instead. The spectral system has at
most five inverse representations:
its initial checked basis, a power-of-two equilibrated shifted system, and fresh
validated EVD/EVR/EVX inverse preconditioners for the same original kernel.
Only eigenvalues within the existing negative-roundoff allowance can be floored
for preconditioning; every positive mode remains. This modifies the auxiliary
inverse action, not the stored kernel or final objective. Each representation's
accepted candidate must pass the original-equation checks. It does not consume
an MM update. Native-trial counts, accepted
corrections, discarded trials and their costs are recorded separately. If a representable scalar intercept correction
can resolve a constant residual, it is committed only after a fresh full check.
The checked spectral inverse-action route normally permits three correction
steps; a contracting
original-equation residual can extend that private refinement budget to eight.
These are corrections within an MM update, not extra MM iterations or a change
to `max_iter`.

For eligible optimized Cholesky fits, with or without `acceleration='restart'`,
`residual_check_order='adaptive'` can assess the untouched state accurately before
trying another correction after two discarded trials whose original states passed
accurate assessment. The same acceptance checks apply. The rules for restart
proposals, objective-descent checks and restart decisions are unchanged. This order
can change finite-iteration results. The original 1.3.9 feature supported only
unaccelerated fits; version 1.3.14 adds the restart combination. See the
[original feature scope](candidate_changes.md) and the
[restart validation](../VALIDATION.md#version-1314-development-validation).
New estimators default to `'auto'`, selecting adaptive ordering only for
optimized Schur fits with backend auto/Cholesky and no supplied or cached
eigenbasis. Other auto routes use `refinement_first`. Explicit Cholesky with
a supplied basis still accepts explicit adaptive ordering. Low-level solver
and linear-system defaults remain `refinement_first`. Fitted diagnostics
record requested, initial and effective ordering; spectral recovery uses
`refinement_first`. Healthy ordinary checks gain no skipped correction work.

If checked spectral inverse-action refinement exhausts its allowance or
stagnates at the same stored
float64 coefficients, a final bounded correction can examine adjacent floats.
It proposes one `nextafter` step for one coefficient and adjusts the free
intercept to reduce the original-equation residual. Candidate selection is
deterministic; each accepted private move must strictly improve a freshly
computed compensated residual. The returned coefficients and intercept must
still pass the existing residual, coefficient-sum and RKHS accuracy checks.
Unsuccessful private changes are discarded. The kernel, shift, target
constraint, full eigenbasis and MM update count are unchanged.

This correction is skipped when the ordinary solve passes its checks. It shares
at most eight accepted coefficient moves across all five inverse representations
of one solve action.
It also shares a budget of 1,048,576 normalized dense-work units. Each charged
dense action costs `n * n` units; these units bound the number of scans and
checks, not actual floating-point operations, bytes accessed or wall-clock time.
Problems too large for the remaining budget skip this optional correction.
Additional arrays have length `n`; no dense matrix copy is created. Exhausting
these limits does not authorize an inaccurate result. Diagnostics include
`readout_refinement_attempts`, `readout_refinement_acceptances`,
`readout_refinement_seconds`, `readout_last_attempt`, `readout_action_moves`,
and `readout_action_matrix_entry_work` when the correction is attempted. The
two `readout_action_*` fields describe the last solve action that attempted
this correction; later healthy actions leave those diagnostic values intact.

Once the original equations and coefficient-sum constraint pass, an overly
conservative RKHS error estimate can be reconsidered using one auxiliary inverse
action through the existing factor or basis. The new estimate combines an upper
estimate of the correction's RKHS norm with a bound on its remaining residual,
carrying uncertainties from both residual evaluations. The auxiliary correction
is not added to the fitted state, and this assessment does not call itself
recursively or construct a new factorization. The original tolerances still
apply. Diagnostics record compensated checks, scalar corrections, auxiliary
actions, acceptance counts and their costs when those paths execute.

## Bounded exact-function recovery

The DWD model is an RKHS function with a free intercept. For a singular PSD
kernel, several coefficient vectors can represent that same function. An
additional coefficient-sum gauge used by an ordinary inverse implementation
can therefore be more restrictive than the mathematical MM update. Relaxing a
failed numerical check without establishing the intended function is not an
acceptable recovery.

After an ordinary MM proposal fails numerically, the package can certify the
complete stored float64 kernel exactly as `K = C B_inverse C.T`. C comprises
selected original kernel columns; B is their positive-definite principal
submatrix. Complete exact identities and a checked inverse establish the factor;
no generating feature matrix, numerical rank cutoff, discarded positive direction
or partial factor is used. The actual input still passes the public kernel
validation policy before this recovery becomes eligible.

The same current MM right-hand side is solved in those coordinates using a small
validated full eigensystem, with at most one rounded-exact alternative. Exact
arithmetic checks the exported coefficients' RKHS-function error, full decision
error, and free-intercept residual against the same intended MM function. The
stronger redundant coefficient-sum gauge is deliberately not imposed in this
representation. The original objective, loss exponent, regularization and free
intercept are unchanged. This recovery does not substitute another public
optimizer. Initial preparation errors and arbitrary observer errors do not
become recovery triggers.

With the default `exact_recovery='standard'`, the caps are 65,536 matrix entries,
rank 16, 4,096-bit intermediate arithmetic and one million counted exact
operations per preparation/step.

These caps apply simultaneously, not as a promise of successful recovery at or
below the rank cap. A matrix within both the entry and rank limits can still
exhaust the bit or operation budget during kernel certification, MM preparation
or a later recovery step. Rank is not row count: a low-rank matrix may have
more rows than the rank cap, subject to the entry and arithmetic limits.
Practical capacity depends on the stored entries and intermediate arithmetic.

The entry cap is checked before finite scans or rational allocations. Operations
and rational intermediate sizes are bounded conservatively; these counters are
not a formal wall-clock or total-process-memory bound. Incomplete certification,
non-PSD exact data or budget exhaustion makes recovery inapplicable, never a
partial-rank model. Every positive direction must fit the complete certificate.

Exact preparation runs only after an ordinary MM proposal fails numerically
and is reused only within that fit. Returned estimators retain normal float64
coefficients and diagnostic summaries, not rational matrices. Once active, the recovery can
serve subsequent MM actions; validation restoration still returns the earlier
model and its representation metadata when appropriate. Successful completion
adds one observed state per accepted MM update, with no hidden increase to the
iteration budget. Optional momentum/restarts retain their ordinary transaction
rules. The exact MM recovery is not used by L-BFGS.

Thirty public fits on previously saved extreme synthetic kernels returned 21
checked models in the preceding range-recovery candidate: all 18 exact rank-five
cases completed, while nine nearly constant RBF fits still rejected at the
rank-16 limit. Those were the first observed refusals, not complete diagnoses
of their stored kernels. The extended investigation below distinguishes a
capacity limitation from exact indefiniteness. Those cases were not MNIST.
See the release-specific [validation](../VALIDATION.md) for source/wheel checks
and their scope.

### Optional extended exact recovery

`KernGDWD(exact_recovery='extended')` and
`KernGDWDCV(exact_recovery='extended')` select a larger bounded capacity for
corrected Schur fits. The default remains `'standard'`.

| Limit | Standard | Extended |
|---|---:|---:|
| Stored matrix entries | 65,536 | 65,536 |
| Complete exact rank | 16 | 32 |
| Rational/intermediate arithmetic bits | 4,096 | 8,192 |
| Counted exact operations per stage | 1,000,000 | 1,000,000 |

The setting is passed to both the complete factor certificate and the MM
action. Certification, action preparation and each subsequent step have their
own operation budgets. These limits bound arithmetic work, not total fit time
or process memory. The entry cap allows square matrices of at most 256 rows;
the option cannot extend this recovery to a full MNIST kernel matrix.

Exact arithmetic remains lazy: a healthy MM fit takes its ordinary path.
The setting does not enable recovery in L-BFGS, which does not use this MM
fallback.
The capacity option preserves the loss, regularization, free intercept, complete represented kernel and all
acceptance thresholds. It never authorizes rank truncation, added diagonal
ridge, an approximate certificate or a repaired indefinite kernel. Failed
certification still returns no model through this recovery route.

In the nine historical nearly constant RBF cases, the three 30-observation
fits have an exactly positive-definite rank-30 stored matrix. Raising rank
alone passes factor certification but still exhausts MM preparation's bit
budget. The extended setting completes five fixed updates in each case and
also survives 100 updates under default objective stopping. Those longer
runs reach `max_iter` with convergence and stopping-criterion flags false;
returning a checked model is not evidence of convergence.

The six cases with 75 or 120 observations still refuse at rank 32. Separate
exact quadratic witnesses establish that those stored float64 matrices are
indefinite, so more resources cannot produce an exact PSD certificate for
them unchanged. This concerns rounded matrices, not the mathematical Gaussian
kernel or every other solver route. Neither profile silently alters them.

Three fresh-process observations measured about 0.066 seconds for default
refusal and 2.744 seconds for the extended five-update fit on the 30-row case.
These are different outcomes, not a speed comparison. Other known trial
workers had exited, but systemwide background activity and peak memory were
not measured. Independent checks verified the recovered equations and original
DWD objectives; see [validation scope](../VALIDATION.md). The capacity is
optional because its benefit is conditional and more work can still end in
refusal.

## Explicit full-inverse action

`linear_action='factor'` remains the default. The explicit `inverse` option
constructs a dense inverse action of the same shifted Cholesky system for
optimized Schur MM. It changes candidate arithmetic, not the objective,
constraint, acceptance tolerances, refinement, or recovery budgets. It supports
ordinary and restart MM. It can amortize setup over many updates, but measured
actual-stopping fits can be slower; it is not an automatic large-data setting.

The workspace belongs to one solve and is discarded on return or before
centered/anchor recovery. A numerical construction refusal retains the factor
action; allocation errors propagate. Eligible automatic spectral retry uses
the existing factor/spectral action and records the original inverse request.
`requested_linear_action_` and `linear_action_` distinguish request from final
active action. Compact ensemble base diagnostics preserve each learner's route.

For n rows, float64 B alone adds `8*n*n` bytes: 512 MiB at 8192 or 2 GiB at 16384.
The qualified `(B+B.T)*.5` construction can transiently hold three B-sized
arrays (1.5 GiB or 6 GiB respectively), plus vectors and native workspace. These
are incremental planning terms, not total fit peaks or automatic admission.
K, the factor, features, and parallel learners consume additional memory.
There is no persistent inverse cache and no fitted inverse to pickle.

Use optimized Schur `backend='auto'` without a supplied eigensystem or explicit
`backend='cholesky'`. Explicit Cholesky still validates any supplied eigenpairs.
Explicit spectral, L-BFGS and auto-with-eigenbasis routes
refuse an inverse request. Kernel `implicit_P` has been removed; use
`linear_action` to select the supported action. Changing the parameter affects the next fit, not saved
predictions. CV forwards the request and may reuse K, never the inverse.

## Numerical scope

Private tolerances and estimated solve-error bounds depend on floating-point
scale and dimension. They are measured consistency estimates, not rigorous
interval certificates or a guarantee for every matrix. A globally scaled
eigenpair check alone does not certify inverse accuracy in tiny-eigenvalue
directions; the original-system residual checks and refinement are still needed. Final objective,
score-consistency, and stationarity checks remain separate safeguards. A full
eigenbasis is still expensive in time and dense memory. Choosing a backend does
not start parameter search. Record the package version, source identity, effective
backend, and numerical diagnostics when reproducing a search.

Sample weights remain explicitly unsupported. Kernel `implementation`, `implicit_P` and
`solver_mode` are no longer accepted parameters.
The existing low-level `solve_gen_kern_dwd` keeps its four-item MM return tuple
`(alpha, offset, objective_history, C)` with corrected algebra and zero
initialization. The estimator exposes the backend, initialization and stopping options
described above. Historical reproduction uses the matching archived release.

### Missing numerical settings on loaded estimators

For objects missing requested `affine_computation`, `acceleration` or
`residual_check_order` fields, parameter inspection, cloning and new fits use
the current defaults: `'joint'`, `'auto'` and `'auto'`. This also covers objects
already in memory; no reload or deserialization rewrite is required. Backend
and supplied-eigenbasis restrictions still govern automatic policy resolution.

Explicit stored choices are preserved, including `'standard'`, `None` and
`'refinement_first'`. Values already materialized by an earlier clone cannot be
distinguished from intentional choices and are not guessed or rewritten.
An explicit removed affine `'auto'` value still requires caller migration to
`'joint'` before a new fit; it is not a missing field.

Existing prediction uses fitted scoring metadata and the stored fitted model.
Missing fitted affine metadata still denotes standard scoring. This preserves
the interpretation of existing coefficients; it does not endorse their original
fitting algebra. A new fit uses current solver mathematics and requested
policies. It may produce different trajectories, scores, selection and costs.

Prepared state and caches must match the newly effective requests and their
existing source/geometry identity. Incompatible state is rebuilt or refused;
historical cache or checkpoint provenance is never rewritten to force reuse.
These rules cover the documented missing settings, not arbitrary damaged or
cross-version serialized objects. Incompatible states may still require a new
estimator fitted from the original training data.


### KernelScaler copy modes

`dwd.kernel_utils.KernelScaler` is a separate dense, square-kernel transformer.
It retains its fitted diagonal normalization and does not automatically run
inside DWD fitting or prediction. It is not a rectangular query-kernel scaler.

- Keep `transform(K, copy=True)` (the default) when the original array must be
  preserved. Computation uses a copy, so an overflow refusal leaves the supplied
  array unchanged. This is also the appropriate mode for read-only inputs or
  views whose logical elements overlap within their own memory storage.
- Use `copy=False` when in-place modification of compatible writable storage
  is acceptable. Validation may still allocate, for example when converting
  an integer array to a supported floating dtype. Ordinary views can share
  their modifications with a parent array or other aliases.

In-place transformation has no rollback guarantee. The scaler applies its
column and row factors before checking that the transformed values are finite.
If that check raises `FloatingPointError`, the input and its aliases may already
be modified or contain infinities. Do not retry using that modified array as an
original kernel: retain an independent original or use the default copying mode.
Input and factor-validation failures that occur before scaling do not authorize
assuming that every later failure also leaves the input unchanged.

A refused transform does not itself invalidate the scaler's fitted diagonal.
The same fitted scaler can be used with valid original data afterward. This
copy-mode clarification changes no scaling formula, numerical refusal, default,
or executable transformation path.
