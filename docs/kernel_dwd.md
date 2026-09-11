# Kernel DWD: API and numerical controls

This guide describes the kernel estimator, its objective, parameter conventions,
stopping rules, and numerical diagnostics. For release-specific validation, see
[VALIDATION.md](../VALIDATION.md). Details of the optional arithmetic accelerator
are in [compiled residual arithmetic](compiled_residual.md).

`dwd.gen_kern_dwd.KernGDWD` fits one binary classifier. It does not search for
parameters or create hidden validation splits. `KernGDWDCV` is an explicitly
selected convenience wrapper; sklearn GridSearchCV can also wrap the plain
estimator or a complete external training procedure.

Generic `run_cv`, `GenDWDCV` and `KernGDWDCV` require finite real numeric scalar
train/test scorer outputs. Invalid outputs raise before choosing/refitting a
winner rather than silently excluding a candidate. Numeric Python/NumPy scalars
and zero-dimensional numeric arrays are supported; object-dtype scorer results
are outside this contract. Valid-score aggregation and first-tie ordering are unchanged.

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
names have not changed. q=1 gives DWD. When comparing the same objective with the
dissertation's convention using `(lambda_dissertation / 2) * norm_squared`,
`lambd = lambda_dissertation / 2`. This mathematical conversion does not identify
the convention actually used by an old script or authorize silently reinterpreting
its stored numerical parameters.

The optional SOCP classifier uses a different constrained linear formulation and
parameter `C`. Passing a Gram matrix as ordinary linear features changes the
regularization geometry; it is not an equivalent kernel SOCP conversion.

## Implementations, backends, and initialization

Both implementations introduced in 1.2.0 remain available. The default
`implementation='optimized'` allows changes to the numerical computation.
`implementation='reference'` repairs the Slicersalt implementation while retaining
its eigen-decomposition, coefficient-MM route, normalized Gaussian initialization,
and default absolute-objective stopping rule. Correcting its loss, MM algebra,
numerical preparation, and API does not promise the outputs of buggy upstream
updates. The reference never substitutes a Cholesky solve.

| Parameter | Behavior |
|---|---|
| `solver_mode='schur'` | Default corrected update algebra, with a free intercept. |
| `implementation='reference'` | Corrected coefficient MM using a checked full eigenbasis. Accepts `backend='auto'` or `'spectral'`. |
| `implementation='optimized'` | Default implementation, with the backend choices below. |
| `backend='auto'` | Optimized: Cholesky unless eigenpairs are supplied/cached; eligible internal RBF fits may restart once through the spectral backend after constrained/MM recovery is exhausted. Reference: eigen-decomposition and coefficient MM. |
| `backend='cholesky'` | Optimized only: solve the original constrained shifted-kernel equations with residual checks, bounded refinement, and centered factor recovery. Unknown/precomputed kernels still require PSD validation. |
| `backend='spectral'` | Use the full checked eigensystem; it can be reused across lambda/q values. The reference retains coefficient updates; optimized mode uses spectral coordinates. |
| `backend='lbfgs'` | Optimized only: optimize the same generalized objective in full spectral RKHS coordinates with a free intercept. It requires spectral preparation. |
| `acceleration=None` | Default ordinary MM. |
| `acceleration='restart'` | Optimized corrected MM only: extrapolate decision values and restart momentum using the original objective. Available with Cholesky or a checked full spectral solve; incompatible with reference, legacy and L-BFGS. |
| `initialization='auto'` | Zero for optimized corrected fits; normalized Gaussian coefficients for the repaired reference and explicit legacy mode. |
| `initialization='zero'` | Deterministic zero coefficients and default zero intercept. |
| `initialization='random'` | Random unit coefficients controlled by `random_state`. |

The corrected kernel backends implement generalized q, not a different loss
for each backend. See the release's [validation record](../VALIDATION.md) for the
tested q values and numerical cases. No backend is universally fastest: caching,
kernel size, conditioning, and number of parameter candidates matter.

`fit(..., alpha_init=..., offset_init=...)` overrides the corresponding initial
values. An explicit alpha vector must match the current training rows. Fits do
not warm start from previously fitted coefficients. `initialization='zero'`
ignores `random_state`; external subsampling or CV still needs its own seed.

For a reproducible repaired reference fit, use, for example,
`KernGDWD(implementation='reference', random_state=7)`. Comparing finite-update
trajectories across implementations requires the same explicit initialization,
kernel, regularization, and stopping settings. Their auto initializations differ.

Explicit `solver_mode='legacy'` keeps the historical spectral update, including known algebra
errors. It supports objective/fixed stopping with `backend='auto'` or
`'spectral'`; other backends, validation/optimality stopping, and callbacks are
rejected. For a historical finite-update comparison, specify mode and seed:

```python
historical = KernGDWD(
    solver_mode='legacy', initialization='auto', random_state=7,
    kernel='rbf', kernel_kws={'gamma': .3}, lambd=.1,
    stopping='fixed', max_iter=100,
)
```

Legacy mode is a compatibility path, distinct from `implementation='reference'`;
combining the two is rejected. This preserves the explicit legacy path of the audit fork. The audit fork already
fixed the loss/gradient; legacy therefore does not promise identical behavior to
every unmodified upstream 1.0.5 input.

## Automatic numerical restart

An unaccelerated optimized fit with `solver_mode='schur'`, `backend='auto'`, and
`callback=None` can make one fresh spectral attempt when a constrained MM update
and its existing bounded recovery both fail. The trigger is specific: the
original/centered/anchor constrained solve must be exhausted, followed by failure
of its certified MM-function recovery. An unrelated objective-descent rejection,
callback exception, memory error, or cancellation does not trigger this restart.

This applies only to named RBF kernels built internally by the package's unchanged
construction methods, with valid known-PSD parameters. A reused CV kernel also
requires the recorded construction provenance and unchanged preparation methods.
Caller-supplied `K` or eigenpairs, precomputed and callable kernels, overridden
construction methods, reference/legacy implementations, explicit backend choices,
and momentum acceleration do not gain this restart. A successful ordinary fit
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

## Optional restarted MM

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
raises if its bounded numerical and exact-function recovery cannot produce an
acceptable update. The recovery receives the same full extrapolated right-hand
side, with the corresponding zero prior-offset convention of the proximal step.
Rejected proposals are not passed to callbacks or validation rules.

`n_iter_` counts accepted MM updates. Extra proposals and restarts consume work
without increasing that count; `diagnostics_` records
`acceleration_proposals`, `acceleration_accepted_updates`,
`acceleration_restarts`, and `acceleration_numerical_restarts`. The latter is a
subset of all restarts. Fixed 100 updates therefore does not imply identical
work across ordinary and accelerated MM. A validation-restored model can come
from an earlier accepted iteration, as with ordinary MM.

Acceleration is optional because lower training objective and better numerical
stationarity do not ensure higher validation accuracy. The development panel
found both accuracy ties and declines at matched update budgets. Use external
CV to compare it with ordinary MM, and count any explicit stopping validation
inside each CV training fold. The package does not choose acceleration or change
the original objective-change/cap defaults automatically.

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
- `dual_gap_` and `dual_equality_residual_`: corrected-fit feasible-dual diagnostics.
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
  kernel models; legacy and older serialized models without the field retain
  ordinary evaluation. This is learned state, not a constructor option. It is
  restored with the best validation checkpoint and preserved by serialization;
  cloning does not transfer it. `diagnostics_` can also record
  `compensated_score_reconstructions`,
  `max_adaptive_score_reconstruction_error_before_retry`, and
  `objective_evaluation_retries` when these paths execute.
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
- `C_`: descriptive conversion from the fitted norm. It does not change `lambd`.
  A zero-norm conversion or an overflowed conversion is not a valid positive
  finite C for a separate SOCP fit. `C_conversion_finite_` identifies whether the
  conversion is finite; a finite zero conversion still is not a valid SOCP C.

Legacy kernel fits retain their objective and coefficient-gradient diagnostics
but set checked RKHS/dual quantities to unavailable and do not claim convergence.

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
`K @ alpha`; `decision_values` includes the intercept. Arrays are protected
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

### RBF construction and query consistency

Corrected named RBF fits first use scikit-learn's pairwise kernel. If that internally
constructed self-kernel exceeds the existing symmetry tolerance
`100 * eps * max(1, max(abs(K)))`, the package reconstructs it from the validated
float64 features. It adds the two squared norms before subtracting the dot-product
term, clips negative squared distances to zero, and applies the RBF exponential.
One computed triangle is mirrored and the self-diagonal is set to its analytic
value of one. External or custom kernel matrices retain the strict validator and
are not averaged.

The fitted `kernel_computation_` is `'sklearn'` on the ordinary path or `'norm_sum'`
on the reconstructed path. `kernel_symmetry_correction_` records the triggering
asymmetry. The selected distance formula is also used for prediction and explicit
validation kernels, including dense/CSR and batched queries. Serialization and
compatible CV caches preserve the policy. Self-kernel mirroring and BLAS/batch
rounding mean copied or partitioned queries are not promised bitwise identity;
the existing original-kernel score and objective checks still apply. The reordered
construction removes the demonstrated reversed-addition asymmetry, but does not
eliminate all cancellation for nearly identical large feature vectors.

Healthy scikit-learn RBF results retain their ordinary construction. The fallback
uses numerical-library matrix products and their configured thread limits; it
does not forward `kernel_kws['n_jobs']` into pairwise-kernel workers. Gamma,
regularization, loss, and the free intercept are unchanged. Corrected reference
and optimized fits share this construction policy; automatic solver restart has
the narrower optimized-only eligibility described above.

A matrix-level callable receives `(X_train, X_query, **kernel_kws)` and must
return a training-by-query matrix. `kernel='precomputed'` instead follows the
public sklearn input convention: square training Gram matrix in `fit`, then
query-by-training values in `predict` or `decision_function`. Standard sklearn
pairwise tags and built-in CV slicing preserve this orientation.

Set `prediction_batch_size` to a positive integer to bound the number of query
rows used by each prediction kernel calculation. This supports named, callable,
and precomputed kernels. Callable kernels must compute consistent pairwise values
when their query rows are partitioned. The training Gram matrix is still dense.

Corrected models evaluate dense and CSR query matrices adaptively. The ordinary
product is accepted per row only when a conservative estimate is no larger than
`5e-7 * max(1, abs(ordinary_score))`. The estimate includes positive-dot rounding
and gradual underflow; an overflowing estimate sends that row to expanded
products instead of rejecting a potentially finite cancellation result.
Uncertain rows use high/low products with accurate summation. CSR duplicate
entries count as their separate stored contributions. Temporary product storage
is at most O(128 * n_training), or O(n_training) for one expanded row; there is
no full query-by-training product temporary beyond the query kernel itself.

If the fit requires fully compensated original-kernel reconstruction, all later
query rows use that policy. The ordinary/adaptive product and the fully
compensated alternative face the same fit score and objective checks. An
objective-only mismatch also permits one compensated reevaluation. Validation
uses the same query policy as the corresponding public model, so a sensitive
state cannot be accepted for stopping using one precision policy and exported
under another. No inconsistent callback state is emitted. These are numerical
estimates under the stated floating-point model, not universal interval proofs.

Starting a new fit clears learned state, including the kernel prediction-precision
flag. If fitting or refitting fails, learned state is cleared before the exception
propagates; prediction then raises `NotFittedError`. This contract applies to the
package's linear and kernel classifiers, their CV wrappers, optional conic
classifiers, and `KernMD`. Constructor parameters and validated private
precomputation caches are preserved. See [fitted-state behavior](failed_refit_state.md).

`cv_init(X)` explicitly caches the exact training kernel and, where needed, its
eigensystem. `run_cv` sets candidate kernel parameters before preparing that
candidate's fold cache. Default optimized CV caches only K and keeps Cholesky;
reference CV caches K and its checked eigensystem. Explicit optimized spectral
or L-BFGS CV also prepares eigenpairs for reuse. Unknown kernels may still need
an eigenvalues-only PSD check at fit time. Cloning clears learned state and does
not transfer hidden caches.

Cache equality accounts for kernel parameters (including nested arrays), data
values/order after the solver's dtype conversion, sparse versus dense representation,
implementation, mode, and backend. Corrected fits compare the promoted float64
features they actually use, allowing identical float32 inputs to reuse preparation;
legacy mode retains its original dtype distinction. The RBF construction
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

Corrected feature paths promote to float64 before constructing their Gram
matrices. Casting an already rounded float32 external kernel cannot repair its
lost precision. Matrices must be finite and symmetric, and materially indefinite
kernels are rejected. The initial quadratic-form guard covers both native and
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

Computed full eigensystems must pass numerical validation before use. The native
sequence is EVD, EVR, then EVX. Each candidate is checked against the same unchanged
matrix; a failed basis is released before another attempt. EVD uses divide and
conquer, the algorithm family used by upstream NumPy eigendecomposition. This
driver choice changes numerical preparation, not the kernel objective, full-rank
representation, free intercept, or reference coefficient-MM architecture.
Different valid bases can differ in signs, repeated-eigenvalue coordinates and
roundoff; historical buggy update algebra is not promised bitwise parity across
drivers or numerical libraries.

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

Optimized Cholesky uses a reciprocal-condition estimate as diagnostic information,
then checks each candidate solve against the original constrained MM equations.
If necessary, bounded residual refinement or a centered factorization solves
those same equations. The centered constant direction is an internal solve
device, not extra regularization or a penalty on the fitted intercept. The
reference instead performs inverse actions and bounded refinement through its
validated eigenbasis; it never switches to Cholesky. Both ordinary implementations retain these checks. If an MM proposal still
fails, the bounded exact-function recovery below can be attempted before the fit
raises. A failure in a callback or validation routine is not intercepted as an
MM numerical failure.
Initial coefficients, objective, lambda, and stopping policy are not changed by
these numerical repairs.

When the ordinary residual check fails, the solver first tries one inexpensive
native correction using its existing factor or basis. It accepts that trial only
after fresh ordinary checks of the corrected state and scores. If the trial
fails or produces a numerical error, it is discarded and the original state is
reassessed using compensated float64 products and summation, with allowances for
rounding and gradual underflow. This preflight is bounded to one trial per
factor representation or reference candidate check. The original and centered
Cholesky forms provide at most two native-trial representations; anchor LU
recovery uses compensated residual measurement instead. The reference has at
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
The reference spectral inverse-action route normally permits three correction
steps; a contracting
original-equation residual can extend that private refinement budget to eight.
These are corrections within an MM update, not extra MM iterations or a change
to `max_iter`.

If reference spectral inverse-action refinement exhausts its allowance or
stagnates at the same stored
float64 coefficients, a final bounded correction can examine adjacent floats.
It proposes one `nextafter` step for one coefficient and adjusts the free
intercept to reduce the original-equation residual. Candidate selection is
deterministic; each accepted private move must strictly improve a freshly
computed compensated residual. The returned coefficients and intercept must
still pass the existing residual, coefficient-sum and RKHS accuracy checks.
Unsuccessful private changes are discarded. The kernel, shift, target
constraint, full eigenbasis and MM update count are unchanged.

This correction is lazy on healthy solves and shares at most eight accepted
coefficient moves across all five inverse representations of one solve action.
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

## Bounded exact-function recovery for singular kernels

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
intercept are unchanged, and no L-BFGS or Cholesky substitution is made for the
reference. Initial preparation errors and arbitrary observer errors do not
become recovery triggers.

The private caps are 65,536 matrix entries, rank 16, 4,096-bit intermediate
arithmetic and one million counted exact operations per preparation/step.
The entry cap is checked before finite scans or rational allocations. Operations
and rational intermediate sizes are bounded conservatively; these counters are
not a formal wall-clock or total-process-memory bound. Incomplete certification,
non-PSD exact data or budget exhaustion makes recovery inapplicable, never a
partial-rank model. Every positive direction must fit the complete certificate.

Exact preparation is lazy and reused only within that fit. Ordinary healthy
fits do not create it. Returned estimators retain normal float64 coefficients
and diagnostic summaries, not rational matrices. Once active, the recovery can
serve subsequent MM actions; validation restoration still returns the earlier
model and its representation metadata when appropriate. Successful completion
adds one observed state per accepted MM update, with no hidden increase to the
iteration budget. Optional momentum/restarts retain their ordinary transaction
rules. The exact MM recovery is not used by L-BFGS or explicit legacy updates.

Thirty public fits on previously saved extreme synthetic kernels returned 21
checked models in the preceding range-recovery candidate: all 18 exact rank-five
cases completed, while nine nearly constant RBF fits still rejected. Their
complete exact rank exceeds this bounded recovery's budget. That limitation is
not proof that those DWD functions are unrepresentable or impossible. Those
cases were not MNIST. See the release-specific [validation](../VALIDATION.md)
for the final source/wheel checks and their scope.

## Numerical scope

Private tolerances and estimated solve-error bounds depend on floating-point
scale and dimension. They are measured consistency estimates, not rigorous
interval certificates or a guarantee for every matrix. A globally scaled
eigenpair check alone does not certify inverse accuracy in tiny-eigenvalue
directions; the original-system residual checks and refinement are still needed. Final objective,
score-consistency, and stationarity checks remain separate safeguards. A full
eigenbasis is still expensive in time and dense memory. Choosing a backend does
not start parameter search. Record the package version, implementation, effective
backend, and numerical diagnostics when reproducing a search.

Sample weights and kernel `implicit_P=False` remain explicitly unsupported.
The existing low-level `solve_gen_kern_dwd` keeps its four-item MM return tuple
`(alpha, offset, objective_history, C)`; it now defaults to corrected algebra and
zero initialization. Request explicit legacy mode for the historical random
initialization/update path. New backend and stopping options live on the estimator.
