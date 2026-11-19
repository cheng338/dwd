# Fitting arguments and conic persistence

DWD 2.0.0 removes the unused training-weight argument and makes the
fitting controls below keyword-only where indicated. Use Ensemble DWD 0.4.0
with this release. Earlier companion versions can forward removed DWD keywords.

## Fitting arguments and metadata

Training sample weights are not implemented. The unsupported `sample_weight`
parameter has been removed from `GenDWD`, `GenDWDCV`, `KernGDWD`, `KernGDWDCV`,
SOCP `DWD` and conic `SVM` fitting, and from their four standalone solvers.
Earlier versions accepted `None` and rejected actual weights. Direct calls
now reject every supplied value with `TypeError`, including `None`, instead
of advertising a weighted-fit interface that the objectives do not implement.
Other classifiers that already lacked this argument remain unweighted.

Use `model.fit(X, y)`, not `model.fit(X, y, sample_weight=None)` or
`model.fit(X, y, None)`. Genuine controls are retained:

| Entry point | Positional arguments retained | Other controls |
|---|---|---|
| `GenDWD.fit` | `X, y` | `P0_eig`, `beta_init`, `offset_init` are keyword-only. |
| `KernGDWD.fit` | `X, y` | `K`, `K_eig`, `alpha_init`, `offset_init`, `validation_data` are keyword-only. |
| `GenDWDCV.fit`, `KernGDWDCV.fit`, conic `DWD.fit`, `SVM.fit` | `X, y` | No training-weight argument. |
| `solve_gen_dwd` | `X, y, lambd, q` | All subsequent controls, including initial coefficients, are keyword-only. |
| `solve_gen_kern_dwd` | `K, y, lambd, q, alpha_init, offset_init` | Subsequent controls are keyword-only. |
| `solve_dwd_socp`, `solve_svm` | `X, y, C` | `solver_kws` is keyword-only. |

Remove the old weight placeholder and name trailing controls explicitly. For
example, use `solve_gen_dwd(X, y, lambd, q, beta_init=beta, offset_init=bias)`
or `solve_svm(X, y, C, solver_kws=options)`. The boundaries prevent obsolete
positional calls from silently binding to a different option. Constructor
parameters, numerical defaults, objectives and valid unweighted fits are
unchanged by this API removal.

`model.score(X, y, sample_weight=weights)` remains weighted accuracy. It
weights the evaluation, not model fitting. An outer search can use supported
scoring metadata, but must not route training weights into these fits.
AdaBoost requires genuine weighted fitting and is unsupported with these
base classifiers. This API removal does not add a weighted objective.

Fresh scikit-learn metadata declarations no longer advertise fit weights.
Existing saved `_metadata_request` state can still contain requests with
`True`, `False`, `None` or an alias. Loading does not rewrite that state, and
`sklearn.base.clone` copies it too. A clone is therefore not a migration.
For the supplied estimators, construct a fresh object and refit:

```python
fresh = type(old_model)(**old_model.get_params(deep=False))
fresh.fit(X_train, y_train)
```

When metadata routing is enabled, reapply only requests supported by the fresh
estimator, such as legitimate preparation arguments or score weights/aliases.
Do not copy or blanket-delete private metadata state: that can retain obsolete
fit requests or remove legitimate requests. The construction above preserves
constructor settings, not fitted state or routing configuration. It is not a
promise of compatibility with every historical pickle or custom subclass.
Direct invalid refits follow the existing
[failed-fit cleanup contract](failed_refit_state.md), clearing learned state
before the error is propagated.

## Conic problems and persistence

The optional `dwd.socp_dwd.DWD` and `dwd.svm.SVM` retain a real CVXPY problem
in `problem_`. Their standalone solvers return the same kind of problem.
New problems use a package-owned CVXPY subclass to preserve fitted state for
the validated pickle, joblib and cloudpickle routes. No installed dependency
is edited and no serializer is invoked during fitting to probe support.

Saving preserves the mathematical problem, parameter and variable graph,
primal/dual values, status, objective, and solver statistics. Model, problem,
variable, constraint, parameter and diagnostic-object aliases are preserved.
Array values are preserved; array identity follows the serializer. Saving does
not clear or modify the live problem's solver cache.
In the serialized copy, known data-only cache entries, including validated SCS
warm-start arrays, are retained. Entries containing native solver/factorization
handles or other opaque state are omitted and rebuilt by a later solve.
Loading does not itself solve the problem. A subsequent solve can consequently
have different startup cost, iteration count or roundoff from a live warm solve;
the saved diagnostics describe the original fit until it is solved again.

Some native diagnostic objects cannot be serialized. Recognized data-only
objects in `problem_.solver_stats.extra_stats` become attribute-readable Python
records after solving, with their public data fields and `_dwd_source_type`
provenance retained. Their concrete Python type can therefore differ from the
native solver's record type even before saving. Reserved native interoperability
method references are stored as explicit descriptions, not callable methods.
Read diagnostic attributes instead of requiring a particular extension type or
trying to call a live solver through saved statistics. These internal record
classes are not a new public solver API.

Outside the documented solver-cache policy, unknown custom diagnostics,
callbacks, solver objects and user-added fields are not silently discarded or
repaired. They can still make serialization fail. The support is for the
checked object graphs and solver routes, not
arbitrary pickle compatibility. Only load serialized models from trusted
sources, with compatible package and dependency versions. The implementation
uses CVXPY's private cache/statistics layout; changes to that layout require
focused validation. The current checks used CVXPY 1.9.3 on Python 3.12 on
Windows/AOCL and Linux/MKL, including CLARABEL/SCS for DWD and
CLARABEL/SCS/OSQP/SCIPY/HIGHS for SVM, dense/CSR inputs, loaded re-solves and
two-worker transport. This is not a guarantee for all future solver versions.

### Caller-owned solver options

Before each conic solve, exact builtin `dict`, `list` and `tuple` containers
in the solve arguments are copied, preserving container aliases and cycles.
This prevents solver adapters from modifying the caller's nested options,
including SciPy option dictionaries. Arrays, custom solver instances,
callbacks, container subclasses and opaque objects retain their identity.
Mutations inside those retained objects are not isolated. Solver selection,
option meanings, tolerances and defaults remain controlled by CVXPY and the
chosen solver; unsupported options still raise.

## SVM expression and numerical limits

The conic `SVM` minimizes mean hinge loss plus `C * ||beta||_1`, with a free,
unpenalized intercept. It is distinct from sklearn's L2-regularized SVC and
from `ensemble_dwd.EnsembleKernelSVM`, which uses SVC.

Its signed margin now multiplies the constant feature rows by their binary
labels before the matrix product, then adds the signed intercept. In exact
arithmetic this equals multiplying the complete affine score by the labels.
It retains the coefficient shape, loss, penalty, free intercept, solver
choices and status checks; floating-point operation order changes.

The former dense expression exposed a CVXPY 1.9.3 bound-propagation problem:
invalid intermediate bounds became false fixed hinge bounds on the
HIGHS/SCIPY routes. A reproduced case returned objective `1.0` instead of the
independent linear-program value approximately `0.4740248934`. The revised
expression recovered the latter value on both tested hosts. Dense/CSR and
ordinary, separable, duplicate, zero-feature and scale-extreme cases were
checked against independent linear programs. This evidence identifies an
expression/canonicalization interaction, not an AOCL/MKL defect or an effect
attributed to a numerical-library upgrade.

A finite iteration budget can still fail. In the tested large-scale OSQP
case, the native solver reached its explicit 100,000-iteration limit before
meeting its requested stopping conditions. This occurred with both expressions
and remains an error with fitted-state cleanup, not an accepted numerical
result. Deliberate equivalent variable scaling resolved the diagnostic case;
no automatic rescaling, relaxed tolerance or larger default budget is added.
Inspect solver status and diagnostics and select suitable supported settings
or another solver when necessary. Arbitrary feature scaling can change the
regularized model and should not be treated as a neutral retry.

## Measured costs

Paired before/after measurements used synthetic 128-by-12 inputs, dense and
CSR, a requested single native thread, and four ABBA blocks per case. The following are mean
total-fit differences for the combined correctness, persistence and option
protection changes, not the cost of diagnostic conversion alone:

| Model / solver / input | Windows/AOCL change | Linux/MKL change |
|---|---:|---:|
| DWD / CLARABEL / dense | +0.048 ms (+0.85%) | +1.148 ms (+7.08%) |
| DWD / CLARABEL / CSR | -0.001 ms (-0.01%) | -0.109 ms (-0.74%) |
| DWD / SCS / dense | +0.037 ms (+0.51%) | +0.268 ms (+1.39%) |
| DWD / SCS / CSR | -0.046 ms (-0.62%) | +0.237 ms (+1.21%) |
| SVM / CLARABEL / dense | +0.146 ms (+3.31%) | +0.350 ms (+3.21%) |
| SVM / CLARABEL / CSR | +0.319 ms (+7.08%) | +0.825 ms (+7.46%) |
| SVM / OSQP / dense | +0.157 ms (+0.67%) | +4.845 ms (+8.94%) |
| SVM / OSQP / CSR | +0.311 ms (+1.32%) | +0.892 ms (+1.62%) |

These small panels are descriptive, not precision estimates, a uniform
no-slowdown claim or a comparison between AOCL and MKL. The changes address
correctness and state preservation; they are not adopted for memory savings.
Larger workloads and other solvers can have different costs.

Separate expression comparisons at 128-by-12 and 1024-by-24 showed that an
explicit epigraph formulation can be faster. It adds auxiliary variables and
constraints to the exposed problem. The signed expression is the smaller
validated repair preserving that problem structure, not a claim of the fastest
formulation. Timing outliers occurred in both signed and epigraph panels;
the four-block results do not establish a universal ranking. Timings of the
old incorrect HIGHS/SCIPY solutions were not used as a performance baseline.

On the small persistence panel, absolute pickle save/load times were roughly
0.4-0.5 ms on Windows and 0.8-1.1 ms on Linux; joblib save/load times were
roughly 6.2-6.6/3.1-3.5 ms and 10.5-11.1/5.4-6.0 ms, respectively. These are
absolute round-trip components, not paired serialization overhead estimates.
Some old solver routes could already serialize; others failed on native state.
