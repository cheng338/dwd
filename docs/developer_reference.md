# Corrected reference machinery for maintainers

Use the private reference path when investigating an MM update, a numerical
regression, or agreement between coefficient and score-state formulations.
Use the public estimators for application code, CV, cache-contract tests and
performance recommendations. The private path is not a supported replacement
for the removed public `implementation` parameter.

The reference computes the **corrected** full-kernel DWD objective with an
unregularized intercept. It is not the retired, incorrect `solver_mode='legacy'`
algebra. It shares kernel preparation, checked linear-system machinery and
reporting with the optimized implementation, so agreement between them is not
an independent mathematical certificate. See [independent checks](#independent-checks)
below and the [public option guide](choosing_options.md) for application choices.

## What remains available

| Entry | Intended use and boundary |
| --- | --- |
| `dwd._kernel_solver.solve_kernel(..., implementation='reference')` | Private checked solve for an explicitly supplied kernel and signed labels; returns a result dictionary. |
| `tests/_kernel_reference.py:PrivateReferenceKernGDWD` | Source-tree test adapter for reference fitting plus estimator-shaped prediction/callback checks. It is not installed as a public estimator API. |
| `tests/_kernel_reference.py:kernel_route` | Test factory selecting the real public `KernGDWD` or the private adapter. Its selector belongs to the test helper, not the estimator. |
| `dwd._reference_mm.reference_update` | One corrected coefficient-coordinate update, given an already validated constrained-system object. It does not perform a complete fit. |
| `get_K_eig(K)` in `dwd.gen_kern_dwd` | Checked eigensystem helper returning `(U, eigenvalues)` in descending eigenvalue order. It has no `reference` argument. |

These private entry points can change with the implementation. In particular,
do not add them to a public parameter grid, dependency contract or serialized
application configuration. Ensemble DWD has no separate reference selector;
its learners are public optimized DWD estimators.

`solver_mode`, kernel-only `implicit_P`, and the ignored
`get_K_eig(reference=...)` selector were removed, not moved behind a new private
switch. Linear `GenDWD`/`GenDWDCV` still support their real `implicit_P` choices.
The remaining standalone kernel step helpers are described separately below;
they do not restore an estimator-level explicit-P route. See the
[migration boundaries](kernel_dwd.md#removed-public-solver-controls).

## Calling the private checked solver

Run the following from a DWD source checkout using an isolated validation
environment with its dependencies available. Check the imported source before
interpreting results:

```console
python -c "import dwd; print(dwd.__file__)"
```

The path should identify the intended checkout. This small example compares
three updates with the same kernel, initial state and numerical policies:

```python
import numpy as np
from dwd._kernel_solver import solve_kernel

K = np.array([[1.0, 0.25], [0.25, 1.0]])
y = np.array([-1.0, 1.0])
alpha0 = np.array([-0.125, 0.125])
settings = dict(
    lambd=0.1, q=1.0, backend='spectral',
    alpha_init=alpha0, offset_init=0.125,
    stopping='fixed', max_iter=3, tol=1e-6, obj_tol=1e-5,
    acceleration=None, residual_check_order='refinement_first',
    affine_computation='joint', exact_recovery='standard',
    linear_action='factor',
)
reference = solve_kernel(K, y, implementation='reference', **settings)
optimized = solve_kernel(K, y, implementation='optimized', **settings)

for result in (reference, optimized):
    print(result['diagnostics']['implementation'], result['n_iter'],
          result['termination_reason'], result['converged'],
          result['rkhs_gradient_norm'])
print('Coefficient difference:',
      np.max(np.abs(reference['alpha'] - optimized['alpha'])))
```

Three fixed updates demonstrate invocation, not convergence or a performance
comparison. Retain both outcomes, including refusals and missed targets. Choose
acceptance bounds for the actual equations and represented inputs; do not
convert a printed small coefficient difference into a universal tolerance.

The solver requires a finite square kernel, labels of shape `(n,)` containing
both `-1` and `+1`, and positive finite `lambd` and `q`. It checks matrix
symmetry/PSD and the initial state under its numerical contract. Supplied
`K_eig=(U, eigenvalues)` must describe that exact represented kernel and passes
the existing eigenpair checks. Leave `psd_known=False`: that private assertion
is reserved for trusted estimator construction, not a way to bypass validation
of an external matrix. Leave `_linear_features` absent on reference calls.

For the reference route:

- `backend='auto'` resolves to spectral; explicit `'spectral'` is supported.
  `'cholesky'` and `'lbfgs'` are rejected.
- `acceleration='auto'` resolves to ordinary MM (`None`). Explicit `'restart'`
  is rejected. Set `None` explicitly in a matched-update comparison.
- `residual_check_order='auto'` resolves to `'refinement_first'`; explicit
  `'adaptive'` is rejected. The direct private function itself defaults to
  `'refinement_first'`.
- `linear_action='inverse'` is rejected; retain `'factor'`.
- The direct function defaults to standard affine scoring, whereas current
  public estimators default to joint. Specify the same affine policy when
  comparing them. These distinct low-level defaults are not public overrides.
- Omitted `alpha_init` and `offset_init` mean zero in the direct function. It
  has no estimator-style `initialization` or `random_state` parameter.

If testing validation stopping, pass `validation=(query_by_training_K,
signed_validation_y)` with `stopping='validation'`. The private solver does not
construct a validation split. For any stopping policy, inspect the returned
state, termination reason and stationarity diagnostics separately.

### State used by the spectral residual checks

`SpectralLinearSystem` initializes its own state because the
`KernelLinearSystem` constructor immediately attempts a Cholesky factorization.
The coefficient reference solver and accelerated spectral MM both use this
class. Its inherited residual checks use the original equation
`(K + shift I)x + s 1 = rhs`. `solve_constrained` checks the requested
coefficient sum (zero by default). `refine_candidate` checks `sum(x) = 0`.
Shared refinement routines request corrections through the spectral
`_candidate` method and record diagnostics in `info`.

Keep `K`, `shift`, `n` and the column mean `K.mean(axis=0)` tied to the
original system throughout spectral recovery. The ordinary check builds
`_ordinary_bounds` lazily from `K` and `mean`. The compensated assessment
also uses `mean` to estimate RKHS error. Recovery can replace the eigenbasis
used to compute a correction while those acceptance checks continue to use
the original kernel. [Spectral solver](../dwd/_spectral_linear_system.py) ·
[Shared residual checks](../dwd/_kernel_linear_system.py)

## Using the source-tree test adapter

For feature-based reference tests needing `decision_function`, use the adapter
already used by the numerical regression suite. From the DWD checkout root:

```python
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path('tests').resolve()))
from _kernel_reference import PrivateReferenceKernGDWD

X = np.array([[-1.0], [1.0]])
y = np.array([-1, 1])
model = PrivateReferenceKernGDWD(
    kernel='linear', lambd=0.1, q=1.0,
    backend='spectral', initialization='zero', random_state=7,
    acceleration=None, residual_check_order='refinement_first',
    affine_computation='joint', stopping='fixed', max_iter=3,
).fit(X, y)
print(model.diagnostics_['implementation'], model.decision_function(X))
```

The adapter contains an exact public `KernGDWD` object for preparation and
readout; it does not subclass or modify the production class. This matters for
qualified named-kernel construction. Its `cv_init` deliberately raises
`NotImplementedError`: it has no public CV/cache contract. Public constructor,
grid, cache and exact-estimator dispatch tests must use the real estimator.

Unlike the current public zero default, the adapter deliberately defaults to
`initialization='random'` to preserve the numerical tests' intended starts.
That draws normalized Gaussian coefficients and uses a zero offset unless an
offset is supplied. An integer seed makes the request reproducible within the
tested runtime; `random_state=None` can consume the global NumPy RNG. Supplied
coefficients take precedence over the initialization setting. Prefer explicit
initial arrays for a controlled update comparison. The removed `'auto'`
initialization alias is not available through the adapter.

The adapter otherwise inherits current constructor defaults, including joint
affine scoring. Auto acceleration and residual ordering resolve for the
reference route as described above. Do not compare omitted adapter settings
against omitted public settings and attribute every difference to the update
formula. In particular, the public defaults now select zero initialization,
normally Cholesky, restart acceleration and eligible adaptive residual checks.

## Independent checks

Run the focused existing tests from the checkout root:

```console
python -m pytest tests/test_reference_mm.py tests/test_two_implementations.py tests/test_public_reference_removal.py -ra
```

`test_reference_mm.py` independently forms the augmented MM equations and a
direct next-state system. It checks general `q`, nonzero starts, singular and
zero kernels, the coefficient-sum constraint and free-intercept sign, as well
as short trajectories. In singular cases, matching equations and represented
predictions is more informative than demanding a unique coefficient vector.
These are finite floating-point checks with declared tolerances, not proofs
over every input.

`test_two_implementations.py` checks the private reference route, initialization
and callback behavior alongside real public API/backend tests.
`test_public_reference_removal.py` ensures removed public selectors remain
rejected, stale selector attributes do not change a new public fit, and the
private solver remains available for equation comparisons. A private-reference
pass must not substitute for these public boundary checks.

For independent inspection of a small stored model, the explicit
`literal_objective_diagnostics` API evaluates the represented kernel rather
than the optimizer's score recurrence. This example performs no fit:

```python
import numpy as np
from dwd.literal_diagnostics import DiagnosticRefusal, literal_objective_diagnostics

K = np.eye(2)
y = np.array([-1.0, 1.0])
alpha = np.array([-0.125, 0.125])
try:
    report = literal_objective_diagnostics(K, y, alpha, 0.0, lambd=0.1, q=1.0)
except DiagnosticRefusal as error:
    print('Diagnostic unavailable:', error)
else:
    print('Objective enclosure:', report['objective_interval'])
    print('Bias-gradient enclosure:', report['bias_gradient_interval'])
    print('Coefficient-gradient enclosures:', report['coefficient_gradient_intervals'])
    print('Dual lower bound:', report['dual_lower_bound'])
    print('Objective-gap upper bound:', report['objective_gap_upper'])
```

The current maximum is 16 rows, with bounded arithmetic operations, bit lengths
and Decimal precision. Its coefficient-gradient intervals are not the solver's
RKHS-gradient norm; do not compare them directly to `tol` as the same statistic.
A returned interval is a bounded diagnostic under its stated arithmetic
assumptions, not a generalization certificate. A refusal provides no missing
certificate and does not mean the fitted classifier was changed. See the
[literal-operator scope](kernel_dwd.md#bounded-literal-operator-diagnostics).

## Bare update helpers are not fit alternatives

Prefer the checked solver above for an end-to-end regression. The following
helpers do not provide its validation, convergence, lifecycle or reporting
contract:

- `_reference_mm.reference_update` expects an already checked `K`, state and
  mean-loss derivative `z`. With `t = n*q/(q+1)**2`, the supplied system must
  solve `(K + 2*lambd*t*I) @ x + s*ones = rhs` with the requested coefficient
  sum. The required target is `sum(alpha)/t`, which can be nonzero; the update
  subtracts `t*x` from alpha and `t*s` from the free offset. See its source
  docstring and the independent tests rather than replacing this target by zero.
- `gen_kern_dwd.get_step_implicit_P` expects mutually consistent `K`, eigenpairs
  `U, Lam`, and `v = (K + 2*lambd*t*I)^(-1) @ ones`. Optional `ULP` must equal
  `U / (Lam + 2*lambd*t)` columnwise, and optional `K_alpha` must be the current
  kernel action. The retained `pi` and `g` arguments are currently unused;
  they are not alternative policies. The returned `(alpha_step, offset_step)`
  are subtraction steps, not a fitted model.
- `gen_kern_dwd.get_step_explicit_P` expects a compatible augmented-system
  inverse `P_inv` for the same kernel, shift and free-intercept MM equations.
  It does not construct, certify or recover that inverse. It likewise returns
  subtraction steps. Its existence does not make kernel `implicit_P=False`
  available on an estimator.

All these callers must preserve data order, dimensions, loss exponent, penalty,
initial state and represented kernel. Do not use a helper to bypass a checked
solver refusal. The retired algebra survives only as historical test material,
not an available new-fit mode.

For developer arithmetic profiling and compiled/portable build selection, use
the separate [compiled residual guide](compiled_residual.md). Those controls do
not select the reference implementation or restore removed estimator options.
