# Linear DWD resource costs

`GenDWD` and `GenDWDCV` accept sparse features, but their current MM solvers
are not sparse throughout. Sparse input can reduce storage of the feature
matrix and the cost of feature products. It does not remove the dense
feature-square system used to prepare each fit or CV fold.

## Feature-space preparation

Let `n` be the number of training samples and `p` the number of features.
The free intercept adds one coordinate. Both linear routes form a dense
`(p + 1) × (p + 1)` system from the feature Gram matrix and column sums:

- `implicit_P=True`, the default, prepares a dense eigensystem and uses it
  for the free-intercept correction and MM updates.
- `implicit_P=False` prepares the dense penalized system and uses a checked
  factor solve. Avoiding an explicit inverse does not eliminate the dense
  system or factor.

One float64 augmented matrix alone requires `8 * (p + 1)**2` bytes.

| Features `p` | Bytes for one augmented matrix | Binary units |
|---:|---:|---:|
| 1,023 | 8,388,608 | 8 MiB |
| 8,191 | 536,870,912 | 512 MiB |
| 16,383 | 2,147,483,648 | 2 GiB |

These are arithmetic storage examples, not measured fit peaks or an admission
limit. The Gram matrix, augmented matrix, eigenvectors or factors, scaled
copies, temporary arrays and native-library workspace can coexist. The feature
matrix and fitted state add further storage. A sparse feature Gram product can
also fill in before it is converted to dense form. Low input density alone
therefore does not establish that a fit will fit in memory.

Dense eigendecomposition and factorization have cubic asymptotic cost in the
feature dimension. Actual elapsed time depends on dimensions, arithmetic,
library implementation and hardware; this page makes no measured speed claim.
Sparse input support is not an out-of-core solver or an automatic total-memory
admission guarantee.

## CV and zero-shift preparation

With `implicit_P=True`, `GenDWD.cv_init(X)` builds and caches a dense augmented
eigensystem and owns a copy of the training features. With `implicit_P=False`,
it still promotes and validates the inputs, but skips building a new eigensystem
that the explicit fit would not use. Explicit fitting still forms its dense
system and prepares its own checked factor; no factor cache is introduced.

The same distinction applies to the optional `P0_eig` argument of
`GenDWD.fit` and `solve_gen_dwd`. Implicit fitting uses it and validates its
consistency with the current augmented Gram matrix. Explicit fitting ignores
it without validation and does not reuse it for factor preparation.
Only precompute `P0_eig` for compatible implicit fits.

An implicit cache already present on an estimator is left intact when explicit
`cv_init` is called. This change does not automatically free previously allocated
cache memory or change ordinary implicit preparation. If the estimator is later
switched back to implicit fitting, existing geometry and input-identity checks
still govern reuse.

`GenDWDCV` uses the fold-preparation machinery across candidates. Implicit reuse
can avoid repeated eigendecomposition; skipping unused explicit preparation
avoids an additional eigensystem, but neither change removes the fit's dense
feature-square storage requirement. Account for the complete search and refit,
not just one cached MM update, when comparing costs.

When the represented penalty shift is zero, normally for `lambd=0`, fitting
requires numerically resolved full column rank of the augmented design
`[1, X]`. A direct solve with fewer rows than augmented columns is refused
before the rank-check allocation. Otherwise that rank check makes a dense
augmented design and runs a column-scaled SVD, including for sparse input.
Positive penalties that underflow to zero follow the same rule. Separate CV
eigensystem preparation is not a zero-shift resource shortcut.

The bounded exact-arithmetic recovery for particular implicit preparation
failures is restricted to small dense inputs. It is not a general sparse or
high-dimensional alternative. Neither adding a ridge nor substituting a
minimum-norm solution is an automatic response to a refused fit.

## An existing sample-space alternative

For a positive `lambd`, `KernGDWD(kernel='linear', lambd=..., q=...)` offers
a sample-space formulation of the same generalized DWD objective and free
intercept in exact arithmetic. In the linear feature formulation the objective
is mean DWD loss plus `lambd * ||beta||²`. Its positive-penalty minimizer can
be represented as `beta = X.T @ alpha`; the linear-kernel penalty is then
`lambd * alpha.T @ K @ alpha`, with `K = X @ X.T`. The intercept remains
unpenalized. Keep the same loss exponent and penalty when assessing this
relationship.

This may be a useful alternative when `n` is much smaller than `p`, because
the dense training kernel is sample-square rather than feature-square.
However:

- The kernel, eigenbasis or factor, checked arithmetic and workspace still
  require dense sample-space storage. One float64 kernel alone uses
  `8 * n**2` bytes; sparse features do not remove it.
- The kernel estimator has its own initialization, preparation, numerical
  checks and stopping behavior. Matching a fixed iteration count does not
  establish matched optimization accuracy, and independently computed
  eigensystems need not give bit-identical fitted states.
- A full fit, prediction workload or CV search is not guaranteed to be faster.
  The kernel estimator retains the training basis and has different query
  work from an explicit fitted coefficient vector.
- This is not a drop-in recommendation for `lambd=0`. The linear estimator's
  augmented-rank and numerical-refusal contracts remain in force; do not
  change the penalty merely to avoid them.
- `GenDWD` does not automatically dispatch to the kernel estimator. Choosing
  a kernel backend does not remove its dense kernel cost or certify that a
  particular problem will be accepted.

Before choosing between formulations, use the same data, preprocessing, loss
exponent and positive penalty. Compare the returned objective, scores and
stationarity diagnostics at the required optimization quality, with the
unpenalized intercept preserved. Check `termination_reason_` and
`converged_` rather than interpreting a returned model or objective-change
stop as a convergence certificate. Include preparation and required queries
or the whole CV search in any cost comparison. Validate the formulation on
the ordinary and difficult cases relevant to the intended workload.

See [linear solver choices](choosing_options.md#linear-dwd),
[linear classifiers and SOCP](../README.md#linear-classifiers-and-socp), and
[kernel numerical scope](kernel_dwd.md#numerical-scope). The SOCP classifier is
a distinct formulation with its own parameterization and solver dependencies;
it is not an automatic replacement for generalized linear DWD.
