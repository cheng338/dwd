# Anchor coordinates for the constrained MM solve

The optimized solver normally uses Cholesky. If its original and mean-centered
representations fail the numerical checks, a bounded LU recovery solves the
same original constrained equations in another set of coordinates. The
repaired reference implementation retains its eigenbasis route.

Write the MM equation as `(K + delta I) alpha + s 1 = rhs`, with
`sum(alpha) = t`. In ordinary MM, `t=0`; refinement also solves equations with
nonzero `t`. Take row 0 as an anchor and express `alpha[0] = t - sum(u)`, with
the remaining coefficients equal to `u`. Subtracting the anchor equation from
the other equations gives a reduced system with entries

```
B[i,j] = (K[i,j] - K[i,0]) - (K[0,j] - K[0,0])
         + delta * (I[i,j] + 1)
reduced_rhs[i] = rhs[i] - rhs[0] + t * (K[0,0] + delta - K[i,0])
```

for `i,j > 0`. This is algebraic elimination of the coefficient-sum constraint.
It adds no regularization, kernel approximation, eigenvalue truncation or
intercept penalty. Grouped differences improve the computation for nearly
constant kernels. LU retains the stored matrix's small nonsymmetric rounding;
the recovery does not silently symmetrize the matrix.

The returned coefficients and intercept remain ordinary float64 values.
Candidates and bounded refinements must pass the original full-matrix equation,
coefficient-sum and RKHS checks. Changing coordinates does not bypass those
checks. Prediction, callbacks, validation-state restoration and serialization
retain their existing interfaces. Diagnostics identify `anchor_lu` when used.

The motivating MNIST case is a 944-row base from pair 2 versus 4 at 10% sampling,
`lambd=1e-12` and `gamma=1e-5/784`. The previous mean/midpoint scalar correction
could not repair its residual spread. The anchor formulation passes independent
Decimal checks of every equation and completes the 100-update base and full
ensemble trajectories. The iteration cap still does not imply convergence.
