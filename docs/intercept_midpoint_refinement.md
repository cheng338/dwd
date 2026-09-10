# Bounded intercept refinement

For a fixed coefficient vector, the original constrained MM equation is
`(K + shift I) x + s 1 = rhs`. A constant change to `s` subtracts the same
constant from every equation residual. The midpoint of the smallest and largest
residual minimizes the maximum absolute residual over such scalar changes.

The implementation first retains the existing mean correction. If necessary,
it tries the midpoint as a second, bounded proposal. The rounded intercept must
be finite and representably different. A cheap equation check is only a filter:
a fresh compensated evaluation must pass the unchanged equation, coefficient-sum
and RKHS checks before the correction can be returned. A successful correction
uses an existing refinement step. Coefficients, kernel entries, regularization,
the free-intercept objective and MM stopping settings are unchanged.

This resolves a full-training-fold MNIST 3-versus-8 case encountered while
screening multiclass parameter ranges. The exact failing 1,917-row sampled base,
with `lambd=1e-12`, `gamma=1e-5/784`, `q=1` and zero initialization, now completes
100 updates. On its first update, the existing mean correction leaves a maximum
residual around `1.78e-10`; the midpoint gives around `8.95e-11` against the
unchanged `1e-10` gate. An independent Decimal calculation of all 1,917 equations
confirms this first-update result. The complete five-base ensemble and final
2,875-row refit also complete. Reaching the iteration cap is not a claim of
convergence or statistical optimality.

This is a targeted numerical repair, not a universal elimination of all
floating-point failures. The frozen 30-fit synthetic replay retains the same
nine rejections. All 21 returned models pass the existing independent score,
objective and serialization checks. Twenty returned states are bitwise equal
to the prior release; one difficult valid case has decision differences below
`7.6e-10` and objective-history differences below `9.9e-12`.
