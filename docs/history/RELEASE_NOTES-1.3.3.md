# Release 1.3.3

This release accelerates the numerical residual checks that can dominate kernel
DWD training for nearly constant kernels with weak regularization. It retains
both repaired implementations in cheng338/dwd and the same estimator API.

The portable residual routine reuses the coefficient vector's exact mantissa
split and converts float64 arrays to Python lists in C. On audited CPython 3.12
builds, guarded native compensated dot products provide a faster check with
explicit outward error bounds. Accepted states retain the equation,
coefficient-sum and RKHS gates. A residual lower bound may establish that
refinement is necessary. Uncertain checks use the existing expanded calculation
on the untouched state. Unsupported versions and extreme arithmetic use the
portable path; no extra dependency is required.

The native calculation uses the triple-length algorithm in CPython 3.12's
`math.sumprod`, with the forward bound in Ogita, Rump and Oishi (2005),
Proposition 5.11. Runtime, rounding, magnitude and term-count guards limit use to
the audited arithmetic. Score uncertainty is propagated into a conservative
lower bound for the relative RKHS scale. This is a numerical error estimate
under stated floating-point assumptions, not a formal interval certificate.

The full kernel, objective `mean(V_q(y*f)) + lambd * alpha.T @ K @ alpha`,
unregularized intercept, MM update rules, parameter notation and initializations
are unchanged. Defaults remain ordinary MM, `q=1`, `obj_tol=1e-5`,
`max_iter=100`, and no acceleration. Reference retains its eigenbasis/coefficient
implementation. The optional `socp` extra continues to provide linear CVXPY DWD.

A 9,671-training-image MNIST fold with RBF gamma `1e-8` and `lambd=2**-30`
reproduced the slow path. A three-update check took 94.27 seconds with 1.3.2 and
23.69 seconds with this repair; coefficients, intercept, objective history and
validation predictions were bitwise identical. The baseline includes a one-time
diagnostic state write, so these are approximate fitting timings. Required
residual evaluation itself fell from 84.88 to 14.24 seconds.

The complete 100-update repaired fit took 452.50 seconds (7.54 minutes), with
97.0223% accuracy on the 2,418-image validation fold, matching the saved 1.3.2
accuracy. It used the full kernel and reached the existing iteration cap. The
earlier full fit took 71.95 minutes while other CV workers ran; the different
contention prevents attributing that entire difference to this repair.

Twelve additional paired 100-update checks across the three dissertation digit
pairs retained bitwise-identical fitted states and predictions. A 30-case public
synthetic replay retained 21 successful fits and the same nine known extreme
rejections; all returned models passed Decimal100 checks and matched 1.3.2 state
bytes. These results do not establish universal accuracy or speed guarantees.

See [validation](VALIDATION.md), the [kernel guide](docs/kernel_dwd.md), and the
[archived 1.3.2 notes](docs/history/RELEASE_NOTES-1.3.2.md) for earlier repairs.
Original authorship and the MIT license are retained.
