# DWD 1.3.9 changes and validation scope

This October 3, 2026 release of the fork, developed under Chang Cheng's guidance,
preserves the corrected DWD objective, loss, free intercept, kernel construction
and stopping tolerances. It changes native
arithmetic reuse and adds an explicitly selected residual assessment order.
The historical `solver_mode='legacy'` remains a compatibility mode with known
incorrect algebra; it is not a correct implementation of the cited methods.

The native residual core shares a double-length product between two separate
expanded accumulators. Nonzero components may be negated under the existing
guarded binary64 assumptions. If either component is zero, the original negative
product is recomputed to preserve signed-zero behavior. Residuals are never
obtained by subtracting a rounded score. Numerical acceptance bounds are unchanged.

The explicit Zig build uses baseline x86-64 instructions for Windows AMD64
wheels, with strict floating-point flags. A general wheel must not silently
inherit the build host's CPU instruction set. The native helper retains its
existing runtime guards; an ABI tag alone does not promise acceleration on every
supported Python runtime. Pre-release numerical validation used conventional
CPython 3.12 on Windows AMD64 with AOCL. It did not exercise other platforms,
older CPU hardware or free-threaded Python.

`KernGDWD` and `KernGDWDCV` accept
`residual_check_order='refinement_first'` (the unchanged default) or `'adaptive'`.
Adaptive is restricted to optimized, unaccelerated Cholesky fits. After two
discarded correction trials whose original states pass accurate assessment,
it tries accurate assessment before another speculative correction. A failed
optional probe falls back to the original ordering. Factor changes reset this
history. Spectral restart and anchor recovery keep their existing behavior.

Adaptive ordering preserves the objective, equation, constraint, RKHS error and
descent checks. It can change finite-iteration states, and is a separate numerical
policy rather than a new statistical model or a new convergence theorem.
Reported probe time overlaps accurate-assessment time; timing counters must not
be added as though all were exclusive. An accurate-first miss can add work.

The generalized kernel DWD objective uses a mean loss plus `lambd * ||f||^2`,
where the norm excludes the intercept. For `q=1`, the dissertation's `lambda/2`
convention therefore uses twice this `lambd`.
Sampling, objective normalization and regularization are unchanged by
these performance changes. Benchmark adoption of adaptive ordering requires a
separate method identity; results obtained with different policies must not be mixed.

Public API repairs reject complex explicit linear and legacy-kernel starting
coefficients before any lossy conversion. Linear prediction raises on nonfinite
decision values; finite ordinary arithmetic is unchanged. Cross-validation
chooses row-only or two-axis Gram slicing after applying each candidate's
`kernel` parameter and reuses that representation within the fold.

`KernelScaler.fit` now clears partial or stale learned state on failure. It
continues to own a copy of the fitted diagonal and preserve the historical
normalization whose diagonal is the training count. Ordinary finite scaling is
unchanged; a fallback evaluates the same factor without first dividing a
positive subnormal diagonal by that count. Nonfinite transformed values raise.
Preparing a new kernel path with `KernGDWD.cv_init` invalidates the old predictor,
so old coefficients cannot be used against newly prepared training rows. A
subsequent successful fit can still reuse the validated private preparation.
These changes do not introduce a new objective, intercept penalty, regularizer
or stopping rule.

Separate validation receipts record the development-build benchmarks and source
checks. These bounded checks are not a completed MNIST search or a claim that
all selectable historical modes have been re-proved correct. The final
development source suite passed 509 tests with three skips; native and portable
wheel installations and eight checks from the extracted source archive also
passed. Those receipts identify their tested artifacts; stable release artifacts
are built and checked separately. No universal speedup is claimed.
