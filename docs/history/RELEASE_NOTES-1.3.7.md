# Release 1.3.7

Compensated residual checks now apply the existing outward-rounded error bounds
to arrays, preserving every rounding point and numerical acceptance threshold.
The optional compiled row evaluator is used for kernels with at least eight
rows. Checks below 2,048 rows use one worker without repeated thread-pool
inspection; larger checks retain the caller's active native-library budget.

The strict-arithmetic C source, guarded runtime support and portable fallback
remain unchanged. This update preserves the kernel DWD objective, unregularized
intercept, solver choices, stopping defaults and optional linear SOCP support.
Regression tests cover dispatch boundaries, scalar/compiled bitwise agreement,
input immutability, fallback behavior and conservative error bounds.

See [compiled residual checks](../compiled_residual.md) for runtime details and
[release 1.3.6](RELEASE_NOTES-1.3.6.md) for prior repairs.
