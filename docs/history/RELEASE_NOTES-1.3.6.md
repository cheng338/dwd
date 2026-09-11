# Release 1.3.6

RBF kernels created internally by the named kernel path now recover from
roundoff-induced asymmetry using the same stable distance formula for training
and prediction. External precomputed kernels and custom kernel outputs retain
strict validation; genuinely asymmetric inputs are not silently averaged.

For trusted internally generated RBF kernels, unaccelerated optimized automatic
MM fits without callbacks may restart once through the existing spectral backend
after constrained recovery is exhausted. The restart uses the original kernel,
initialization and stopping settings. It preserves the DWD objective and
unregularized intercept. The iteration limit applies to each attempt; diagnostics
distinguish the successful attempt from discarded work and include total time.
Explicit backends retain their behavior, and unresolved numerical failures raise.

The repairs were validated on 14 saved constrained-recovery failures and 99 saved
RBF-symmetry failures, alongside construction, cache, prediction, API and solver
regressions. These checks establish behavior on the tested cases, not universal
acceptance or an accuracy or speedup guarantee. The default stopping rule and
optional SOCP formulation remain unchanged.

See the [kernel guide](../kernel_dwd.md) for the numerical policy and
[release 1.3.5](RELEASE_NOTES-1.3.5.md) for preserved prior changes.
