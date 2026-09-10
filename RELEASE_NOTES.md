# Release 1.3.5

Failed refits now invalidate learned state consistently across the classifier
interfaces. Previously, several classes could retain old coefficients while
overwriting their labels, or retain an earlier best estimator after CV failed.
After a failed fit, prediction now raises `NotFittedError` until a later fit
succeeds. Validated private precomputation caches remain available. This repair
does not change successful-fit objectives, numerical updates or stopping rules.
See [the fitted-state contract](docs/failed_refit_state.md).

Lazy anchor-coordinate LU recovery resolves a second reproduced MNIST failure
when the original and centered Cholesky representations cannot meet the existing
accuracy checks. The anchor representation solves the same stored-kernel
constrained linear system. It does not symmetrize the kernel, add jitter, remove
positive modes or change regularization. Its returned coefficients and intercept
must pass the original equation, coefficient-sum and RKHS checks.

A numerical intercept refinement resolves a reproduced MNIST 3-versus-8
ensemble base failure at `lambd=1e-12` and `gamma=1e-5/784`. The usual mean
residual correction remains first. When needed, a bounded midpoint proposal
must pass fresh, unchanged original-equation, coefficient-sum and RKHS checks.
No kernel approximation, added regularization or change to the unregularized
intercept is introduced. MM defaults and the optional linear SOCP API remain.

The preceding numerical repair passed 390 source tests. At the weakest
regularization and RBF coefficient,
all 45 first-fold MNIST pair ensembles complete all 270 learner fits at 10% base
sampling; five learners use anchor recovery. This is a fit-validity screen, not
an accuracy measurement. The preceding midpoint repair also completes all 270
learner fits at 20% sampling. Those repair checks did not access the official
test set. Final release validation is summarized in [VALIDATION.md](VALIDATION.md).

The exact failed 1,917-row base and the full five-base ensemble with its final
2,875-row refit complete all 100 updates. The full pair ensemble takes 74.406
seconds in a one-thread validation replay and reaches 95.8698% validation
accuracy on 2,397 held-out fold images. These are diagnostic results for one
extreme parameter setting, not the final multiclass comparison or a speedup
claim. All fits correctly report reaching the iteration cap without convergence.

All 21 successful models in the frozen 30-fit historical replay pass independent
score, objective and serialization checks, with unchanged predictions. Twenty
states are bitwise unchanged; one extreme valid fit has decision differences
below 7.6e-10. The same nine extreme synthetic numerical rejections remain.

See [the intercept explanation](docs/intercept_midpoint_refinement.md),
[anchor recovery equations](docs/anchor_linear_system.md),
[change history](CHANGES.md), and the preserved
[1.3.4 performance release](docs/history/RELEASE_NOTES-1.3.4.md).
