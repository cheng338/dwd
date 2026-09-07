# Release 1.3.2

This release updates **cheng338/dwd as an independently maintained DWD fork**, bringing the accumulated numerical, API, cross-validation and performance repairs into one package. Both repaired kernel implementations remain available through the same estimator API:

```python
from dwd.gen_kern_dwd import KernGDWD

fast = KernGDWD(implementation="optimized")  # default
reference = KernGDWD(implementation="reference", random_state=7)
```

The reference preserves Slicersalt’s full-eigenbasis, coefficient-MM approach. The optimized implementation uses checked Cholesky solves by default. Both use corrected MM updates for the same full-kernel DWD objective, with an **unregularized intercept**. Their default initializations differ, so finite-iteration predictions can differ. Neither implementation approximates the kernel or silently adds regularization.

### Accumulated changes from upstream and the earlier fork

- Correct the order-dependent loss-gradient defect and kernel MM algebra, including the intercept and regularization terms.
- Repair internal CV fold direction, Cartesian parameter enumeration, winner refitting and kernel-cache invalidation. Plain estimators support scikit-learn cloning, pipelines and `GridSearchCV`; `fit` performs no hidden parameter search.
- Promote corrected feature calculations to float64, replace removed NumPy aliases, and preserve original class labels in predictions.
- Add checked kernel preparation, singular-kernel handling, bounded numerical refinement and recovery, and explicit stopping/convergence diagnostics. Generic checks remain; earlier MKL-specific subprocess workarounds have been removed.
- Reduce repeated numerical work and preserve accurate prediction through adaptive or compensated accumulation, including batching and serialization.
- Retain generalized linear DWD and improve the optional vectorized CVXPY/SOCP path. **The `socp` extra provides linear SOCP DWD; it does not switch kernel DWD to SOCP.**

The incremental 1.3.2 changes reject invalid internal-CV scorer outputs before selection/refitting and reduce conversion overhead in compensated prediction. Publication testing additionally found spectral refinement stagnation at float64 rounding limits under OpenBLAS. A lazy, bounded adjacent-float correction now checks nearby coefficient representations and the free intercept against the original equations and unchanged residual, coefficient-sum and RKHS gates. It preserves the kernel, objective and MM iteration budget. Frozen test oracles retain their exact bytes on Windows, and intercept-refinement tests accommodate already-accurate native solves while preserving their Decimal checks. Two saved 3-vs-8 models showed approximately **15–16% less decision-evaluation time**, with bitwise-identical scores. This is a scoped prediction measurement, not a training-speed or universal-speed guarantee.

### Objective, defaults and migration

The kernel objective is `mean(V_q(y * f)) + lambd * alpha.T @ K @ alpha`, with free intercept `b` in `f = K @ alpha + b`. Parameter names remain `lambd`, `q` and `kernel_kws`. For a convention written as `(lambda / 2) * ||f||²`, use `lambd = lambda / 2`; do not silently reinterpret parameters from older scripts.

Defaults remain ordinary MM, `q=1`, absolute objective-change tolerance `1e-5`, and a **100-update cap**. Acceleration is optional and disabled. Objective stopping and numerical convergence are reported separately. External ensemble methods should cross-validate their complete training procedure; no ensemble-specific tuning is imposed by this package.

Python 3.11+ and scikit-learn 1.6+ are required. Install the attached wheel, or install this tagged checkout with `pip install .` / `pip install ".[socp]"`. An unqualified `pip install dwd` may retrieve upstream’s PyPI distribution.

### Validation and limitations

The revised source and actual wheel each passed **334 tests** under updated MKL. The same wheel passed **334 tests under OpenBLAS**. On each runtime, base-only validation passed **328 tests with six expected optional skips**, without importing CVXPY. The MKL environment used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0 and scikit-learn 1.9.0; the OpenBLAS environment used Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.0. The GitHub release gate additionally requires the Linux/Windows, Python 3.11/3.12, base/SOCP matrix to pass.

MNIST comparisons support faster optimized training, with mixed accuracy. A 6,000-training/1,000-development ensemble comparison found 3.6–6.3× faster training than original Slicersalt across three digit pairs. The development set had been reused; all 108 base and final learners in the repaired ensembles reached the 100-update cap without meeting the convergence check. These are finite-budget results, not proof of solved optima or universal accuracy gains. A new actual-wheel replay of 30 stored public synthetic cases retained 21 successful fits and nine unresolved extreme nearly constant RBF fits. Every returned model passed independent Decimal100 score/objective, finite-output and serialization checks. These bounded checks do not guarantee success for every valid kernel.

See the [kernel guide](docs/kernel_dwd.md) and [validation record](VALIDATION.md).

Original Iain Carmichael attribution, Slicersalt/Kitware credits and the MIT license are retained.
