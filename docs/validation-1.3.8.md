# DWD 1.3.8: validation and justification of major changes

This document records why the current implementation differs from the original SlicerSALT DWD package, what was actually tested, and where evidence remains limited. The main baseline is the original package. Timings of the latest numerical checks use the optimized implementation immediately before those checks; those timings are not comparisons against SlicerSALT.

Three kinds of evidence are distinguished below:

- **Current suite:** regression tests rerun against the repaired native and portable builds.
- **Audit:** focused independent or ablation experiments conducted for the package-wide review.
- **Historical:** earlier defect reproductions or performance experiments retained as evidence, not represented as newly rerun benchmarks.

Correcting the objective's implementation does not guarantee higher classification accuracy at every fixed parameter setting. Numerical equivalence also does not imply identical finite-iteration trajectories.

## Mathematical contract

For a symmetric positive-semidefinite represented kernel matrix \(K\), the objective is

\[
 \frac1n\sum_i V_q\!\left(y_i[(K\alpha)_i+b]\right)
 +\lambda\,\alpha^\top K\alpha,\qquad \lambda>0,\ q>0.
\]

The package parameter is named **lambd**. The intercept \(b\) is unregularized. With a convention that places \(\lambda_{\mathrm{other}}/2\) before the quadratic penalty, \(\lambda_{\mathrm{other}}=2\,\mathrm{lambd}\).

Let \(L=(q+1)^2/q\), \(t=n/L\), \(\delta=2\lambda t\), and \(z_i=y_iV_q'(y_i[(K\alpha)_i+b])/n\). A corrected MM action satisfies

\[
 (K+\delta I)\alpha_{\mathrm{new}}+s\mathbf1=K\alpha-tz,\qquad
 \mathbf1^\top\alpha_{\mathrm{new}}=0,\qquad b_{\mathrm{new}}=b+s.
\]

Coefficient, eigenbasis, Cholesky and constrained recovery coordinates solve this same action in exact arithmetic. The shift \(\delta\) comes from the stated penalty. No extra jitter is introduced and no positive kernel mode is intentionally discarded. Exact-arithmetic equivalence is not a promise that all finite-precision inputs and runtimes can be solved successfully.

## Evidence matrix

The test links identify reproducible regression coverage in this release. The separately described audit experiments include independent oracles and controlled removals of particular mechanisms.

| Major change | Why it is retained | Validation and result | Limits or cost |
|---|---|---|---|
| Loss-gradient dtype/order repair | The original vectorized derivative could infer an integer output type and erase fractional tail gradients. | Historical counterexample: at \(q=1\), inputs \([0,1,2]\) could return \([-1,0,0]\), rather than \([-1,-1/4,-1/16]\). Current [linear tests](../tests/test_linear_audit.py) and [kernel mathematical tests](../tests/test_kernel_math_extended.py) cover corrected arithmetic. | This is a correctness repair; matching the defective trajectory is not the target. |
| Two corrected kernel MM formulas | The original algebra did not consistently implement the stated free-intercept surrogate. | Historical independent \(8\times8\) oracle: error about 0.07319 before correction and \(2.22\times10^{-16}\) after correction. [Reference-MM tests](../tests/test_reference_mm.py) and [two-implementation tests](../tests/test_two_implementations.py) cover coordinate agreement. | Finite stopping can produce different predictions from the original implementation. |
| Linear Sherman–Morrison/general-\(q\) formula | The linear update must agree with the dense augmented system. | Audit: 64 comparisons covering \(q=0.1,1,2,10\), rank deficiency, ill-scaled float32 inputs and CSR input; labels and iteration counts matched the independent oracle, with maximum score difference \(5.71\times10^{-11}\). [Linear release tests](../tests/test_linear_release.py). | Ill-scaled floating input is judged against the represented problem; no all-input accuracy claim. |
| Float64 conversion, current NumPy/sklearn compatibility and validation | Arithmetic must use the intended precision before Gram construction; removed aliases and estimator metadata must not break fitting or external CV. | [Compatibility tests](../tests/test_compatibility.py), [kernel API tests](../tests/test_kernel_api_final.py), and [release-edge tests](../tests/test_release_edges.py) cover labels, clone/parameters, query orientation and validation. | Conversion cannot recover precision already lost in supplied data. |
| Free intercept and singular-kernel support | Penalizing the intercept or dividing blindly by zero eigenvalues changes or breaks the method. | Independent surrogate derivation above; [general linear-system tests](../tests/test_kernel_linear_system_general.py), [extended mathematical tests](../tests/test_kernel_math_extended.py), and independent singular/low-rank audit cases. | Supported PSD kernels can still exceed bounded numerical recovery capacity. |
| Cholesky and eigenbasis implementations | They provide alternative costs for the same MM problem. Eigenbasis storage can avoid repeated dense coordinate transformations; Cholesky avoids an unnecessary eigendecomposition in suitable cases. | [Performance controls](../tests/test_performance_controls.py), [two implementations](../tests/test_two_implementations.py), and [reference MM](../tests/test_reference_mm.py). Historical larger-kernel benchmarks support optimization. | Neither route is uniformly fastest; current numerical checking has a measured cost below. |
| Centered recovery | Separates a dominant constant direction without penalizing the intercept. With \(H=I-\mathbf1\mathbf1^\top/n\), its lift in the constant direction is inactive under the zero-sum constraint. | Audit family \(K=2^{54}\mathbf1\mathbf1^\top,\delta=1\): the raw diagonal shift disappears in binary64, while the centered constrained system is well conditioned. Independent residuals were zero. Controlled recovery medians: 0.3992 vs 0.4405 s at \(n=2048\), and 1.5455 vs 1.7325 s at \(n=4096\), against anchor recovery under common accurate gates. | Keep as a bounded fallback for conditioning and cost. It was not uniquely necessary in the tested removal panel, and the 9–11% component improvement is not a universal training speedup. |
| Anchor recovery | Provides a different representation when dominant constant components defeat other routes, including explicitly selected or custom kernels. | Audit removal added failures on near-constant RBF kernels for every tested \(q=0.1,1,2,10\), including callable/precomputed explicit-Cholesky cases. [Anchor tests](../tests/test_anchor_linear_system.py) and [integration tests](../tests/test_anchor_recovery_integration.py). | Automatic spectral retry has narrower eligibility and cannot replace this general capability. |
| Scalar intercept correction and midpoint proposal | Correct the same equation without changing coefficients' mathematical role. For fixed coefficients, a midpoint shift minimizes maximum scalar residual. | Removal of scalar correction failed natural and structured cases. Constructed residuals \((0,0,1.8\times10^{-10})\): mean correction leaves \(1.2\times10^{-10}\), midpoint \(0.9\times10^{-10}\), on opposite sides of a \(10^{-10}\) gate. [Scalar](../tests/test_intercept_residual_refinement.py) and [midpoint](../tests/test_intercept_midpoint_refinement.py) tests. | Keep both, with bounded checks. Dispatch is not proven optimal: an earlier MNIST replay became faster after bypassing midpoint while preserving labels. This release does not change that scheduling policy. |
| Auxiliary RKHS bound and readout refinements | Assess or improve the same candidate rather than silently changing regularization. | Audit removal of the auxiliary check rejected structured solvable cases. [A-posteriori RKHS](../tests/test_aposteriori_rkhs_bound.py), [readout](../tests/test_readout_refinement.py), [initial quadratic guard](../tests/test_initial_quadratic_guard.py), and [score reconstruction](../tests/test_score_reconstruction_recovery.py) tests cover strict-improvement, rollback and budget gates. | Floating assumptions apply; these are not an unconditional interval proof of every solver branch. |
| Bounded exact function recovery | Recovers the full represented kernel action in difficult low-rank cases instead of dropping small modes. | Audit: 72 baseline and 72 removal fits; baseline accepted 48, and removal lost 45 of those successes. Independent Decimal380 primal solutions agreed within \(8.14\times10^{-15}\) in feature norm. [Exact-factor](../tests/test_exact_kernel_factor.py) and [MM integration](../tests/test_certified_mm_integration.py) tests. | Explicit work, rank and rational-size budgets bound cost. Rank limits are recovery budgets, not permission to approximate the model. Some valid inputs still exhaust them. |
| Automatic spectral retry | Can complete an eligible failed attempt using the same kernel, initialization and objective. | [Fallback](../tests/test_kernel_auto_fallback.py), [dispatch](../tests/test_kernel_restart_dispatch.py), and [restart-branch](../tests/test_kernel_restart_branches.py) tests. | Narrow policy for trusted internal RBF automatic dispatch; not a substitute for all callable/precomputed paths. Total work can exceed the per-attempt iteration cap. |
| Compensated residual/prediction and optional native implementation | Cancellation can make ordinary products unreliable. Native arithmetic reduces the cost of the same checked computations. | [Residual](../tests/test_compensated_residual.py), [prediction](../tests/test_compensated_kernel_scores.py), [native integration](../tests/test_native_residual_integration.py), and [compiled-path](../tests/test_compiled_residual.py) tests, plus historical native/portable parity and MNIST/LETTER profiling. | Portable fallback remains available. Raw-score error control does not guarantee identical labels arbitrarily close to zero. |
| Eigen-result validation and recovery | A successful library return alone does not establish an accurate eigenpair. | [General eigen validation](../tests/test_eigen_validation_general.py), [native runtime](../tests/test_eigen_native_runtime.py), and [spectral recovery](../tests/test_native_spectral_recovery.py) tests. | General checks remain useful after an MKL update. Obsolete environment-specific probing is not the justification for retaining mathematical recovery. |
| Internal CV folds, Cartesian grids, refit and kernel cache | The original routine reversed training/validation roles, omitted grid combinations and could refit the wrong parameter. | Historical 12-row/three-fold example trained on 4 rather than 8 rows; a \(2\times3\) grid evaluated 2 instead of 6 combinations. Audit: actual six-candidate, three-fold estimator comparison matched GridSearchCV, including the correct full-data refit. [Precision/cache](../tests/test_cv_precision_cache.py), [score validation](../tests/test_cv_score_validation.py), and [remaining audit](../tests/test_remaining_audit.py) tests include two-axis precomputed slicing. | Internal CV remains supported. This establishes correctness, not universal superiority to GridSearchCV in runtime. |
| Failed-refit cleanup and serialized numerical state | A failed new fit must not expose an old fitted model as its result; numerical representation must survive serialization. | [Failed-refit](../tests/test_failed_refit.py), [kernel API](../tests/test_kernel_api_final.py), and [precision/cache](../tests/test_cv_precision_cache.py) tests. | Cleanup deliberately changes failure behavior; it does not replace a failed fit with another classifier. |
| Optional CVXPY/SOCP vectorization | Retains the original cones, objective and free intercept while reducing construction overhead. | Audit: two original/current pairs, on MNIST and rank-deficient input, had exactly equal coefficients, intercepts, objectives and query scores. [Independent SOCP regression](../tests/test_remaining_audit.py). Historical \(200\times784\) example: 0.370 to 0.250 s with objective relative difference \(1.9\times10^{-16}\). | One historical timing is not a general speed guarantee. CVXPY remains an optional dependency. |
| Kernel mean difference, KernelScaler and ancillary helpers | These original-package interfaces require their own correctness checks. | [Remaining audit tests](../tests/test_remaining_audit.py) check an independent feature-space mean-difference formula, the retained scaler formula/diagonal validation, ancillary SVM labels and solver failures, and helper argument forwarding. | Evidence is component-specific; it is not inferred from kernel DWD's MNIST accuracy. |
| Initialization, stopping, acceleration and reference/legacy APIs | These are explicit optimization or compatibility policies. | [API defaults](../tests/test_kernel_api_final.py), [accelerated MM](../tests/test_accelerated_mm.py), and [reference](../tests/test_reference_mm.py) tests. | Zero initialization and optional acceleration can change finite trajectories. Legacy mode is distinct from corrected reference mode. They are not universally necessary correctness repairs or guaranteed accuracy improvements. |

## New numerical-check repairs in 1.3.8

### Residual acceptance includes arithmetic uncertainty

The former ordinary-product acceptance gate could report zero residual while the exact equation for the represented floating inputs had a substantial error. Valid constant-dominant PSD families demonstrate this independently of MNIST and independently of a particular MKL version.

Initial and trial acceptance now include allowances for matrix products, scalar arithmetic and the coefficient-sum constraint. Relative RKHS scaling uses a lower norm estimate, so uncertainty cannot inflate the acceptance threshold. If a cheap screen is inconclusive, short BLAS dots may recompute the complete kernel product with bounded accumulation; existing compensated checks and recovery remain available. No kernel entries or modes are omitted.

On the 96-row trigger defined by RandomState(83) normal features of shape \((96,6)\), RBF \(\gamma=2^{-28}\), positive labels at indices divisible by three, \(q=1\), and \(\mathrm{lambd}=2^{-18}/48\):

| Stopping | Old equation-tolerance violations | Repaired violations | Old/new updates | Changed labels |
|---|---:|---:|---:|---:|
| Default objective stopping | 2 | 0 | 18 / 18 | 0 |
| Fixed 100 updates | 4 | 0 | 100 / 100 | 0 |

The old maximum error was \(1.28967\times10^{-10}\) against a \(10^{-10}\) gate; the repaired maximum was \(9.97222\times10^{-11}\). Scores changed by approximately \(1.4\times10^{-10}\) at most. See [residual acceptance tests](../tests/test_residual_acceptance_bounds.py).

Independent checks for each build included 196 accepted measurements spanning ordinary/block and compensated paths, 24 direct structured systems, and 27 block-product families with C, Fortran and reversed layouts. All accepted measurements satisfied the independently evaluated equation and constraint requirements. These finite tests support the error analysis; they do not prove arbitrary BLAS implementations satisfy its assumptions.

### Dual diagnostics respect the free intercept

A feasible dual slope must obey both its box constraints and exact signed mass zero. A floating dot product rounded to zero is insufficient. Quadratic uncertainty is then divided by lambda, so a fixed small tolerance cannot repair the defect for every positive lambda.

The diagnostic now uses exactly balanced dyadic slopes, conservative conjugate/quadratic endpoints, and an explicit zero-slope fallback if a stronger usable bound is unavailable. Constant kernels use an exact identity at every size. A bounded small exact-arithmetic escalation improves tightness without truncating the kernel. Accepted bitwise-asymmetric kernels receive the weaker zero fallback because the literal score map lacks the symmetric-kernel dual formula.

For a 32-row constant kernel with 19 negative and 13 positive labels, \(q=1\), and \(\mathrm{lambd}=6.25\times10^{-102}\), the old diagnostic could reject a valid five-update result with objective 0.9560546875. Another small-shift case reported 0.9035452772 as a lower bound although the independent optimum was 0.8973823014. A near-optimal 39-update result was also falsely rejected. The repaired diagnostic passes these cases without changing the MM iterate.

Coverage includes [dual certificate tests](../tests/test_dual_certificate.py), 36 independent exact slope-feasibility cases, 72 conjugate endpoint checks, and 90 independent dual comparisons. Conjugate tests at extreme positive \(q\) values validate that helper, not whole-model training at every such value. Twenty-four accepted near-asymmetric fits retained identical coefficients, intercepts, histories and predictions; only their diagnostic became weaker.

The primal objective and reported gap remain numerical estimates. A fallback certificate does not declare convergence. Default stopping remains successive absolute objective change below \(10^{-5}\), with 100 updates per attempt; reaching that cap is not a convergence certificate.

## Complete suites and MNIST regression

The repaired source was tested on Windows with Conda CPython 3.12.14. The native build includes the optional compiled arithmetic extension; the portable build omits it.

| Build | Passed | Expected skips | Failures |
|---|---:|---:|---:|
| Native | 467 | 0 | 0 |
| Portable | 458 | 9 | 0 |

The nine skips require the intentionally absent native extension. The complete suites include both original-package regressions and the new numerical-check tests. This table reports local validation, not a claim about remote CI jobs.

Six paired full-kernel MNIST fits compared the optimized pre-check baseline with the repaired source: three pairs (2 vs 3, 4 vs 9, 3 vs 8), each with 2,400 and 6,000 stratified official-training examples, seed 20260911. Pixels were float64 divided by 255; \(\gamma=0.01\), \(\mathrm{lambd}=2^{-18}\), \(q=1\), and default stopping were fixed. Complete official pair-specific test sets were used only after fitting. There was no CV or test-set tuning in this regression. All six comparisons retained the same held-out labels and iteration counts.

### Cost of the latest checks

These sequential, controlled fit-only timings used the native builds, 16 native threads on a Ryzen 9 9955HX, and medians of three repetitions for 2 vs 3. The baseline is the optimized solver immediately before the two checking repairs.

| Training rows | Before checks | Repaired | Ratio |
|---|---:|---:|---:|
| 2,400 | 0.477 s | 1.017 s | 2.13× |
| 6,000 | 2.930 s | 6.048 s | 2.06× |

The checks therefore incur a real cost; this release must not be described as a speedup over that baseline. A first more conservative checking draft was slower, and bounded block products reduced that overhead. These timings do not compare against the original SlicerSALT implementation or establish performance on every kernel.

## Remaining limits and reproduction

The general design retains centered and midpoint recovery, but their dispatch order is not proven optimal. Valid extreme PSD problems can still exhaust recovery budgets: the audit included a dyadic Gram with 24 feature columns plus a constant direction, beyond the exact recovery's rank-16 budget, and very small shifts that exhausted rational-size limits. Those cases are not all attributable to the old environment and are not evidence of invalid user data.

The arithmetic arguments assume the documented IEEE binary64 round-to-nearest, gradual-underflow, BLAS accumulation and faithful-summation behavior. Optimized eigenbasis evolution keeps its existing checkpoint/final checks; this repair does not turn every iteration of every backend into a formal interval certificate. Nor can score tolerances guarantee unchanged classification on points arbitrarily close to the boundary.

The linked tests can be run from a checkout with its test dependencies using **python -m pytest tests**. Validate native and portable installations separately, confirming which build was imported. Small independent systems and failure-edge tests establish details that a large accuracy benchmark alone cannot test. Historical performance claims above remain scoped to their stated experiments; they are not repeated as current release-wide guarantees.
