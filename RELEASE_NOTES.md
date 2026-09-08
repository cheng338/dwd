# Release 1.3.4

An optional strict-arithmetic C extension accelerates the compensated residual
checks that still dominated some large kernel DWD fits after 1.3.3. It evaluates
the same ordered score and original-equation residual sums without repeated
Python-list conversions. Independent rows run within the caller's active BLAS
thread budget and share the same kernel matrix.

The Python layer retains the existing numerical bounds, acceptance gates and
fallbacks. The objective, full kernel, unregularized intercept, MM updates,
initialization and stopping controls are unchanged. Defaults remain ordinary
MM, q=1, obj_tol=1e-5, max_iter=100 and no acceleration. Both repaired reference
and optimized implementations remain available. The ensemble API and optional
linear CVXPY/SOCP solver are unchanged.

On the saved 12,089-training-image MNIST 2-versus-3 configuration, gamma=1e-4
and package lambd=2^-31, a fresh eight-thread 100-update fit took **47.12 seconds**.
The previous study recorded a **377.39-second median** and 379.00-second primary
fit. Coefficients, intercept, the entire objective history, preprocessing values
and fixed training prediction probes were bitwise identical to the saved model.
The same 105 compensated checks and five numerical refinements occurred.
Checking took 20.03 seconds versus 352.11 seconds in the saved primary fit.

A fresh matched three-update comparison took 24.23 seconds with 1.3.3 and
15.40 seconds with the candidate; residual checking fell from 9.64 to 0.56 seconds,
with all compared states and probes bitwise identical. Short prefixes include
setup/readout costs and are not full-fit speed ratios. The historical median and
a fresh single fit are different summaries; these are scoped measurements,
not universal performance guarantees.

The platform wheel uses the CPython stable ABI and has no NumPy C API dependency.
A compiler is a build-time requirement for accelerated source builds; the Python
checker remains available if the extension is absent. The numerical native-screen
runtime gate remains audited CPython 3.12. There is no runtime compiler download.
See [build and runtime details](docs/compiled_residual.md).

The adapted accumulation helpers retain CPython attribution and its complete PSF
license. Original DWD credits and MIT licensing remain intact. The frozen public
synthetic replay returns the same 21 models and retains the same nine documented
extreme numerical rejections; this performance repair does not resolve those
separate cases.

See [validation](VALIDATION.md) and [1.3.3 history](docs/history/RELEASE_NOTES-1.3.3.md).
