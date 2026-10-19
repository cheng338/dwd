# DWD 1.3.15 — October 17, 2026

This release rejects invalid complex inputs before conversion or optimization
and improves reporting in the resumable CV example. The DWD objective, free
intercept, real-input solver arithmetic, numerical acceptance checks, stopping
tolerances and defaults are unchanged.

## Real-input validation

Complex starting intercepts and complex values hidden in object arrays are
rejected consistently, including zero-imaginary complex values. The checks
cover initial coefficients, class labels and supplied eigendecompositions,
including nested object arrays. Invalid supplied eigenpairs are rejected rather
than silently replaced by recomputed pairs. Failed fits retain the existing
cleanup of fitted state.

Normal numeric arrays use dtype checks; object arrays require inspection of
their entries. This corrects input validation and does not promise a performance
speedup or zero validation overhead. These malformed-input regressions do not
imply that earlier real-data studies used complex inputs.

## Resumable CV diagnostics and grid reporting

The example retains existing ordinary, native and compensated residual-check
counts, ordinary acceptances and refinement-step counts in its compact
diagnostics and completed-fold receipts. Missing counts remain unknown. No
additional solver checks are performed.

The report identifies the winner as best among the supplied candidates. It
includes exact ties, the gap to the next ranked candidate, the gap to the next
strictly lower score, and numeric parameter positions within conditional slices
that hold other parameters fixed. These positions describe the supplied grid;
they do not establish convergence or a global optimum.

Candidate order, duplicate candidate labels, exact equally weighted fold
accuracy, first-maximum selection and final refitting are preserved. Reporting
does not expand or refine the grid, merge candidates or reuse additional fits.
If report construction fails, the successful CV result remains available with
a notice that the report is unavailable. See the
[resumable CV guide](../../docs/resumable_cv.md).

The changed example source enters checkpoint identity. Checkpoints from older
example sources require their matching source and runtime, or a new run
directory with the current example. Existing checkpoints are not migrated.

## Validation scope

Local development checks passed 141 tests for input validation and existing
numerical/API behavior, plus 16 valid-real comparisons that exactly matched
pre-change coefficients, intercepts, objective histories, scores and predictions.
A separate reporting/example run passed 71 tests. These test sets have overlapping
coverage and are not presented as one count of distinct tests.

The checks used Windows and conventional CPython 3.12, with one native thread
for small fixtures and the existing two-worker CV regression. They establish
behavior within the tested scope, not a universal timing or cross-platform
claim. Built-artifact and CI checks are separate from these development results;
see the [validation record](../../VALIDATION.md#version-1315-development-validation).

## Python support and distribution

Python 3.11+ remains required. The Windows AMD64 native wheel is
`dwd-1.3.15-cp311-abi3-win_amd64.whl`; `dwd-1.3.15-py3-none-any.whl` omits the
optional compiled helper. Extension sources, compiler flags and numerical
runtime guards are unchanged. An ABI-compatible extension does not enable
accelerated paths outside their qualified runtime conditions; NumPy/SciPy BLAS
is independent.

Python 3.10 companion maintenance has ended. Historical assets and validation
records remain available; this release does not produce a Python 3.10 wheel.
The configured CI matrix covers Python 3.11, 3.12 and 3.13 on Windows and Ubuntu,
with base and optional SOCP configurations.

This is a fork of slicersalt/dwd, originally implemented by Iain Carmichael,
with upstream maintenance by David Allemang and Kitware. Subsequent fork
development is guided by Chang Cheng. Original credits and the MIT license
are retained. See the [changelog](../../CHANGES.md), [installation](../../README.md#installation)
and [previous release notes](../../docs/history/RELEASE_NOTES-1.3.14.md).
