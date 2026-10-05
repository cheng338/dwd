# DWD 1.3.9 — October 3, 2026

This release improves numerical implementation, API validation and examples.
The DWD objective, unregularized intercept, regularization convention and
default stopping settings are unchanged.

- Reuse a double-length product in the native residual evaluator while retaining
  separate score and residual accumulators, strict floating-point arithmetic and
  existing acceptance bounds. A zero-component fallback preserves signed-zero
  behavior.
- Target baseline x86-64 explicitly in Windows AMD64 Zig builds instead of
  inheriting the build host's CPU features.
- Add opt-in `residual_check_order='adaptive'` for eligible optimized,
  unaccelerated Cholesky fits. The default remains `'refinement_first'` and
  acceptance gates are unchanged. Optional ordering can change finite-iteration
  results; it does not establish solver convergence.
- Repair fitted-state cleanup in `KernelScaler` and public kernel initialization,
  and avoid unnecessary intermediate underflow for tiny positive kernel
  diagonals. Reject complex initial coefficients and nonfinite linear decision
  scores explicitly.
- Correct candidate-specific precomputed-kernel slicing in cross-validation
  while retaining compatible fold preparation reuse.
- Repair historical notebook and figure imports, retain both intended gamma
  choices, remove duplicate fits, and include notebook fixtures and test build
  dependencies in source distributions.

Install a compatible wheel from the
[GitHub release assets](https://github.com/cheng338/dwd/releases/tag/v1.3.9),
or install this release checkout. The native wheel is
`dwd-1.3.9-cp311-abi3-win_amd64.whl`; the portable alternative is
`dwd-1.3.9-py3-none-any.whl`. These artifacts are distributed through GitHub,
not PyPI. See the [installation instructions](README.md#installation).

Pre-release validation included independent arithmetic checks, baseline
comparisons, bounded MNIST fits, source tests and installations of both wheel
variants. It used conventional CPython 3.12 / Windows AMD64 with AOCL. Full
MNIST search, other platforms, older CPU hardware and free-threaded Python were
not validated by those checks. No universal speedup is claimed. See the
[detailed changes and evidence scope](docs/candidate_changes.md).

This is a fork of [slicersalt/dwd](https://github.com/slicersalt/dwd), originally
implemented by Iain Carmichael with upstream maintenance by David Allemang and
Kitware. Subsequent development of this fork is guided by Chang Cheng. Original
credits and the MIT license are retained.

The previous [1.3.8 release notes](docs/history/RELEASE_NOTES-1.3.8.md) are
preserved separately.
