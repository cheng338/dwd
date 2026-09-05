# DWD 1.0.5 compatibility fork

Historical record for `1.0.5+compat2`. The newer `1.0.5+audit1` candidate also
contains result-changing solver corrections and optimizations; see
[AUDIT_CHANGES.md](AUDIT_CHANGES.md). Statements below describe compat2 only.

This repository is a compatibility-focused fork of
[`slicersalt/dwd`](https://github.com/slicersalt/dwd), based on upstream commit
`b564db19193674d967a9dc9327709869d5b67078` and the DWD 1.0.5 release.

## Original work and license

DWD was originally implemented in Python by Iain Carmichael and is maintained
upstream by Kitware, Inc. The project implements methods described by Marron,
Todd, and Ahn (2007) and Wang and Zou (2018). Full citations and author links
remain in [README.md](README.md).

The upstream MIT license and its copyright notice are preserved verbatim in
[LICENSE.txt](LICENSE.txt). These compatibility changes are distributed under
the same license.

## Compatibility changes

The `1.0.5+compat2` version contains four narrowly scoped source corrections:

1. Replace the removed NumPy alias `np.int` with Python's `int` when converting
   predicted class indices. This changes neither floating-point precision nor
   the predicted labels.
2. Put `KernelClfMixin` before `BaseEstimator` for `KernGDWD` and `KernGDWDCV`,
   following scikit-learn's estimator mixin convention.
3. Put `LinearClassifierMixin` before `BaseEstimator` for the SOCP `DWD`
   estimator for the same reason.
4. Set `otypes=[float]` on `np.vectorize(V_grad_)`. Without an explicit output
   type, NumPy infers the output dtype from the first evaluated element; an
   integer-valued first element can truncate later fractional gradient values.
   The scalar `V_grad_` formula itself is not modified.

The DWD objective, scalar gradient formula, optimization stopping criteria, and
default 100-iteration limit are unchanged.

## Validation

The compatibility tests verify that:

- vectorized gradients retain fractional values and agree with scalar
  `V_grad_` evaluations;
- binary prediction returns the estimator's class labels without relying on
  `np.int`;
- estimator mixins precede `BaseEstimator` in the affected class MROs; and
- package metadata reports `1.0.5+compat2`.

The fixes were also exercised with Python 3.11, NumPy and scikit-learn.
