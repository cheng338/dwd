# Estimator display and dependency limitation

Estimator display has two separate layers. Current `GenDWDCV` and `KernGDWDCV`
use immutable tuple defaults so parameter inspection and text representation
do not encounter ndarray-default comparisons. Their grid values and numerical
behavior are unchanged; see [default CV grids](kernel_dwd.md#default-cv-grids-and-display).
HTML rendering is provided by scikit-learn and also serves ensemble-DWD,
`Pipeline`, `GridSearchCV` and unrelated sklearn estimators.

The affected scikit-learn 1.9.1 renderer can still fail during HTML display.
DWD and Ensemble DWD ship without the dependency backport described below.

## Affected sklearn 1.9.1 HTML renderer

The source investigated on 2026-11-04 reads its UTF-8 `estimator.js` resource
without an explicit encoding. On an interpreter whose native text encoding
cannot decode the asset, HTML display can raise `UnicodeDecodeError` even
though plain `repr` works. The investigated examples were GBK/cp936 on Windows
and ASCII under a Linux C locale. This is a text-resource defect, not a DWD
optimization error or an AOCL/MKL numerical issue.

The separately investigated source backport added one encoding argument in
`sklearn/utils/_repr_html/estimator.py`:

```diff
-        with open(str(Path(__file__).parent / "estimator.js"), "r") as f:
+        with open(str(Path(__file__).parent / "estimator.js"), "r", encoding="utf-8") as f:
```

This historical candidate left the JavaScript bytes, CSS reads and generated
HTML logic unchanged. The defect lies in the shared renderer: a package-local
display override would not cover direct sklearn rendering or outer containers.
Neither package edits that renderer, installs an import-time monkeypatch,
changes global locale/UTF-8 settings, or disables HTML to conceal the issue.

The recorded investigation compared the
[sklearn 1.9.1 source](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/utils/_repr_html/estimator.py)
with the explicit UTF-8 read in upstream main. See the
[upstream issue](https://github.com/scikit-learn/scikit-learn/issues/35001).
The comparison identifies the issue in 1.9.1. Other releases need their own
source or runtime checks.

## Validation on Windows and Linux

The UTF-8 backport produced the same normalized HTML as the working UTF-8
reference on each tested host, including Unicode escaping and concurrent
display. Both isolated copies and temporarily patched installations passed.
After restoration, both hosts again used the original renderer. See the
[November 4 display checks](../VALIDATION.md#cv-default-grids-and-estimator-display-2026-11-04).

The full-file hashes identify the two historical sklearn 1.9.1 configurations:

| State | SHA-256 of `sklearn/utils/_repr_html/estimator.py` |
| --- | --- |
| Original, restored in the validation environments | `4905b323b62c2cfd74fce723e54f4506c4dfffce04b857f34e873737985c59dc` |
| Tested UTF-8 candidate, not retained | `1168b81490b31e8cef185a29f665b0e14f7fba8f476a2f196ec0354557f3060f` |

The original scikit-learn 1.9.1 renderer can still fail under the affected
non-UTF-8 configurations. DWD's tuple defaults fix CV parameter display.
Installation leaves scikit-learn's renderer and the interpreter locale intact.
