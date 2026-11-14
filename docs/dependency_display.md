# Estimator display and dependency limitation

Estimator display has two separate layers. Current `GenDWDCV` and `KernGDWDCV`
use immutable tuple defaults so parameter inspection and text representation
do not encounter ndarray-default comparisons. Their grid values and numerical
behavior are unchanged; see [default CV grids](kernel_dwd.md#default-cv-grids-and-display).
HTML rendering is provided by scikit-learn and also serves ensemble-DWD,
`Pipeline`, `GridSearchCV` and unrelated sklearn estimators.

The package fix does not repair scikit-learn's separate HTML-resource decoder.
The temporary dependency backport tested during validation was outside the
authorized package-only scope and has been reverted in both validation
environments. Their original sklearn 1.9.1 renderers remain affected under the
non-UTF-8 conditions below. No environment edit or locale change is included in
these package instructions.

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
This is a versioned source observation, not a claim that all earlier or later
releases have the defect or that a particular replacement release has been
validated. It does not authorize a dependency upgrade or local patch.

## Historical validation and current scope

The backport passed isolated Windows and Linux checks against each host's
working UTF-8 reference, including Unicode escaping and concurrent display.
Later checks also passed while it was temporarily installed. Those results
describe the patched configuration only. The installed dependency changes were
reverted; they are not an applied package fix. See the preserved chronology and
restoration scope in [validation](../VALIDATION.md).

The full-file hashes identify the two historical sklearn 1.9.1 configurations:

| State | SHA-256 of `sklearn/utils/_repr_html/estimator.py` |
| --- | --- |
| Original, restored in the validation environments | `4905b323b62c2cfd74fce723e54f4506c4dfffce04b857f34e873737985c59dc` |
| Tested UTF-8 candidate, not retained | `1168b81490b31e8cef185a29f665b0e14f7fba8f476a2f196ec0354557f3060f` |

These hashes and the diff record the investigation; they are not deployment or
reapplication instructions. Neither package contains code that rewrites
installed dependency files, changes locale settings, or reapplies this backport
after an environment reinstall or upgrade. Any future dependency change is separate
from this package-only fix and requires its own authorization and validation.

The remaining limitation is specific to HTML rendering under affected
non-UTF-8 configurations. Tuple defaults fix the CV parameter-representation
defect, while fitting and numerical results are unchanged. No package version,
dependency lock, wheel or public release was changed to deploy the HTML
candidate.
