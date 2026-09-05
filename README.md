# Overview

> **Audited research fork — 1.0.5+audit1:** Based on upstream DWD 1.0.5, this
> candidate includes compatibility fixes, performance improvements, and opt-in
> corrections to the generalized DWD solvers. These corrections can change fitted
> models and accuracy. Read [AUDIT_CHANGES.md](AUDIT_CHANGES.md) before upgrading.
> The original DWD loss and gradient formulas and the unregularized intercept are
> unchanged. Historical compatibility fixes are in
> [COMPATIBILITY_FIXES.md](COMPATIBILITY_FIXES.md).

This package implements Distance Weighted Discrimination (DWD). DWD For details see
([Marron et al 2007][marron-et-al], [Wang and Zou 2018][wang-zou]). Originally
implemented in Python by [Iain Carmichael][iain-carmichael], with upstream
maintenance by David Allemang / [Kitware, Inc][kitware]. This independent fork is
maintained at [cheng338/dwd](https://github.com/cheng338/dwd); audit changes were
prepared with Codex. Original author credit and the MIT license are retained.

The package currently implements:

- Original DWD formulation solved with Second Order Cone Programming (SOCP) and solved
using cvxpy.

- Generalized DWD (gDWD) and kernel gDWD solved with the Majorization-Minimization
algorithm presented in Wang and Zou, 2018.


Marron, James Stephen, Michael J. Todd, and Jeongyoun Ahn. "Distance-weighted
discrimination." Journal of the American Statistical Association 102, no. 480 (2007):
1267-1271.

Wang, Boxiang, and Hui Zou. "Another look at distance‐weighted discrimination." Journal
of the Royal Statistical Society: Series B (Statistical Methodology) 80, no. 1 (2018):
177-198.

# Installation

This fork targets Python 3.11+ and scikit-learn 1.6+. The audited configuration is
Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.0 on Windows.
Other dependency combinations are not claimed to have been tested. Install from
a checkout of this fork, not the upstream PyPI package:

```
python -m pip install .
```

The conic solver `socp_dwd.DWD` depends on optional `cvxpy`. From this checkout,
use `python -m pip install ".[socp]"` to include it; see the
[cvxpy installation instructions][cvxpy].

[Flit][flit] is the build backend, with metadata in `pyproject.toml`. To build a
wheel and source distribution, install the `build` frontend and run
`python -m build`. This does not require compiling NumPy or SciPy.

## Choosing the solver

Set `solver_mode='schur'` explicitly for the corrected linear/kernel MM updates.
The package default remains `'legacy'` to avoid silently changing existing
experiments; **legacy retains known update-algebra errors** and is intended for
historical comparisons only. Both modes leave the intercept unregularized.
Corrected mode forms its matrices from float64 features. For an externally
precomputed kernel or eigensystem, compute it in float64 too; casting an already
rounded float32 matrix cannot restore the lost precision.

`max_iter` controls the step cap. Inspect `converged_`, `n_iter_`,
`termination_reason_` and `gradient_inf_norm_`; a small objective change alone
does not prove convergence to the optimum. Larger iteration caps do not guarantee
better classification accuracy. Corrected mode can reject numerically unstable
rank-deficient/tiny-penalty combinations rather than silently modify the kernel.

[cvxpy]: https://www.cvxpy.org/install/index.html
[flit]: https://github.com/takluyver/flit

# Example

```python
from sklearn.datasets import make_blobs
from dwd.socp_dwd import DWD

# sample sythetic training data
X, y = make_blobs(
    n_samples=200,
    n_features=2,
    centers=[[0, 0],
             [2, 2]],
)

# fit DWD classifier
dwd = DWD(C='auto').fit(X, y)

# compute training accuracy
dwd.score(X, y)  # 0.94
```

![dwd_sep_hyperplane][dwd_sep_hyperplane]

```python
from sklearn.datasets import make_circles
from dwd.gen_kern_dwd import KernGDWD

# sample some non-linear, toy data
X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)

# fit corrected kernel DWD with a Gaussian kernel
kdwd = KernGDWD(
    lambd=.1, kernel='rbf',
    kernel_kws={'gamma': 1},
    solver_mode='schur', random_state=1,
    max_iter=1000, obj_tol=1e-7,
).fit(X, y)

# compute training accuracy
kdwd.score(X, y)
print(kdwd.converged_, kdwd.n_iter_, kdwd.gradient_inf_norm_)
```

![kern_dwd][kern_dwd]

For more example code see [these example notebooks][example-notebooks] (including the code
to generate the above figures). If the notebooks aren't loading on github you can copy/paste the notebook url into https://nbviewer.jupyter.org/.

# Help and Support

Additional documentation, examples and code revisions are coming soon.

## Documentation

Fork: https://github.com/cheng338/dwd . Upstream: https://github.com/slicersalt/dwd .
The historical example figures/notebooks below are upstream artifacts, not
new validation results for the corrected solver.

## Testing

From a checkout, install the test extras and run the self-contained regression
suite (no external datasets or neighboring folders required):

```shell
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests cover the scalar loss/gradient, preserved legacy trajectories, independent
augmented-system checks of corrected updates, kernel/CV behavior, and optional
SOCP utilities. Optional-dependency tests skip when those dependencies are absent.
Passing these tests is not a guarantee of improved accuracy on a new dataset.

## Contributing

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.

[iain-carmichael]: https://idc9.github.io/
[kitware]: https://kitware.com/

[marron-et-al]: https://amstat.tandfonline.com/doi/abs/10.1198/016214507000001120
[wang-zou]: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12244

[dwd_sep_hyperplane]: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/dwd_sep_hyperplane.png
[kern_dwd]: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/kern_dwd.png
[example-notebooks]: https://github.com/idc9/dwd/tree/master/doc/example_notebooks
