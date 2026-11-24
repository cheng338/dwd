Welcome to dwd's documentation!
===============================

This package provides linear and kernel distance-weighted discrimination
with scikit-learn-compatible estimators. The kernel estimator offers checked
full-kernel numerical backends with an unregularized intercept. Reference
computation is private validation machinery, not a public estimator option.

See the `kernel guide <https://github.com/cheng338/dwd/blob/v2.0.2/docs/kernel_dwd.md>`_
for objectives, parameters, stopping rules, and numerical controls, and
`installation and examples <https://github.com/cheng338/dwd/blob/v2.0.2/README.md>`_
for usage with the 2.0.2 release.

For shared-fold searches, including process-worker handling in DWD 2.0.2, see ``docs/shared_fold_search.md`` in the source archive or checkout
that matches your installed DWD build.

For decisions about supported optional settings, see
`choosing options <https://github.com/cheng338/dwd/blob/v2.0.2/docs/choosing_options.md>`_.
Maintainers investigating equations and numerical regressions can use the
`private corrected-reference guide <https://github.com/cheng338/dwd/blob/v2.0.2/docs/developer_reference.md>`_.
The private guide does not add a public solver option.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   solvers

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
