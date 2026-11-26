DWD Classifiers
===============

.. autoclass:: dwd.socp_dwd.DWD

.. automethod:: dwd.socp_dwd.DWD.fit

.. automethod:: dwd.socp_dwd.DWD.predict

.. automethod:: dwd.socp_dwd.DWD.decision_function

.. autoclass:: dwd.gen_dwd.GenDWD

.. autoclass:: dwd.gen_kern_dwd.KernGDWD

The classifiers' ``score(X, y)`` returns the fraction of correctly predicted
labels and preserves exact class identities, including large integer IDs.
Passing ``sample_weight`` to ``score`` gives weighted accuracy. The classifiers
support scikit-learn's `metadata-routing interface <https://scikit-learn.org/stable/metadata_routing.html>`_.

Omit ``sample_weight`` from fitting and standalone-solver calls, even when its
value is ``None``. See the
`fitting and conic persistence guide <https://github.com/cheng338/dwd/blob/v2.0.2/docs/fitting_and_conic_persistence.md>`_
for argument boundaries, metadata migration, solver diagnostics and saving
fitted conic problems.

Cross-validation
----------------

.. autoclass:: dwd.gen_dwd.GenDWDCV

.. autoclass:: dwd.gen_kern_dwd.KernGDWDCV
