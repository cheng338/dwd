DWD Classifiers
===============

.. autoclass:: dwd.socp_dwd.DWD

.. automethod:: dwd.socp_dwd.DWD.fit

.. automethod:: dwd.socp_dwd.DWD.predict

.. automethod:: dwd.socp_dwd.DWD.decision_function

.. autoclass:: dwd.gen_dwd.GenDWD

.. autoclass:: dwd.gen_kern_dwd.KernGDWD

The classifiers inherit scikit-learn's
`accuracy score method <https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin.score>`_
and `metadata-routing interface <https://scikit-learn.org/stable/metadata_routing.html>`_.

Training ``sample_weight`` is not supported and is no longer an accepted fit
or standalone-solver argument, even when its value is ``None``. Weighted
accuracy through ``score(..., sample_weight=...)`` remains supported. See the
`fitting and conic persistence guide <https://github.com/cheng338/dwd/blob/v2.0.1/docs/fitting_and_conic_persistence.md>`_
for argument boundaries, metadata migration, solver diagnostics and saving
fitted conic problems.

Cross-validation
----------------

.. autoclass:: dwd.gen_dwd.GenDWDCV

.. autoclass:: dwd.gen_kern_dwd.KernGDWDCV
