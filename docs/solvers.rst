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

Cross-validation
----------------

.. autoclass:: dwd.gen_dwd.GenDWDCV

.. autoclass:: dwd.gen_kern_dwd.KernGDWDCV
