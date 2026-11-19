# Fitted state after an error

A classifier becomes usable for prediction only when `fit` completes
successfully. Starting another fit discards its previous learned state. If
validation, optimization, result handling or cross-validation fails, learned
state is cleared again before the exception propagates.

This contract prevents a failed refit from combining old coefficients with new
class labels, or exposing an old `best_estimator_` through a failed CV wrapper.
It applies to `GenDWD`, `GenDWDCV`, `KernGDWD`, `KernGDWDCV`, optional conic `DWD`
and `SVM`, and `KernMD`. After an error, `predict` and `decision_function` raise
`NotFittedError`. A later successful `fit` makes the object usable again.

Public learned attributes follow the scikit-learn trailing-underscore
convention. Kernel training rows are cleared with that learned state.
Constructor parameters and private, validated precomputation caches are
preserved; their existing data and parameter checks still govern reuse. Fit
signatures are retained for scikit-learn inspection and external GridSearchCV.

The change concerns estimator lifecycle only. It does not change the loss,
unregularized intercept, regularization scale, MM equations, optional SOCP
objective, convergence thresholds or finite-iteration predictions after a
successful fit. No failed model is converted into a valid fallback classifier.

Regression coverage includes changed labels, early parameter errors, failures
after fitting has started, partial result state, successful refits, private
cache reuse, and the existing public estimator signatures.
