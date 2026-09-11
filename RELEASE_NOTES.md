DWD 1.3.8 repairs two numerical checks while preserving the generalized kernel DWD objective and unregularized intercept.

- Constrained-system acceptance now accounts for floating-point error in ordinary products and correction trials. Inconclusive cases can use bounded block products before existing compensated recovery.
- Dual diagnostics now enforce exact signed mass balance and use conservative objective bounds. A valid weaker fallback prevents an unreliable diagnostic from falsely rejecting a fitted model.
- Original-package repairs and retained recovery mechanisms are documented individually, with independent failure cases, regression evidence and remaining limits.

Local validation passed **467 native tests** and **458 portable tests**, with **9 expected native-only skips** and no failures. Six paired full-kernel MNIST regressions across 2 vs 3, 4 vs 9 and 3 vs 8 retained the same held-out labels and iteration counts.

The new checks have a measured cost: controlled full-kernel 2-vs-3 fits took approximately **2.1 times** the optimized pre-check baseline at 2,400 and 6,000 training rows. This is a correctness release, not a claimed speedup over that baseline. Default objective stopping and the 100-update limit per attempt are unchanged. Extreme valid problems can still exceed bounded numerical recovery capacity.

See the [validation and justification report](https://github.com/cheng338/dwd/blob/v1.3.8/docs/validation-1.3.8.md) for the evidence matrix, test protocols, timings and scope.
