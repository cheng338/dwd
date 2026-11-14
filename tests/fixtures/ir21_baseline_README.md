# IR-21 pre-change linear solver fixture

`ir21_baseline_gen_dwd.py` is a byte-for-byte snapshot of `dwd/gen_dwd.py`
immediately before the No.21 zero-shift rank-admission integration. It retains
the package source license and historical implementation for regression tests.

SHA-256: `e32956c68e0f259fef3d610e2b778529b42e0e324e2c732f3870dd39bcf608ba`.

`test_zero_shift_design_rank.py` verifies that hash before loading the fixture
under a separate, unregistered test-only module identity. Its classes keep their
own implementation globals and use the current shared package dependencies.
It is not an alternative public solver or an installed runtime entry point.
Update this fixture only as an explicit change to the regression oracle.
