# Frozen performance regression oracle

These two small source snapshots preserve the pre-performance numerical route.
They are loaded only by `test_performance_controls.py`, under private test-only
module names. Relative numerical-helper imports use the package being tested.
This keeps score routing and duplicate-objective recomputation independently
reviewable without requiring a historical checkout or external data. The pinned
hashes are in `provenance.json`. Do not update these snapshots merely to match a
new implementation; investigate a behavior difference first.
