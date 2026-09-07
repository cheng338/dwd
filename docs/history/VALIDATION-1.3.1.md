> Historical 1.3.1 record. The counts, artifact hashes and measurements below
> belong to 1.3.1; they do not assert 1.3.2 validation.

# DWD 1.3.1 validation

**Local source and actual-wheel validation passed.** Checks used the requested
Windows Anaconda `py12_df` environment. The tested wheel SHA256 is
`4debfab97249264a85aad17dccdf16cf6ad3a70770eb615599dd196a72cb4e13`.
Sealed 1.3.0 artifacts and historical measurements remain separate. The source
distribution is refreshed for these final status documents without rebuilding
the tested wheel; assembly verifies source, wheel, RECORD, sdist and archive bytes.

| Acceptance item | Status |
|---|---|
| Complete source and actual-wheel suites | 318 passed each; no skips. |
| CVXPY-blocked source and wheel | 312 passed each; six expected optional skips; no CVXPY import. |
| Actual-wheel Pipeline/GridSearchCV | Both repaired implementations, eight candidates, three folds, two jobs; independent CV scores and selected refit agree. |
| Documentation examples | All seven README/guide Python examples pass against the actual wheel. |
| Original-script interfaces | Twelve synthetic direct-fit/CV groups pass against the actual wheel; original wrappers retain the caveats below. |
| Exact-factor/MM transaction and adaptive prediction | Included in the complete suites; portable fixed oracles cover recovery, stopping, observers, state restoration and prediction. |
| Known extreme public synthetic kernels | Actual wheel: 21/30 return and pass Decimal100 score/objective and serialization checks; nine nearly constant RBF fits remain rejected. |
| Paired whole-pipeline cost check | 24/24 fits pass; all 12 source-paired state archives have identical NPY bytes. Measured production modules match the wheel except its version literal. |
| Distribution provenance | Source/module/test manifests, wheel RECORD and copied artifact/source bytes are checked by the assembly helper. |

## Scope of retained numerical evidence

The actual 1.3.1 wheel passed its full and optional-dependency-blocked suites
and repeated the public synthetic characterization. The replay uses exact preserved float64
kernels, the same labels/regularization/initialization, and a five-update fixed
budget. Independent Decimal100 evaluations check returned original-kernel scores
and objectives. All 18 exact rank-five cases complete; nine nearly constant RBF
cases still reject. These matrices are not MNIST and their arbitrary private
linear-system right-hand sides must not be confused with actual public DWD
trajectories. Rank-budget failure is not proof that a model is mathematically
unrepresentable. All 30 outcomes are retained. No returned state failed the
independent checks: maximum score error was 2.27e-13 and objective error 1.15e-13.

The exact identity certificate has bounded rational operations and intermediate
bits. The ensuing MM gates assess the actual exported function rather than an
unnecessary singular coefficient gauge. Other eigen, scalar, solve and query
checks are numerical consistency estimates under their floating-point model;
they are not universal interval certificates or promises of success for all
valid inputs. Dense memory and recovery cost remain relevant.

## Performance scope

The completed comparison uses the existing 3,000-fit/1,000-development MNIST rows,
both implementations, lambda values 1e-8 and 1e-5 at gamma .001, and three fresh
same-seed repetitions per source. Whole Pipeline.fit and Pipeline.predict are
timed separately, with standardization inside each fit, four BLAS threads and
serial monitored workers. It is a fixed Task 4 cost check, not CV, new parameter
selection or an official-test evaluation. The baseline is the frozen range-MM
candidate, and the proposal includes same-state objective reuse plus the bounded
dense query shortcut; neither row denotes raw Slicersalt or sealed 1.3.0.

| Implementation | lambd | Baseline fit (s) | Proposal fit (s) | Baseline predict (ms) | Proposal predict (ms) | Development accuracy, both |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 1e-8 | 2.2659 | 2.2702 | 50.83 | 43.64 | 98.3% |
| Reference | 1e-5 | 2.2952 | 2.2529 | 48.22 | 44.40 | 97.7% |
| Optimized | 1e-8 | .7626 | .7532 | 47.59 | 43.53 | 98.3% |
| Optimized | 1e-5 | .7756 | .7562 | 48.58 | 45.20 | 97.7% |

Entries are medians of three same-seed repetitions. Prediction improvements in
these four cells are 3.4–7.2 ms (7.0–14.1%); whole-fit differences are small and
mixed. Each source-paired state NPZ has identical uncompressed NPY bytes, including
dtype, shape, signed-zero values, coefficients, objective history, scaler and row
IDs. All fits stop at cap100, with `converged_=False`, and none activates exact-MM
recovery. Maximum sampled worker-plus-descendant RSS is .515 GiB; all source,
runtime, input and resource receipts pass. This does not establish a large or
universal training speedup or independent statistical accuracy improvement.

The whole-pipeline source snapshot precedes final 1.3.1 metadata and documents.
The final integration receipt verifies that all production bytes match except
the version literal in `__init__.py`; the sole test difference is its matching
version assertion. These are source timing measurements, not actual-wheel timing.
The measured receipt is
`whole-pipeline-final-prototype.json`, with independent exact-byte and hash review
in `whole-pipeline-verification.json` in the Task 4 performance evidence.

Earlier saved-query component profiles use fixed kernel/alpha arrays and exclude
kernel construction. Their weak case uses historical coefficients on preserved
training-kernel rows, whereas their medium case uses disjoint development rows.
Small synthetic objective-call profiles have a different scope again. None is
a replacement for final complete-pipeline timing or evidence of improved
classification accuracy. Historical 1.3.0/1.2.1 figures retain their original
source versions and runtime attribution.

## Script, CV and runtime boundaries

The script controls use AST-extracted original class bodies on 72 synthetic rows;
they do not import the data loaders or run either MNIST main. Original `binary.py`
still has a classifier-tag/extra-Pipeline fitted-state limitation, and both original
ensembles couple subset sampling to global RNG and treat M=1 differently. Task 6
uses its separately checked external consensus wrapper. Replacing the DWD package
does not itself edit or repair those original script wrappers.

The CV split/grid/refit/cache algorithms retain their previously tested behavior;
`KernGDWDCV.fit` additionally copies the selected model's prediction precision.
Current internal-CV regression and external GridSearchCV checks pass. Task 5's
engine-timing experiment was not repeated or relabeled as a 1.3.1 benchmark.

Checks used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0,
CVXPY 1.8.2 when enabled, and the separately updated native MKL 2025.3.1 /
LLVM OpenMP 22.1.2 runtime. Receipts include loaded-library paths and binary hashes.
The base-only variant uses a CVXPY import blocker in this same environment; it is
not a separately provisioned environment. Exact arithmetic adds no dependency.
The defined Windows/Linux CI jobs were not run on hosted infrastructure here.

The first 1.3.1 suite attempt exposed only an outdated test assertion expecting
the 1.3.0 version string. Its failure receipts are preserved; updating that sole
assertion produced the complete passing source and wheel suites above. No numerical
behavior was changed for that test correction.

The release preserves the full-kernel objective, unregularized intercept, native
initialization policies, ordinary-MM default, objective threshold 1e-5 and cap100.
No package installation into the user environment, GitHub upload, publication,
dissertation edit or original-script edit occurred during this release work.

Task 6 broad comparison has not started. It must use the final verified frozen
repaired source and its separate protocol; no Task 6 result is claimed here.
