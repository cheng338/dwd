# DWD 1.3.2 validation

The parent validated the prepared source and the isolated, actual built wheel.
Package/test hashes and the numerical runtime were unchanged during every listed
suite. This file records current 1.3.2 acceptance; older evidence is labeled below.

| Current acceptance | Result |
|---|---|
| Source suite (`source-suite-v2.json`) | 324 passed, zero skips/failures/errors |
| Actual-wheel suite (`wheel-suite.json`) | 324 passed, zero skips/failures/errors |
| Actual wheel with CVXPY blocked (`base-wheel-suite.json`) | 318 passed, six expected optional skips; no CVXPY import |
| Imported path and package/test identity | Verified against intended source or isolated wheel |
| Generic-CV score regressions | Included in both full suites: invalid scores fail before refit; supported finite scores retain selection behavior |
| Distribution integrity | Assembly requires source/wheel/RECORD/sdist/archive and copied-evidence checks before delivery |

The wheel SHA256 is `aad920b143b206efd596a22c8b241332de6921b2941fd6dd4c2566c276affd1f`. The builder extracts it without installing
into a user environment. Assembly preserves the tested wheel and initial sdist;
only the sdist is refreshed in a separate directory to include finalized release
status documents. Delivered `validation/input-map.json` links receipt/helper
copies to their original paths and hashes. The source, fixture and distribution
manifests bind all delivered bytes.

The three acceptance suites used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0,
scikit-learn 1.9.0, and the updated native MKL runtime reporting 2025.3-Product.
Their exact DLL/extension and OpenMP fingerprints are preserved in the suite
receipts. Optional base-wheel testing explicitly blocked CVXPY; it was not a
different optimization implementation or an environment installation.

## Targeted change evidence

The original generic-CV defect was reproduced using a manufactured metadata-only
estimator: a NaN-scored candidate beat a finite 0.75 candidate and was refitted.
That reproduction performed no DWD/SVM optimization or official-data access.
The new tests cover NaN, positive/negative infinity, invalid scalar types/shapes,
overflowing score means, supported finite values, first ties and no refit after
invalid scores. Exceptions raised by the scorer itself still propagate.

The performance change is solely `_fsum(high.tolist() + low.tolist())` in place
of list expansion of the same two ordered float64 arrays. A separate prototype
used three saved full models and 1,000 training queries per model, four BLAS
threads, warmups and six paired timings. It performed no model fit or official
test access. Every returned decision-score array had identical bytes.

| Saved model | Rows using compensation | Original median | Converted median | Time reduction |
|---|---:|---:|---:|---:|
| 3-8 reference | 513 | 0.663190 s | 0.559000 s | 15.7% |
| 3-8 optimized | 495 | 0.642350 s | 0.545819 s | 15.0% |
| 2-3 optimized control | 0 | 0.180227 s | 0.178522 s | 0.9% |

These prototype timings support this allocation change on the measured saved
queries. They are not a new actual-wheel whole-fit benchmark or evidence of
improved accuracy. The actual wheel's current suite validates the combined
package. Solver mathematics, stopping, initialization and thresholds did not
change in 1.3.2.

## Preserved scope and limits

The historical [1.3.1 validation](docs/history/VALIDATION-1.3.1.md) retains its
318 full-suite passes, 312 base passes plus six optional skips, and nine known
extreme synthetic RBF rejections in a 30-fit characterization. Those experiments
were not rerun or relabeled here. This point release leaves their solver/recovery
policies unchanged. There is no new arbitrary-input success or classification-
accuracy guarantee. Task6's strict external scorer did not use the generic-CV
selection bug; this fix must not be cited as explaining its accuracy differences.
