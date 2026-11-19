# Shared feature folds for serial KernMD searches

`dwd.model_selection.fit_shared_fold_search` is an explicit optimization for
repeated `KernMD` fits on the same feature folds. It snapshots the input and
caches immutable training and validation subsets across candidates, avoiding
repeated feature indexing and ownership copies. Exact contiguous row selections
automatically use private read-only views of the input snapshot; other selections
keep the original indexed-copy path. Kernel construction and solver arithmetic
remain unchanged. The ordinary `KernMD.fit` ownership protection remains in place,
including for the final fit on all training rows.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dwd.kern_md import KernMD
from dwd.model_selection import fit_shared_fold_search

# X: finite, two-dimensional float64 ndarray; y: finite numeric labels.
# Any preprocessing learned from data must still be fitted within each fold.
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
search = GridSearchCV(
    KernMD(kernel='rbf'),
    {'kernel_kws': [{'gamma': float(g)} for g in np.logspace(-3, 1, 8)]},
    cv=folds, n_jobs=1, scoring='accuracy', refit=True,
    error_score='raise',
)
fit_shared_fold_search(search, X, y)  # Returns this same search object.
model = search.best_estimator_       # An ordinary KernMD instance.
print(search.shared_fold_info)      # Route and cache accounting.
```

The helper does not learn preprocessing or transform features. If preprocessing
needs fitting within each fold, retain it inside a scikit-learn pipeline; that
configuration uses the ordinary search route.

## Eligible configurations and fallback

Shared snapshots apply only to an exact `GridSearchCV` with an exact `KernMD`,
`n_jobs=1`, `scoring=None` or `'accuracy'`, and Boolean `refit`. Input must be an
ordinary finite float64 ndarray with nonempty two-dimensional shape, with an
ordinary one-dimensional finite numeric label array. C-contiguous, Fortran,
strided and read-only input layouts are supported.

Supply folds explicitly as a list or tuple of pairs of nonempty integer ndarray
indices. Order and duplicate indices are retained. Supported kernels are the
named linear and RBF kernels, with no keywords or only a positive finite ordinary
Python numeric RBF `gamma`. Grid entries must remain within these settings, with
`naive_bayes=False`. Parameter-grid dictionaries use list or tuple values.

Other configurations delegate to `search.fit(X, y, **fit_params)` using the
original inputs. These include pipelines, custom estimators, custom scorers,
callable or precomputed kernels, parallel searches, automatic fold generators,
custom array types, float32 features, metadata routing and fit parameters. Their
usual validation and exceptions are preserved. An object that is not a
`GridSearchCV` instance raises `TypeError`.

This is a serial helper. It does not make concurrent changes to input labels,
fold definitions, search parameters or private Python attributes safe.

## State and ownership

Scikit-learn still performs candidate evaluation, scoring, ranking and refitting.
Numerical score and selection results retain their ordinary meaning; timing
columns naturally include different indexing and copy work.
Snapshots use C order, which can change floating-point reduction order for
other input layouts. Bitwise equality across arbitrary layouts or backends is
not promised.

A private view is used only when the entire requested index sequence is exactly
an increasing contiguous span of the helper's immutable root snapshot. Reordered,
duplicate or separated rows keep the original gather-and-snapshot behavior,
preserving the requested row order. This handling is automatic within the
explicit helper; it adds no public setting or eligibility requirement.

The helper restores `search.estimator` after success or failure. A successful
refit returns an ordinary `KernMD` that owns its feature basis and supports normal
serialization. With `refit=False`, the ordinary search behavior is retained.
Private search registration is cleared after success or failure. After an
exception, scikit-learn's partial fitted state may remain, as with an ordinary
search; the helper does not claim that a failed search is usable.

A private contiguous view deliberately retained outside the helper can keep the
entire immutable root alive, even for a small subset. The normal returned public
final estimator instead owns independent feature data; it does not borrow that
private root.

The private fold-token registry remains available through the ordinary final
refit. The helper records cache statistics before clearing that registry during
its exit cleanup, on success or failure. The full input snapshot and cached
subsets can therefore overlap the final estimator's owned feature copy.
Clearing the registry releases its references; it does not force garbage
collection or guarantee immediate reclamation of externally retained references
or allocator storage. Fold evaluation, selection and final-model ownership are
unchanged.

`search.shared_fold_info` is diagnostic metadata, deliberately without the
trailing underscore used for learned estimator state. Inspect `route`, `reason`
and `completed` to distinguish a shared search, a stock fallback and a failed
call. Shared-route counters describe cache use during that call. `root_bytes`
and `cached_bytes` count logical feature payload: the full snapshot and the
distinct cached subsets, respectively. A cached view can share root storage,
so adding these counts can count the same backing bytes more than once. They
measure neither unique allocation, current retained memory nor peak process
memory. The completed statistics remain available after the registry is cleared;
they continue to describe the preceding fold evaluation, not its live storage.

## When to use it

Use it when repeated copying of wide feature folds is a material part of serial
search time. Ordinary `GridSearchCV` remains the general path: ordinary or
moderate-sized searches can be faster without the helper, and short grids need
not benefit. Keep the folds required by the statistical design; do not change
them to create contiguous spans. Measure the complete helper call on
representative data and grids before choosing it.

A later direct comparison on 2026-11-09 measured the current helper against
ordinary `GridSearchCV.fit` using the same current `KernMD` implementation.
Both routes retain fitted-data ownership and numerical safeguards. The helper
saves repeated feature copies, but immutable snapshot construction and indexing
can outweigh those savings. These measurements used the registry lifetime
retained here: cached subsets survive through final refit until helper cleanup.
They are historical results, not a new measurement of the lifetime reversal or
of later source changes.

The following selected workloads used float64 data, three fixed folds,
`n_jobs=1`, accuracy scoring, final refitting and `return_train_score=True`.
Each entry is the geometric mean of paired block time ratios (helper / ordinary
search), ordered **prepared-first / warm-1 / warm-2**. A ratio below 1 favors
the helper. The positions and hosts are separate comparisons. The Windows environment
loaded AOCL-BLAS 5.3.0 and MKL 2026.2. The Linux environment used MKL 2026.2.
The library inventory does not identify the provider of each operation.

| Rows x features; kernel; grid candidates | Windows, 16 native threads | Linux, 20 native threads |
| --- | --- | --- |
| 96 x 4,096; linear; 2 | 0.905 / 0.832 / 0.877 | 1.159 / 1.098 / 1.138 |
| 1,024 x 64; RBF; 2 | 1.026 / 1.033 / 1.032 | 1.109 / 1.224 / 1.094 |
| 1,536 x 32; RBF; 4 | 1.140 / 1.130 / 1.231 | 1.035 / 1.030 / 1.048 |
| 32 x 65,536; RBF; 8 | 0.684 / 0.701 / 0.608 | 0.792 / 0.735 / 0.742 |

Each position uses six qualified paired blocks, except the eight-candidate
wide-RBF row on both hosts and the Linux four-candidate RBF row, which use five.
Original timing exclusions were retained without replacement. Prepared-first
followed reference preparation; warm calls still created fresh private caches.
These finite observations do not establish statistical significance, cold-start
performance, or results for `return_train_score=False`. That setting needs its
own representative whole-search measurement when choosing between routes.

For these measured training-score configurations, prefer `search.fit(X, y)`
for the Linux wide-linear case, the 1,024-by-64 RBF case on both hosts and the
Windows 1,536-by-32 RBF case. The wide-RBF grid benefited from the helper on both
hosts. Size, kernel and host labels alone are not automatic routing rules:
measure the actual data, grid and scoring settings. Retain the existing helper
as an explicit choice; these results require no estimator or search default
change. They do not attribute the differences to an AOCL or MKL version upgrade.

Earlier studies used two separate comparisons. The snapshots-only predecessor
was compared directly with ordinary search. The added contiguous-row
optimization was then compared directly with that predecessor. At that stage,
the updated helper had not been directly compared with ordinary search in the
same panel. The historical percentages below must not be combined into such an
estimate or pooled with the later direct comparison.

- **Snapshots-only predecessor versus ordinary search:** for a 32-row,
  65,536-feature float64 RBF search with three fixed folds and eight candidates,
  warm whole-search time was about 28.3% lower on Windows and 24.5% lower on Linux.
  In contrast, an ordinary 1,024-row, 64-feature RBF search with two candidates
  was about 2.0% slower on Windows and 14.8% slower on Linux when warm. A Windows
  1,536-row, 32-feature RBF search with four candidates was about 12.7% slower
  when warm.
- **Added contiguous-row optimization versus the snapshots-only predecessor:**
  original Windows contiguous wide-linear first/warm times improved by about
  9.4%/18.1%, and contiguous wide-RBF eight-candidate times by 17.5%/18.5%.
  First-search time for contiguous wide-linear folds decreased by 8.4% on Windows
  with MKL 2026.2, by 11.4% on WSL2 with MKL 2026.2, and by 14.5% on WSL2
  with AOCL-BLAS 5.3.2 and MKL 2026.2 loaded. Warm results were mixed. These panels shared one physical host and did not repeat ordinary,
  moderate or wide-RBF cases.
- **Retained costs of the added optimization:** the original Linux shuffled
  wide-linear first search was 12.85% slower (+11.216 ms), and the Windows ordinary
  shuffled warm search was 2.62% slower (+1.863 ms). Other ordinary, layout and
  first/warm results were also mixed. Later measurements do not erase these
  observations or establish a no-slowdown guarantee.

Whole-search timings include setup, snapshots, scoring, final refit and cleanup.
"First" means the first search in a prepared runtime, not an untouched cold
start; warm searches still build fresh caches. These are separate
workload/environment measurements with finite uncertainty, not a universal
speedup or a causal comparison of numerical-library vendors.

The helper keeps a full input snapshot and copied noncontiguous subsets;
contiguous views share the root. It can still increase simultaneous feature
storage, with kernel matrices and solver storage as additional costs. The
three-fold 32-by-65,536 workload has 16 MiB of logical root payload and 48 MiB of
logical cached-subset payload; those counts do not establish peak memory use.
Ordinary `KernMD.fit`, `GridSearchCV.fit`, and the separate `KernGDWD` CV APIs
keep their existing defaults.
