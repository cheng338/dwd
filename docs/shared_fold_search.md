# Shared feature folds for serial KernMD searches

`dwd.model_selection.fit_shared_fold_search` is an explicit optimization for
repeated `KernMD` fits on the same feature folds. It snapshots the input and
caches immutable training and validation subsets across candidates, avoiding
repeated feature indexing and ownership copies. Exact contiguous row selections
automatically use private read-only views of the input snapshot; other selections
keep the original indexed-copy path. Kernel construction and solver arithmetic
remain unchanged. The ordinary `KernMD.fit` ownership protection remains in place,
including for the final fit on all training rows.

These examples give the search an explicit CV worker count. Its parent process
uses one default estimator job and one native thread for sklearn's timed refit.
When adapting them to another estimator, set any explicit model-worker count
to one, including `kernel_kws['n_jobs']`. A solver with its own thread pool
also needs its own one-thread setting, such as Clarabel's `max_threads=1`.

```python
import numpy as np
from joblib import parallel_config
from threadpoolctl import threadpool_limits
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dwd.kern_md import KernMD
from dwd.model_selection import fit_shared_fold_search

X = np.random.default_rng(42).normal(size=(24, 4))
y = np.repeat([0, 1], 12)
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
search = GridSearchCV(
    KernMD(kernel='rbf'),
    {'kernel_kws': [{'gamma': float(g)} for g in np.logspace(-3, 1, 8)]},
    cv=folds, n_jobs=1, scoring='accuracy', refit=True,
    error_score='raise',
)
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    fit_shared_fold_search(search, X, y)  # Returns this same search object.
model = search.best_estimator_       # An ordinary KernMD instance.
print(search.shared_fold_info)      # Route and cache accounting.
```

The helper does not learn preprocessing or transform features. If preprocessing
needs fitting within each fold, retain it inside a scikit-learn pipeline; that
configuration uses the ordinary search route.

## Eligible configurations and fallback

Shared snapshots apply to an exact `GridSearchCV` with an exact `KernMD`,
the explicit Python integer `n_jobs=1`, `scoring=None` or `'accuracy'`, and
Boolean `refit`. The default `n_jobs=None` takes the ordinary search route
even when it runs serially. Use plain NumPy ndarrays for features, labels
and indices. Supply a nonempty two-dimensional array of finite native-endian
float64 features and a one-dimensional array of finite real numeric or
Boolean labels, one per row. C-contiguous, Fortran, strided
and read-only feature layouts are supported. `KernMD` requires exactly two
classes in each training fold and in the full training set for a final refit.

Supply folds in a plain list or tuple. Each pair must itself be a plain
list or tuple of two nonempty, one-dimensional integer ndarrays with indices
in `[0, len(X))`. Order and duplicate indices are retained. Supported kernels
are the named linear and RBF kernels, with no keywords or only a positive finite
ordinary Python numeric RBF `gamma`. Grid keys are limited to `kernel`,
`kernel_kws` and `naive_bayes`, with `naive_bayes=False` throughout.
Parameter-grid dictionaries use list or tuple values.

Other configurations delegate to `search.fit(X, y, **fit_params)`.
These include pipelines, custom estimators, custom scorers,
callable or precomputed kernels, parallel searches, automatic fold generators,
custom array types, float32 features, callbacks, enabled Array API dispatch or
metadata routing, and fit parameters passed to the helper.
Scikit-learn validates the search parameters and applies its scoring-error policy.
The helper raises `TypeError` for objects other than `GridSearchCV` instances.

## Process-parallel fallback

DWD 2.0.2 implements the worker handling described
below and requires joblib 1.6.0 or later. DWD 2.0.1, whether installed from
a released wheel or its source archive, passes fallback searches directly to
`search.fit`.

When the search is configured to use multiple worker processes on entry to the
helper and uses the stock `GridSearchCV.fit` method, the helper snapshots
features, targets and direct array-valued fit parameters backed by copy-on-write
(`mode='c'`) memory maps. These copies cover inputs of exact type `numpy.ndarray`
or `numpy.memmap` whose dtype contains no Python objects.
Workers reopening the file would otherwise miss edits private to the
parent process. This applies to one-dimensional arrays and to strided and
read-only views of a private mapping.
The helper also copies contiguous arrays whose C or Fortran order differs from
their backing mapping, as can happen after transposition. The helper passes
these copies into `search.fit`, so final refitting in the parent can also
receive detached `X`, `y` or fit parameters. If an estimator mutates or retains
one of these arrays, writes through the copy are independent of the caller's
mapping. Inputs that originally shared storage can receive separate copies.
If final refitting relies on that shared storage, use a direct serial fit
with the original arrays.
Set the intended worker count before calling the helper. If a setup callback
switches a serial search to process workers, make owning copies of affected
mapped NumPy `X`, `y`, and direct array-valued fit parameters before calling the
helper. It prepares supported explicit fold indices when scikit-learn selects
the executor for candidate evaluation.
On the fallback path, if you override or wrap `fit`, the helper calls that
implementation. With process workers, give it owning copies of affected
arrays and fold indices unless the override handles their transport. The process count follows
joblib's integer conversion of an inherited `parallel_config(n_jobs=...)`
setting.

**Pandas feature tables.** With the stock fit method and process workers,
the helper prepares the affected storage in an exact pandas DataFrame supplied
as `X`. Even a read-only NumPy mapping can be affected because pandas may store
a transposed view. The splitter and final refit in the parent receive the
caller’s original frame. The helper uses the executor scikit-learn actually
selects, including when a callback or splitter changes the worker count during
setup. If process workers are already requested, a file-backed object buffer
can be rejected before the splitter counts the folds. If fold counting selects
process workers, the helper checks before calling `split()`. Keep the data
stable once worker tasks start.

This covers NumPy blocks, pandas nullable numeric and Boolean values and masks,
categorical codes and categories, native datetime, timedelta and period data,
interval endpoints, and pandas sparse values and positions. It also covers
ordinary numeric, categorical, interval, datetime-like and multilevel row
and column indexes, including the stored frequency of datetime and timedelta
indexes. Keep object references in memory. Other extension
storage, custom indexes and objects in `DataFrame.attrs` follow their own
serialization rules. This handling has been checked with pandas 2.2.3 and 3.1.0.

**Sparse integer class IDs.** If a pandas sparse target would round class
IDs during validation or fold selection, the helper asks you to prepare an
exact target before fitting. For a Series with stored `uint64` values and a
zero fill, use the original data to make a dense integer Series. This also
works for a direct scikit-learn search.

```python
y_exact = pd.Series(y.to_numpy(dtype=np.uint64, copy=True), index=y.index, name=y.name)
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    fit_shared_fold_search(search, X, y_exact)
```

Categorical targets can store signed or unsigned integer levels in a sparse
index, including an index with `object` storage. A level can also come from
an integer-valued fill, even if the stored entries are `float32`. Some pandas versions accept
`Fraction` and `Decimal` fills that preserve a large integer in a Series
but round it when a frame combines the Series with a second output. The
helper checks conversions of observed levels, including folds that omit
other types of labels. For example, a fold containing only intervals can
round a large integer endpoint when another interval has floating-point
endpoints. When a sparse column or its category dictionary stores dates or
durations, its fill can have finer precision than its stored values. For
example, a nanosecond dictionary can supply a picosecond timestamp by scalar
lookup, then round it when pandas converts a fold to an array. In a sparse
object column, the order of dates in a validation fold can also affect how
pandas interprets a value stored in two-day ticks.
If pandas also raises while reading an original scalar, rebuild the target
from the source labels with their original units.
For mixed labels, including NumPy timestamps with picosecond precision, use
`pd.Series([y.iat[i] for i in range(len(y))], dtype=object, index=y.index, name=y.name)`
if the estimator and scorer accept those objects. For a categorical Series whose observed
labels are all `uint64` integers, read its values by scalar position to
prepare an owning target.

```python
ids = np.fromiter((y.array[i] for i in range(len(y))), dtype=np.uint64, count=len(y))
y_exact = pd.Series(ids, index=y.index, name=y.name)
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    fit_shared_fold_search(search, X, y_exact)
```

For a multi-output target, check the other columns as well. The categorical
IDs can stay exact while pandas rounds a label in a second sparse column
during row selection or conversion of the whole frame to one array.
For example, a sparse object column can contain intervals and a text label.
Selecting only the intervals can convert their endpoints to floating point.
An integer endpoint of `9007199254740993` then becomes `9007199254740992`.
A Python integer endpoint such as `2**1024` can also raise an overflow error
when pandas selects that row, even if it can read the whole column.
If the helper rejects either case, build the owning frame shown below from
the original interval objects.
For example, a dense `int64` column next to a `uint64` column can give the
frame a common `float64` dtype. The same conversion can first happen inside a
fold if selecting a sparse column with a Python integer fill changes its dtype.
An exactly representable integer such as `2**54` survives this conversion,
while `2**53 + 1` rounds to `2**53`. This can change which candidate wins.
Other sparse columns can carry labels in their fills. For example, a
Python float fill may contain an integer that the column's stored `float32`
dtype cannot represent. Even `2**54 + 4`, which is exact in a Python float,
can become `2**54` in that conversion. Read the fill from the original column as well.

When every target column contains only integer labels in the `uint64` range,
construct all columns with that same owning dtype. For mixed types and an
estimator and scorer that accept object targets, build an owning object frame
from the original scalars. Position-based access preserves duplicate column
names and their order.

```python
data = np.empty(y.shape, dtype=object)
for i in range(y.shape[1]):
    column = y.iloc[:, i].array
    data[:, i] = [column[row] for row in range(len(column))]
y_exact = pd.DataFrame(data, index=y.index, columns=y.columns, dtype=object)
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    fit_shared_fold_search(search, X, y_exact)
```

Use the same target when constructing any folds or scorers that retain it.
If they use additional target metadata, carry that metadata over as well.
Keep the target's values, sparse fill value and fill type stable from search
entry through scoring and refitting, including in custom splitters and callbacks.
Converting an already rounded floating-point value back to an integer cannot
recover the original class ID.

For a sparse `uint64` column in an exact pandas DataFrame, the stock search inside the helper
protects the integer values when pandas selects rows for a fold. Numerical
estimators can still convert features to floating point when fitting. For a
direct scikit-learn search, a DataFrame subclass, or an overridden search
`fit` on pandas versions with lossy row selection,
convert each affected zero-filled column to dense `uint64` in a copy of the
original frame before constructing the search.
If a sparse column mixes integers with a text fill, use an owning object
Series for estimators that support both types. For example,
`pd.Series(list(s.array), dtype=object, index=s.index, name=s.name)` preserves
the distinction between the integer `9007199254740993` and the same digits as text.

For DWD 2.0.1 or a direct process-parallel `GridSearchCV.fit`, construct an
owning frame from homogeneous numeric mapped features after all private edits,
for example `X = pd.DataFrame(mapped, index=row_labels, columns=feature_names, copy=True)`
with in-memory labels. With either the helper or a direct fit, use that same
owning frame wherever a scorer, estimator or splitter keeps its own reference
and needs to see those values in a worker.

When callbacks are configured or propagated from a parent estimator, the
callbacks, splitter and estimator can save an otherwise serializable search
during callback setup, split counting or cloning of the base estimator.
At those stages the search uses its own pickle protocol and callback state.
If a search subclass uses a property or another descriptor to select
`_run_search` at lookup time, scikit-learn still selects and calls that method.
The helper prepares supported explicit fold indices for the selected process
workers without assigning a replacement method to the search.
This preparation uses scikit-learn’s internal candidate evaluator. If the
helper reports an unrecognized evaluator, use a direct serial search or make
owning copies of the affected inputs and folds before fitting directly with
process workers.

The helper checks the outer fold container before scikit-learn collects it.
When callbacks select new folds, it checks the selected container with the
worker settings in effect at that point. It prepares the explicit fold rows
using the executor scikit-learn selected for candidate evaluation.
A nested joblib context inside the search can change the settings for new
parallel work. Jobs submitted to the existing executor still use its backend.

**Reversed mapped views.** A worker reopening a NumPy memory map can read
incorrect values or fail if a view has a negative stride on an axis of length
greater than one, as in `mapped[::-1]`. With process workers configured on
entry, the stock fallback copies affected NumPy `X`, `y` and direct
array-valued fit parameters. If a setup callback enables process workers later,
make those copies before calling the helper. For supported explicit fold arrays,
the helper checks the executor selected for candidate evaluation. This handling
includes read-only and read/write maps and ordinary ndarray views backed by
them. Serial fits and threading backends require no copy for process transport.

For explicit folds, the helper handles iterables of train/test pairs, including
lists, tuples, reusable containers and one-pass iterators such as
`zip(train_indices, test_indices)` or `KFold.split(X, y)`. Pairs can also
use named tuples and other containers that unpack into train and test indices.
For equal-length index vectors, a numeric array can represent a train/test pair
with shape `(2, n_indices)` or a stack of folds with shape
`(n_folds, 2, n_indices)`. The helper handles reversed mapped views of these
arrays even when the reversal is on the fold or train/test axis. It prepares
the train and test indices used for fitting before dispatch. For a numeric
fold stack or a plain list of numeric pair arrays, it prepares the worker's
`search.cv` only if the search itself is serialized, and reuses the result
within that candidate batch. For a custom fold record, the helper can prepare
its container on the fit thread before dispatch so that copy hooks run there.
If that copy fails, the error surfaces when the serialized search state needs
the prepared public folds. With NumPy fold indices, a search's saving hook
can still return state that omits those folds.
If the record refers back to the active search, the corresponding reference
in the worker points to the worker's copy of that search.
An in-memory object array can hold separate train and test indices.

**Pandas fold selectors.** An explicit train or test selector of exact type
`pandas.Index` or `pandas.Series` can hold integer row positions or a Boolean
mask. On joblib's standard `loky` and `multiprocessing` paths, the helper
checks the selector's stored values and the labels of a Series against the
same mapping rules as a DataFrame. Workers receive the parent values for
affected NumPy buffers, including a nullable integer or Boolean array's data
and mask. A RangeIndex needs no value buffer. Pandas serializes the selector
and its metadata. Objects in `Series.attrs`, custom indexes and other
extension storage follow their own serialization rules.

A worker scorer can refer to the same selected Index or Series through its
own fields and through `search.cv`. On the standard path those references
share the worker selector, and overlapping NumPy views of the affected
storage carry the parent values. The parent keeps its original selectors
throughout fitting.

When a selector buffer needs a snapshot, a search subclass can override
`__getstate__` alone and return `vars(self).copy()`. The returned state must
be a plain dictionary with the same keys as `vars(self)`. Each key must be
a plain `str` and each value must be the same object as its field. If the
hook omits fields, adds derived values or replaces a field with an equal
copy, build the selectors from owning storage. Do the same for a search with
a custom `__reduce__`, `__reduce_ex__` or `__setstate__`.

With DWD 2.0.1, a direct process-parallel fit, or a custom serializer that
prevents the helper from preserving these values, build the selectors from
owning storage after making all private edits. For example, for ordinary
integer positions and numeric labels, use
`pd.Series(np.array(rows, copy=True), index=pd.Index(np.array(labels, copy=True)))`.
Use the same prepared selector in the folds and in any scorer that refers to
it. When you reuse the same private folds for many short searches, preparing
owning selectors once can reduce the time spent sending each search to workers.
If an alternate splitter supplies the fitting rows, also make owning
copies for unrelated public folds that the scorer reads.

A raw NumPy memory map with an object-containing dtype stores Python object
references that belong to the process that created them. Use ordinary
in-memory arrays for object-valued inputs. With process workers configured
on entry, the stock fallback rejects file-backed object arrays supplied
directly as features, targets or fit parameters. For supported explicit folds,
the helper checks the outer container, fold pairs and their indices for
file-backed object arrays before sending them to process workers.
Build the folds in memory from the original numeric indices, for example as a list of
`(train_indices, test_indices)` tuples. Numeric arrays
in those tuples can still be memory-mapped. Use the same in-memory format
when a setup callback replaces `search.cv`. The helper checks the replacement
outer container after the callbacks run and before scikit-learn collects the
splits. Custom callbacks and splitters are responsible for the arrays they
read themselves.

If you pass scikit-learn's `check_cv(explicit_pairs)` as `cv`, the helper
prepares the selected train and test indices for process workers. The same
applies when a search subclass passes that splitter to the candidate
evaluator, whether it reselects its original splitter or supplies a different
one. If a callback supplies the checked splitter and the search saves it
in an attribute before replacing `cv`, a worker scorer reading that attribute
can still use the selected splitter's named records and fold weights. The
pairs can yield their indices through a custom iterator.

If a subclass keeps a *separate* alternate splitter in its own attribute
and its scorer reads that attribute, build the alternate from owning index
arrays. Preparing the rows used for fitting does not rewrite arbitrary
attributes of the subclass.

Workers can also receive the original `search.cv`. For unused explicit
folds, the helper can inspect items stored in lists, tuples (including
named tuples), deques, and ordinary non-scalar NumPy object arrays.
It leaves an unused custom record's iterator
alone. A `UserList` pair or another custom fold record can therefore carry
its original train and test arrays into a worker. Supply plain
`(train_indices, test_indices)` tuples built from the original numeric
indices, or ensure every member of your custom records is safe to serialize
to another process. Keep file-backed object arrays out of custom splitters
and fold metadata as well.

When a subclass or callback replaces a pair in a public fold list after
scikit-learn collects the splits, the fit still uses the collected rows.
For ordinary numeric train/test pairs, a worker scorer reading `search.cv`
sees the public records at dispatch, including private edits to the mapped
indices. The scorer can therefore see different rows from those used to fit.

When a worker inspects the original folds under a splitter override, make
owning copies of reversed, transposed or privately edited numeric maps in
those original records. If you supply a custom splitter object that generates
its own indices, make the same copies for the indices it yields.

**Search pickle hooks for NumPy fold indices.** If you implement `__getstate__`
on a search subclass, the hook runs on your search instance. You can omit
the folds from that state. If you include references to `cv` or
`_checked_cv_orig`, the helper can replace them in
ordinary dictionary values, lists and tuples, including nested containers,
shared references and cycles. For example, an alias in a slot dictionary and
another in a nested list will point to the same worker folds. The helper
leaves dictionary keys and the internals of other objects to your pickle
protocol. Make any fold references stored there safe for process transport.
If a scorer stores the same fold record both in the search's plain state and
inside a separate object, the two can deserialize as different objects when
the search needs different row values. If your scorer or restore hook checks
identity or writes through one reference and reads through the other, make
owning copies of the affected index arrays before building the folds, then
use the same fold record in both locations.
If the entire state is an opaque object and a fold replacement is needed,
the helper raises `TypeError` during serialization. Return ordinary state,
supply owning numeric indices in reusable pairs before building the search,
or fit the search directly with `n_jobs=1`.

If your search exposes `__getstate__` through a read-only or writable
property, you can pass the original mapped folds to `fit_shared_fold_search`
with joblib's `loky` or `multiprocessing` backend.
When a process worker needs a prepared copy of the folds, the helper
preserves the search's serialization protocol. A saving hook selected by
that protocol runs on the original instance, even if the property returns
a fresh callable each time. The property setter is never needed. If the
search ordinarily supports saving and reloading, you can do so after fitting.
With joblib's `loky` and `multiprocessing` backends, a subclass's
`__getstate__` can inspect the search's own instance fields when workers
need prepared folds. For example, a hook can save both `vars(self).copy()` and
`tuple(vars(self))`. A matching `__setstate__` can verify the field names
when it restores the search in the worker. The hook can also save values
from slots or a registry keyed by the original search instance.
To check which fold indices a process received, inspect `search.cv` inside
the worker scorer. Calling `search.__getstate__()` in the submitting
process can still show the caller's original arrays.

Preserving these fields and fold references during serialization can add
noticeable time to a short parallel search. For repeated searches with fixed
indices, you can make owning copies once, before building the folds and the
scorer, and reuse them. Time both approaches on your workload if speed matters.
Rebuild the copies if the indices change. Owning indices also let a
third-party backend that requires its own custom serialization send the search
directly. Use owning indices if a custom loky or multiprocessing backend
replaces the submitted batch or defers forwarding it to the joblib pool.
For NumPy fold indices, a custom pickler can configure its reducer in its
initializer or in `dump`. If a worker scorer captures a mapped fold array
before pickle reaches the search, a custom backend that needs file-backed
values can register the exact search class as a stored key in the batch
pickler's active dispatch dictionary (a `dict` or subclass). Set the entry
before the first affected array is serialized and keep it stable for the batch.
An implicit lookup such as `__missing__` can select file-backed values
after DWD has prepared the caller's private values. The helper rejects that
batch before workers score it. It also rejects a batch if it prepared the
file-backed values but never encountered the search to confirm that choice.
To avoid either conflict, make an owning copy of each affected index array
before constructing both the folds and the scorer, and reference the same
copy wherever the original array was shared. If the scorer changes an owning
array between batches, disable joblib's automatic mapping with
`joblib.parallel_config(backend='loky', max_nbytes=None)` so workers receive
each batch's values.
The helper prepares mapped folds during one successful serialization of the
submitted batch. If your pickler calls `pickle.Pickler.dump(self, obj)`
directly or discards a failed stream and retries, build the folds from owning
indices before creating the search and scorer. Do the same if a reducer
temporarily takes over the underlying C pickler's callbacks while it writes
objects. A pickler may append metadata after the successful batch. Exceptions
from its `persistent_id` hook propagate. If it returns a custom persistent
ID, its backend must also supply the matching reader. The shared-fold worker
reader accepts only the references DWD issued for that batch.

An outer generator may reuse an index buffer
between yields. As in scikit-learn, the buffer then has its final contents
when the collected pairs are unpacked. To give each fold an independent
selection, return a copy of its indices before reusing that storage.

Fit hooks and callable selection policies see the caller's `search.cv` and
`search.refit` throughout the parent fit. For example, a setup hook can log the
policy's name, and the policy can weight scores by the lengths of the original
folds. With `cv=splitter.split(X, y)`, the helper can prepare affected mapped
indices after scikit-learn collects the explicit pairs. For a custom splitter
that generates its own indices, use process-safe arrays in its `split`
method. Custom metadata-routing objects retain their own behavior.

### Scorers that inspect the folds

The indices used for fitting and the object a worker scorer sees as
`search.cv` can differ. With a `collections.UserList` of named tuples holding
in-memory indices, the worker receives the correct indices and sees plain
`(train, test)` pairs in its copy of the search. Other containers or records
can also lose their types and attributes. For a one-pass iterator, the
collected splits are evaluated, the parent iterator is consumed, and the worker's `search.cv`
is `()`. If a scorer needs to inspect the folds, materialize the splits once
before constructing the search and apply the guidance below for named records.

Reading a missing field such as `fold.test` can raise `AttributeError`.
With the default `error_score=np.nan`, scikit-learn warns and assigns NaN to
failed scores. If all candidates receive NaN, the search can still refit the
first candidate. Use `error_score='raise'` to surface the scoring error.

If a scorer chooses a weight based on the class or identity of a fold record
or its outer container, a worker can apply a fallback weight and return a
finite score. This can change which candidate is refitted without a warning,
even with `error_score='raise'`.

In DWD 2.0.2, an ordinary `GridSearchCV` using joblib's standard
process backends and serializer gets the caller's values for NumPy fold
index arrays that DWD copies before dispatch. This includes private edits to a
copy-on-write memory map, whether the folds are tuples, two-row arrays or one
stacked array. In the worker, those indices are plain NumPy arrays.

With a list of `(train, test)` tuples, a scorer that also holds one of those
detached index arrays sees the same array object in the worker. With an
in-memory NumPy object array of folds, the index in the worker's
`search.cv` and the scorer's copy can be distinct objects with the same
values. Use row identifiers for scoring weights. If the scorer requires
the original object's identity, fit with `GridSearchCV(..., n_jobs=1)`.

With an ordinary `GridSearchCV` and joblib's standard process serialization,
a separately created scorer view, such as `priority = stacked_folds[0, 0, ::2]`,
keeps its private values when it shares elements with the affected folds.
If a search subclass defines its own pickle state or reducer, a view held
inside a separate scorer object can still reach the worker through joblib's
native serialization. The worker can then see the backing file's values in
that view even though `search.cv` contains the intended training rows. If the
scorer needs the caller's values, make an owning copy of its view before
constructing the scorer. Reuse that copy wherever the same view is referenced.

Treat scoring inputs as read-only. Code that
requires writes through one view to be visible through another should run
serially. For a separately captured view that joblib could reopen incorrectly,
DWD checks for overlap with affected fold indices from the same open mapping.
It detaches the view if it finds overlap or the bounded check is inconclusive.
When the same numeric view is submitted in another batch, reusing its
detached copy requires checking its layout and comparing its current contents.
This adds a scan for large scorer-owned views. A change in shape or values
is visible to the later batch.
If a custom memory map reports conflicting access modes during the fit,
use an owning copy for the affected indices and scorer views. The helper can
reject the batch when it cannot verify which values workers would receive.
If a custom mode string overrides `__eq__` or `__ne__`, open the map with
the same access mode as a plain Python string. For example, `"c"` keeps
writes private, `"r"` is read-only, and `"r+"` writes through to the file.
Python and NumPy string subclasses that inherit their usual comparisons can
be used directly.
Views that joblib can reopen correctly, provably disjoint views, and views
from a separately opened mapping follow joblib's native transport. If a scorer needs
private edits in one of those arrays, give it an owning copy before fitting.
A scorer that keeps a fold array in a reusable worker after fitting can
keep its temporary file alive until it releases the array or the worker exits.
Custom serialization protocols control their own state.

Joblib can send an ordinary owning array as an `ndarray` or an `np.memmap`,
depending on its size and `max_nbytes`. Compute weights from row or group
identifiers available to the scorer. Inferring weights from
`type(index) is np.ndarray` can change which model is selected.

If an external scorer requires plain arrays with native joblib transport,
make an owning copy of each affected index before constructing the folds and
scorer. Reuse those copies in both. Run the search inside
`joblib.parallel_config(backend='loky', max_nbytes=None)` to disable automatic
mapping of owning arrays, which can increase data transfer. With
`mmap_mode=None`, each fold index or captured view that DWD detaches travels
to the worker as a plain NumPy array. If other owning arrays exceed joblib's
mapping threshold, use `max_nbytes=None` to disable automatic mapping for
those arrays as well. Joblib can still reopen an existing memory map.

If the scorer needs only the fold size, use the targets it receives. For
one-dimensional numeric targets where absolute differences are meaningful,
the following scorer weights negative mean absolute error (MAE) by the
number of held-out observations.

```python
from sklearn.metrics import mean_absolute_error


def neg_total_absolute_error(estimator, X, y):
    return -len(y) * mean_absolute_error(y, estimator.predict(X))
```

Pass `scoring=neg_total_absolute_error` to the search. On a fixed set of
folds, selecting by this test score minimizes total held-out absolute error.
`mean_test_score` divides the negative total by the number of folds. For
pooled MAE, divide the total absolute error by the number of held-out
predictions. With `return_train_score=True`, each training score uses the
size and errors of its training fold. After refit, `search.score(X, y)`
returns negative total absolute error on the supplied batch.

If `user_folds` is a `UserList` of serializable `collections.namedtuple` or
`typing.NamedTuple` pairs with in-memory indices, a scorer that reads their
fields can use `folds = list(user_folds)`. Pass `cv=folds` to `GridSearchCV`
and use the same list for any scorer references. This shallow copy keeps the
named tuples and shared indices, and the helper preserves their fields in
the worker's copy of the search.

An independent scorer can still encounter a pickling error if a local fold
class captures a thread lock, even when only its display method uses the
lock. Some scikit-learn versions send the checked folds to each process with
the search. `list(user_folds)` retains the original records and their class.
If fitting, scoring, refit and any callbacks need only the indices, give
the search plain pairs before fitting:

```python
folds = [(train, test) for train, test in user_folds]
search = GridSearchCV(estimator, param_grid, cv=folds, scoring=scoring, n_jobs=2)
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    fit_shared_fold_search(search, X, y)
```

This keeps the indices and their order. If a scorer or callback reads named
fields, use the named-record recipe below. With in-memory indices and a
mix of lock-bearing and serializable named tuples, you can replace just
the former with field-only named tuples. Keep the serializable named tuples
in the same list. Give all consumers the representation they expect. A
scorer that captures the original display method still captures its lock.

A plain list containing other unpackable records, such as iterable dataclasses,
can still become a list of plain pairs in the worker. If the scorer only needs
named fields, rebuild each pair with the field names it reads. This also works
when the original record class captures a lock and the indices are serializable.

```python
from collections import namedtuple

Fold = namedtuple("Fold", ["train", "test"])
folds = [Fold(train, test) for train, test in user_folds]
```

Use this list for both `cv` and any fold references captured by the scorer.
These are new records. If you own the lock-bearing class and want to keep
its display behavior in the parent, its pickle protocol can deliberately
return the field-only record in workers. For example, define `__reduce__`
on that class to return `(Fold, (self.train, self.test))`, with the
serializable `Fold` defined above. Supply a plain list of the originals.
Use this only when that replacement is appropriate wherever the record is
pickled, including outside the search.

If scoring depends on the original class, container
or relationships between objects, pass the originals to ordinary
`GridSearchCV.fit`. For process workers, those relationships must survive
serialization. When scoring needs the actual parent objects, an essential lock
or other process-local state, score in the parent. For example, given a stock
`GridSearchCV` instance:

```python
from joblib import parallel_config
from threadpoolctl import threadpool_limits

search.set_params(cv=user_folds, n_jobs=1)
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    search.fit(X, y, **fit_params)
```

When the estimator, scorer and callbacks are thread-safe, joblib's
`threading` backend also lets evaluations run in parallel in the same
process. Set the search's worker count explicitly so two folds can run
concurrently and the full-data refit uses one worker and one native thread:

```python
search.set_params(cv=user_folds, n_jobs=2)
with parallel_config(backend="threading", n_jobs=1), threadpool_limits(limits=1):
    fit_shared_fold_search(search, X, y, **fit_params)
```

Worker threads can use the original records and a thread-safe display method
that uses the parent's lock. Use serial fitting if an operation must run on
the calling thread itself.

For direct process-parallel `GridSearchCV.fit` or the published DWD 2.0.1
fallback, make an owning copy with `np.array(view, copy=True)` of each
reversed mapped view, each mapped array whose parent-private edits workers
need, and each contiguous view whose C or Fortran order differs from its
backing map. Prepare these arrays before building the folds and scorer.
Even with the DWD 2.0.2 helper, make these copies for affected arrays
yielded by a custom splitter object, nested in fit parameters, stored in
extra attributes of a custom fold (such as `fold.weights`), or held by a
scorer independently of the affected fold indices. The standard process
path handles the same index and overlapping views as described above.
Reuse the same owning copy wherever the same affected array is referenced.
If the caller requires a read-only copy, set `owned.flags.writeable = False`
on the new array.

### Checking a custom scorer

For a scorer that reads fold objects, compare two fresh searches on the same
data, split indices, ordered candidates and random seeds. Build the reference
with the original fold representation and explicit `n_jobs=1`. Build the trial
with the intended worker configuration and fold representation. For a one-pass
source, save the pairs before either fit. If the producer reuses buffers and
you need each fold's original values, copy the indices as each pair is yielded.
To test iterator behavior, pass a fresh `iter(saved_folds)` to each search.
A single iterator shared by both searches will be exhausted by the first fit.

Bind each scorer to its own search and intended fold references. Set
`error_score='raise'` on both searches. A separate small case with unequal
fold sizes makes a change in size-dependent weighting easier to detect.

For a single metric and a refitted classifier, the following check uses
`X_check`, the same prediction inputs for both fits. Set `use_helper` to
match the production entry point. Supply the same fit-parameter values to
both routes, including any sample weights.

```python
with parallel_config(n_jobs=1), threadpool_limits(limits=1):
    reference.fit(X, y, **fit_params)
    if use_helper:
        fit_shared_fold_search(trial, X, y, **fit_params)
    else:
        trial.fit(X, y, **fit_params)
assert reference.n_splits_ == trial.n_splits_
columns = [f"split{i}_test_score" for i in range(reference.n_splits_)]
for column in [*columns, "mean_test_score"]:
    expected = reference.cv_results_[column]
    actual = trial.cv_results_[column]
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
assert trial.best_index_ == reference.best_index_
np.testing.assert_array_equal(trial.predict(X_check), reference.predict(X_check))
```

With the same ordered grid, the winning index identifies the same parameter
setting. Use tolerances appropriate to the metric and numerical method. For
regression, compare numerical predictions with an appropriate tolerance.

Scores, winners and predictions can agree even when a worker sees the wrong
rows or row order. To check the rows themselves, use a diagnostic dataset
with a unique ID per source row. In the parent, save each split's expected
train and test ID sequences in order, including duplicates. Have a diagnostic
estimator retain the training IDs in `fit` and a scorer read the test IDs from
its input. Assign each distinct expected pair of sequences a numeric code.
Have the scorer raise on an unexpected pair and return the code for an exact
match. Compare each `split{i}_test_score` to the code for that split.

## State and ownership

Scikit-learn performs candidate evaluation, scoring, ranking and refitting.
Use a separate search instance for concurrent fits and keep the input data,
fold definitions and search parameters stable during fitting.
On the shared serial path, scores and rankings use scikit-learn's usual
definitions. Timing columns include the helper's indexing and copy work.
Snapshots use C order, which can change floating-point reduction order for
other input layouts. Bitwise equality across arbitrary layouts or backends is
not promised.

A private view is used only when the entire requested index sequence is exactly
an increasing contiguous span of the helper's immutable root snapshot. Reordered,
duplicate or separated rows keep the original gather-and-snapshot behavior,
preserving the requested row order. This handling is automatic within the
explicit helper; it adds no public setting or eligibility requirement.

The helper restores `search.estimator` after success or failure. A successful
shared-route refit returns an ordinary `KernMD` that owns its feature basis
and supports normal serialization. With `refit=False`, the ordinary search
behavior is retained.
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
call. A `route` of `'stock'` means the serial feature cache was bypassed.
The helper can still prepare mapped arrays and explicit fold indices for
process workers, as described in the [process-parallel guidance](#process-parallel-fallback).
Shared-route counters describe cache use during that call. `root_bytes`
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

On November 9, 2026, the helper and ordinary `GridSearchCV.fit` were timed
using the same `KernMD` implementation. Both routes preserved fitted-data
ownership and numerical safeguards. The helper reused feature copies and
retained its cached subsets through final refit. Snapshot construction and
indexing could outweigh the time saved. The
[validation record](../VALIDATION.md#shared-fold-performance-decision-2026-11-09)
describes the workloads and measurement conditions for that panel.

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

In the November 9 panel, ordinary search was faster for the Linux wide-linear
case, the 1,024-by-64 RBF case on both hosts and the Windows 1,536-by-32 RBF
case. The wide-RBF grid benefited from the helper on both hosts. On November
13, 2026, two within-version comparisons used the same Linux wide-linear
workload and checked correctness after timing. The helper took 15.15% longer
than ordinary search in the eight-block panel and 8.53% less time in the
six-block panel, using the median paired whole-search time ratio in each.
Choose the route using whole-search timings for the data, grid and scoring
settings you intend to use.

Earlier studies used two separate comparisons. The snapshots-only predecessor
was compared directly with ordinary search. The added contiguous-row
optimization was then compared directly with that predecessor. At that stage,
the updated helper had not been directly compared with ordinary search in the
same panel. The historical percentages below must not be combined into such an
estimate or pooled with the later direct comparison.

- **Snapshots-only predecessor versus ordinary search.** In the November 6,
  2026 warm whole-search comparison, the helper took 28.3% less time on
  Windows and 24.5% less on Linux for the 32-row, 65,536-feature float64
  RBF case with three fixed folds and eight candidates. For the 1,024-row,
  64-feature RBF case with two candidates, the helper took 2.0% more time
  than ordinary search on Windows and 14.8% more on Linux. For the
  1,536-row, 32-feature RBF case with four candidates, the helper took
  12.7% more time on Windows.
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
