"""Scoring and monitoring compare temporal rows across sparse and ordinary storage."""

import numpy as np
import pytest

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD


pd = pytest.importorskip('pandas')
X = np.array([[-1.], [1.]] * 3)
WEIGHTS = [7, 11, 5, 3, 2, 1]


def sparse_labels(low, high):
    # Establish the gaps before inserting the true fill. This keeps pandas
    # from discarding a scalar's original unit during sparse construction.
    labels = pd.arrays.SparseArray(np.fromiter(['gap', high] * 3, dtype=object),
                                   dtype=pd.SparseDtype(object, 'gap'))
    labels.fill_value = low
    return labels


def boxed_labels(rows, carrier):
    values = np.fromiter(rows, dtype=object)
    if carrier == 'series':
        return pd.Series(values, dtype=object)
    if carrier == 'index':
        return pd.Index(values, dtype=object)
    if carrier == 'frame':
        return pd.Series(values, dtype=object).to_frame()
    return values


def validation_history(training, truth):
    model = KernGDWD(stopping='validation', initialization='zero', max_iter=2)
    model.fit(X, training, validation_data=(X, truth))
    return [entry['score'] for entry in model.validation_history_]


@pytest.mark.parametrize('scalar,box', [(np.datetime64, pd.Timestamp), (np.timedelta64, pd.Timedelta)])
@pytest.mark.parametrize('carrier', ['array', 'series', 'index', 'frame'])
@pytest.mark.parametrize('ordinary_training', [False, True], ids=['sparse-fit', 'boxed-fit'])
@pytest.mark.parametrize('units', ['mixed', 'retained'])
def test_boxed_and_sparse_temporal_rows_match_in_both_directions(scalar, box, carrier, ordinary_training, units):
    low, high = scalar(1000, 'ns'), scalar(3, 'us')
    if units == 'retained':
        low, high = box(scalar(1, 'us')), box(high)
    sparse = sparse_labels(low, high)
    boxed = boxed_labels([low, high] * 3, carrier)
    training, truth = (boxed, sparse) if ordinary_training else (sparse, boxed)
    model = KernMD().fit(X, training)
    if ordinary_training or units == 'retained':
        for result in (model.classes_, model.predict(X)):
            assert result.dtype == object
            assert all(type(label) is type(low) for label in result)
    else:
        assert model.classes_.dtype == low.dtype
    if units == 'retained' and not ordinary_training:
        # Both operands retain ordinary pandas objects. Keep sklearn's existing
        # target-domain decision, including the same unsupported boxed self-score.
        with pytest.raises(ValueError, match='unknown'):
            model.score(X, truth)
    else:
        assert model.score(X[[1, 1, 2, 3, 4, 5]], truth, sample_weight=WEIGHTS) == 22 / 29
    assert validation_history(training, truth) == [1., 1.]


@pytest.mark.parametrize('scalar', [np.datetime64, np.timedelta64])
@pytest.mark.parametrize('carrier', ['array', 'series'])
def test_boxed_temporal_rows_without_a_shared_tick_do_not_wrap(scalar, carrier):
    # The high ns tick wraps onto 1000 ps if NumPy promotes both to ps.
    # The held 1 ps row prevents an exact shared grid even though individual
    # held rows can still match either trained class.
    low, high = scalar(1000, 'ps'), scalar(2**61 + 1, 'ns')
    training = sparse_labels(low, high)
    truth = boxed_labels([high, high, low, scalar(1, 'ps'), low, high], carrier)
    assert KernMD().fit(X, training).score(X, truth, sample_weight=WEIGHTS) == 19 / 29
    with pytest.raises(ValueError, match='Validation labels must belong'):
        validation_history(training, truth)


@pytest.mark.parametrize('scalar', [np.datetime64, np.timedelta64])
@pytest.mark.parametrize('all_missing', [False, True], ids=['some-missing', 'all-missing'])
def test_boxed_temporal_nat_is_a_mismatch_and_cannot_be_monitored(scalar, all_missing):
    low, high = scalar(1000, 'ns'), scalar(3, 'us')
    training = sparse_labels(low, high)
    rows = [pd.NaT] * 6 if all_missing else [pd.NaT, high, low, scalar('NaT', 'ps'), low, high]
    truth = boxed_labels(rows, 'series')
    assert KernMD().fit(X, training).score(X, truth, sample_weight=WEIGHTS) == (0. if all_missing else 19 / 29)
    with pytest.raises(ValueError, match='Validation labels must belong'):
        validation_history(training, truth)
    with pytest.raises(ValueError):
        KernMD().fit(X, truth)


def test_calendar_and_fixed_boxed_durations_are_incomparable():
    training = sparse_labels(np.timedelta64(12, 'M'), np.timedelta64(2, 'Y'))
    truth = boxed_labels([np.timedelta64(365, 'D'), np.timedelta64(730, 'D')] * 3, 'array')
    with pytest.raises(ValueError, match='incompatible units'):
        KernMD().fit(X, training).score(X, truth)
    with pytest.raises(ValueError, match='incompatible units'):
        validation_history(training, truth)


@pytest.mark.parametrize('scalar', [np.datetime64, np.timedelta64])
def test_boxed_held_projection_selects_rows_once(scalar):
    low, high = scalar(1000, 'ns'), scalar(3, 'us')
    training = sparse_labels(low, high)
    calls = []

    class Projected(pd.Series):
        def __array__(self, dtype=None, copy=None):
            calls.append(1)
            return np.fromiter([low, high] * 3, dtype=object)

    held = Projected([0, 0] * 3)
    assert KernMD().fit(X, training).score(X[[1, 1, 2, 3, 4, 5]], held, sample_weight=WEIGHTS) == 22 / 29
    assert calls == [1]
    calls.clear()
    assert validation_history(training, held) == [1., 1.]
    assert calls == [1]


@pytest.mark.parametrize('family', [np.datetime64, np.timedelta64])
@pytest.mark.parametrize('column', [False, True])
def test_passive_array_view_of_held_temporal_scalars(family, column):
    class LabelView(np.ndarray):
        pass

    low, high = family(-4000, 'ns'), family(7, 'us')
    held = np.fromiter([low, high] * 3, dtype=object).view(LabelView)
    if column:
        held = held[:, None]
    training = sparse_labels(low, high)
    assert KernMD().fit(X, training).score(X[[1, 1, 2, 3, 4, 5]], held, sample_weight=WEIGHTS) == 22 / 29
    assert validation_history(training, held) == [1., 1.]


def test_boxed_temporal_view_with_numpy_dispatch_keeps_its_original_operand():
    from dwd._validation import _temporal_comparison_targets

    class Dispatched(np.ndarray):
        def __array_function__(self, func, types, args, kwargs):
            raise AssertionError("unexpected dispatch while inspecting labels")

    ticks = np.array([1, 3], dtype="timedelta64[us]")
    observed = np.fromiter(ticks, dtype=object).view(Dispatched)
    held, classes = _temporal_comparison_targets(observed, ticks)
    assert held is observed and classes is ticks


@pytest.mark.parametrize('family', ['M', 'm'], ids=['datetime', 'duration'])
@pytest.mark.parametrize('source,target,factor', [('3us', '6ns', 500), ('6ns', '3ns', 2)])
def test_fixed_cast_exact_counts_at_stepped_unit_limits(family, source, target, factor):
    from dwd._validation import _temporal_cast

    src = np.dtype(f'{family}8[{source}]')
    dst = np.dtype(f'{family}8[{target}]')
    limit = (2**63 - 1) // factor
    counts = np.array([[-limit, -1, 2, 3], [-2, 0, 1, limit]], dtype='i8')
    converted = _temporal_cast(counts.view(src), dst, pd)
    assert converted.dtype == dst
    np.testing.assert_array_equal(converted.view('i8'),
                                  [[int(value) * factor for value in row] for row in counts])

    # With factor two the first count below the negative limit would encode NaT.
    for outside in (-limit - 1, limit + 1):
        assert _temporal_cast(np.array([0] * 7 + [outside], dtype=src), dst, pd) is None


@pytest.mark.parametrize('family', ['M', 'm'], ids=['datetime', 'duration'])
@pytest.mark.parametrize('source,target,counts,expected', [
    ('3us', '6ns', [-2**63, -7, 9], [-2**63, -3500, 4500]),
    ('Y', 'M', [-1, 0, 2], [-12, 0, 24]),
    ('2us', '3ns', [-6, 3], [-4000, 2000]),
    ('2us', '3ns', [3, 1], None),
    ('D', 'as', [0], [0]),
    ('D', 'as', [1], None),
], ids=['nat', 'calendar', 'divisible-counts', 'nondivisible-count', 'wide-zero', 'wide-overflow'])
def test_temporal_cast_fallback_preserves_exact_counts(family, source, target, counts, expected):
    from dwd._validation import _temporal_cast

    values = np.array(counts * 8, dtype=f'{family}8[{source}]')
    dst = np.dtype(f'{family}8[{target}]')
    converted = _temporal_cast(values, dst, pd)
    if expected is None:
        assert converted is None
    else:
        assert converted.dtype == dst
        np.testing.assert_array_equal(converted.view('i8'), expected * 8)


@pytest.mark.parametrize('swap_source', [True, False], ids=['source', 'destination'])
def test_fixed_cast_respects_nonnative_byte_order(swap_source):
    from dwd._validation import _temporal_cast

    src = np.dtype('m8[us]').newbyteorder('S' if swap_source else '=')
    dst = np.dtype('m8[ns]').newbyteorder('=' if swap_source else 'S')
    converted = _temporal_cast(np.array([-1, 7] * 4, dtype=src), dst, pd)
    assert converted.dtype == dst
    np.testing.assert_array_equal(converted.astype('i8'), [-1000, 7000] * 4)


def test_fixed_cast_respects_array_subclass_scalar_observations():
    from dwd._validation import _temporal_cast

    class ProjectedTicks(np.ndarray):
        @property
        def flat(self):
            return iter((np.timedelta64(-3, 'us'), np.timedelta64(4, 'us')) * 4)

    values = np.array([1, 2] * 4, dtype='m8[us]').view(ProjectedTicks)
    converted = _temporal_cast(values, np.dtype('m8[ns]'), pd)
    np.testing.assert_array_equal(converted.view('i8'), [-3000, 4000] * 4)


@pytest.mark.parametrize('layout', ['fortran', 'strided'])
def test_fixed_cast_preserves_shape_and_owns_target_storage(layout):
    from dwd._validation import _temporal_cast

    grid = np.arange(-12, 12, dtype='i8').reshape(4, 6)
    counts = np.asfortranarray(grid) if layout == 'fortran' else grid[:, ::-2]
    values = counts.view('M8[3us]')
    values.flags.writeable = False
    dst = np.dtype('M8[6ns]', metadata={'label': 'held'})
    converted = _temporal_cast(values, dst, pd)
    assert converted.dtype == dst and converted.dtype.metadata == dst.metadata
    assert converted.flags.c_contiguous and converted.flags.owndata
    assert not np.shares_memory(converted, values)
    np.testing.assert_array_equal(converted.view('i8'), counts * 500)


def test_temporal_cast_of_scalar_retains_shape():
    from dwd._validation import _temporal_cast

    values = np.array(-3, dtype='m8[us]')
    assert _temporal_cast(values, values.dtype, pd) is values
    converted = _temporal_cast(values, np.dtype('m8[ns]'), pd)
    assert converted.shape == () and converted.dtype == np.dtype('m8[ns]')
    assert converted.view('i8').item() == -3000


@pytest.mark.parametrize('scalar', [np.datetime64, np.timedelta64])
def test_batched_boxed_score_distinguishes_wrapped_class_alias(scalar):
    low, high = scalar(1000, 'ns'), scalar(3, 'us')
    model = KernMD().fit(X, sparse_labels(low, high))
    repeats = 256
    features = np.tile([[1.], [1.], [-1.], [1.]], (repeats, 1))
    weights = np.tile([7, 11, 5, 3], repeats)
    held = boxed_labels([low, high, low, high] * repeats, 'series')
    assert model.score(features, held, sample_weight=weights) == 19 / 26

    # This value aliases the low class under unchecked multiplication by 1000.
    # The affected prediction was correct and had weight five.
    held.iloc[-2] = scalar(2**61 + 1, 'us')
    assert model.score(features, held, sample_weight=weights) == (19 * repeats - 5) / (26 * repeats)
