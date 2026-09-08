#define PY_SSIZE_T_CLEAN
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030B0000
#endif
#include <Python.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* No NumPy headers or CPython-private interfaces. Py_buffer joined the stable
 * ABI in3.11. Inputs remain pinned while the arithmetic runs without the GIL. */
extern int dwd_sumprod_rows(
    const void *, ptrdiff_t, ptrdiff_t, const void *, ptrdiff_t,
    const void *, ptrdiff_t, size_t, size_t, size_t, double, double,
    double *, double *, double *);

typedef struct { uintptr_t low; uintptr_t high; int empty; } Span;

static int native_double(const Py_buffer *view)
{
    const char *format = view->format;
    const uint16_t one = 1;
    int little = *((const unsigned char *)&one) == 1;
    if (view->itemsize != (Py_ssize_t)sizeof(double) || format == NULL)
        return 0;
    return strcmp(format, "d") == 0 || strcmp(format, "@d") == 0
        || strcmp(format, "=d") == 0
        || (little && strcmp(format, "<d") == 0)
        || (!little && (strcmp(format, ">d") == 0 || strcmp(format, "!d") == 0));
}

static int checked_span(const Py_buffer *view, Span *span)
{
    ptrdiff_t low = 0, high = 0;
    span->empty = 0;
    for (int axis = 0; axis < view->ndim; ++axis) {
        Py_ssize_t count = view->shape[axis];
        ptrdiff_t stride = (ptrdiff_t)view->strides[axis];
        if (count < 0 || (Py_ssize_t)stride != view->strides[axis])
            goto invalid;
        if (count == 0) {
            span->empty = 1;
            span->low = span->high = (uintptr_t)view->buf;
            return 1;
        }
        if (count == 1)
            continue;
        ptrdiff_t steps = (ptrdiff_t)(count - 1);
        if ((Py_ssize_t)steps != count - 1)
            goto invalid;
        if ((stride > 0 && steps > PTRDIFF_MAX / stride)
            || (stride < 0 && stride < PTRDIFF_MIN / steps))
            goto invalid;
        ptrdiff_t extent = steps * stride;
        if (extent > 0) {
            if (high > PTRDIFF_MAX - extent)
                goto invalid;
            high += extent;
        } else {
            if (low < PTRDIFF_MIN - extent)
                goto invalid;
            low += extent;
        }
    }
    if (view->buf == NULL)
        goto invalid;
    uintptr_t address = (uintptr_t)view->buf;
    uintptr_t negative = low < 0 ? (uintptr_t)(-(low + 1)) + 1 : 0;
    if (negative > address || (uintptr_t)high > UINTPTR_MAX - address)
        goto invalid;
    uintptr_t last = address + (uintptr_t)high;
    if (sizeof(double) > UINTPTR_MAX - last)
        goto invalid;
    span->low = address - negative;
    span->high = last + sizeof(double);
    return 1;
invalid:
    PyErr_SetString(PyExc_ValueError, "Buffer shape/strides overflow accessible pointer offsets.");
    return 0;
}

static int get_view(PyObject *object, Py_buffer *view, int dimensions, int writable)
{
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    if (writable)
        flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, view, flags) < 0)
        return 0;
    if (!native_double(view)) {
        PyErr_SetString(PyExc_TypeError, "All arrays must expose native float64 ('d') buffers.");
        return 0;
    }
    if (view->ndim != dimensions || view->shape == NULL || view->strides == NULL) {
        PyErr_SetString(PyExc_ValueError, "Unexpected buffer dimensions or missing shape/strides.");
        return 0;
    }
    for (int axis = 0; axis < view->ndim; ++axis) {
        if (view->suboffsets != NULL && view->suboffsets[axis] >= 0) {
            PyErr_SetString(PyExc_ValueError, "Indirect buffers are unsupported.");
            return 0;
        }
    }
    return 1;
}

static int overlaps(Span a, Span b)
{
    return !a.empty && !b.empty && a.low < b.high && b.low < a.high;
}

static PyObject *evaluate(PyObject *module, PyObject *args)
{
    (void)module;
    PyObject *objects[6];
    Py_buffer buffers[6] = {{0}};
    Span spans[6];
    double shift, intercept;
    Py_ssize_t start, stop;
    int status = -1;
    if (!PyArg_ParseTuple(args, "OOOddnnOOO:evaluate", &objects[0], &objects[1], &objects[2],
                          &shift, &intercept, &start, &stop,
                          &objects[3], &objects[4], &objects[5]))
        return NULL;
    for (int i = 0; i < 6; ++i) {
        if (!get_view(objects[i], &buffers[i], i == 0 ? 2 : 1, i >= 3))
            goto done;
        if (!checked_span(&buffers[i], &spans[i]))
            goto done;
    }
    Py_ssize_t n = buffers[1].shape[0];
    if (n < 1 || n > ((Py_ssize_t)1 << 20) - 3
        || buffers[0].shape[0] != n || buffers[0].shape[1] != n
        || buffers[2].shape[0] != n || start < 0 || stop < start || stop > n) {
        PyErr_SetString(PyExc_ValueError, "Expected square K, matching x/rhs, and0 <= start <= stop <= n.");
        goto done;
    }
    for (int i = 3; i < 6; ++i) {
        if (buffers[i].shape[0] != stop - start
            || (!spans[i].empty && buffers[i].strides[0] != (Py_ssize_t)sizeof(double))
            || (!spans[i].empty && (uintptr_t)buffers[i].buf % _Alignof(double) != 0)) {
            PyErr_SetString(PyExc_ValueError, "Outputs must be aligned contiguous float64 arrays of length stop-start.");
            goto done;
        }
        for (int j = 0; j < i; ++j) {
            if (overlaps(spans[i], spans[j])) {
                PyErr_SetString(PyExc_ValueError, "Output storage must not overlap inputs or other outputs.");
                goto done;
            }
        }
    }
    /* Empty output exporters may supply NULL. The core writes nothing for an
     * empty interval, but still performs its guarded arithmetic validation. */
    double empty_outputs[3];
    Py_BEGIN_ALLOW_THREADS
    status = dwd_sumprod_rows(
        buffers[0].buf, (ptrdiff_t)buffers[0].strides[0], (ptrdiff_t)buffers[0].strides[1],
        buffers[1].buf, (ptrdiff_t)buffers[1].strides[0],
        buffers[2].buf, (ptrdiff_t)buffers[2].strides[0],
        (size_t)n, (size_t)start, (size_t)stop, shift, intercept,
        spans[3].empty ? &empty_outputs[0] : (double *)buffers[3].buf,
        spans[4].empty ? &empty_outputs[1] : (double *)buffers[4].buf,
        spans[5].empty ? &empty_outputs[2] : (double *)buffers[5].buf);
    Py_END_ALLOW_THREADS
done:
    for (int i = 0; i < 6; ++i)
        if (buffers[i].obj != NULL)
            PyBuffer_Release(&buffers[i]);
    if (status < 0)
        return NULL;
    return PyLong_FromLong(status);
}

static PyMethodDef methods[] = {
    {"evaluate", evaluate, METH_VARARGS,
     "evaluate(K,x,rhs,shift,s,start,stop,scores,residuals,maxima) -> status\n"
     "Buffers must remain unchanged during the call. Discard outputs on nonzero status.\n"
     "Status:0 success,1 bad arguments,2 operand decline,3 nonfinite,4 unsupported arithmetic."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_residual_accel", "Optional bounded residual arithmetic; stable ABI3.11+.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__residual_accel(void)
{
    return PyModule_Create(&definition);
}
