#define PY_SSIZE_T_CLEAN
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030B0000
#endif
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* Private ordered-row entry. The included arithmetic and its license are unchanged.
 * Including it makes its static finite_fsum/supported_arithmetic available
 * without exporting or rewriting either arithmetic routine. */
#include "dense_score_core.c"

#define PROBE_MAX_ROWS 64
#define PROBE_MAX_TERMS 12288

static int native_double(const Py_buffer *v)
{
    const uint16_t one = 1;
    int little = *((const unsigned char *)&one) == 1;
    const char *f = v->format;
    if (v->itemsize != (Py_ssize_t)sizeof(double) || f == NULL)
        return 0;
    return strcmp(f, "d") == 0 || strcmp(f, "@d") == 0
        || strcmp(f, "=d") == 0 || (little && strcmp(f, "<d") == 0)
        || (!little && (strcmp(f, ">d") == 0 || strcmp(f, "!d") == 0));
}

static int direct_aligned(const Py_buffer *v, int dimensions)
{
    return v->buf != NULL && v->ndim == dimensions
        && v->shape != NULL && v->strides != NULL
        && v->suboffsets == NULL && native_double(v)
        && ((uintptr_t)v->buf % sizeof(double)) == 0;
}

static PyObject *reduce_rows(PyObject *self, PyObject *args)
{
    PyObject *packed_object, *output_object;
    Py_buffer packed = {0}, output = {0};
    int status = ARGUMENT;
    double staged[PROBE_MAX_ROWS];
    (void)self;
    if (!PyArg_ParseTuple(args, "OO:reduce_rows", &packed_object, &output_object))
        return NULL;
    /* Exporter exceptions propagate. No arithmetic or writes precede them. */
    if (PyObject_GetBuffer(packed_object, &packed, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
        return NULL;
    if (PyObject_GetBuffer(output_object, &output, PyBUF_FORMAT | PyBUF_STRIDES) < 0) {
        PyBuffer_Release(&packed);
        return NULL;
    }
    if (!direct_aligned(&packed, 2) || !direct_aligned(&output, 1)
        || output.readonly)
        goto done;

    /* Positive bounded dimensions precede every size multiplication. */
    Py_ssize_t rows = packed.shape[0], width = packed.shape[1];
    if (rows < 1 || rows > PROBE_MAX_ROWS || width < 1
        || width > PROBE_MAX_TERMS / rows || output.shape[0] != rows)
        goto done;
    Py_ssize_t input_bytes = rows * width * (Py_ssize_t)sizeof(double);
    Py_ssize_t output_bytes = rows * (Py_ssize_t)sizeof(double);
    if (packed.len != input_bytes || output.len != output_bytes)
        goto done;
    /* The buffer helper's stride products now have bounded positive shapes. */
    if (!PyBuffer_IsContiguous(&packed, 'C') || !PyBuffer_IsContiguous(&output, 'C'))
        goto done;
    uintptr_t input_start = (uintptr_t)packed.buf;
    uintptr_t output_start = (uintptr_t)output.buf;
    if ((uintptr_t)input_bytes > UINTPTR_MAX - input_start
        || (uintptr_t)output_bytes > UINTPTR_MAX - output_start)
        goto done;
    uintptr_t input_end = input_start + (uintptr_t)input_bytes;
    uintptr_t output_end = output_start + (uintptr_t)output_bytes;
    if (input_start < output_end && output_start < input_end)
        goto done;

    if (!supported_arithmetic()) {
        status = ARITHMETIC_MODE;
        goto done;
    }
    /* Keep the GIL. Input must also be free of foreign/native writers.
     * No NumPy API, product recomputation, new accumulator or partial cap. */
    const double *terms = (const double *)packed.buf;
    for (Py_ssize_t row = 0; row < rows; ++row) {
        status = finite_fsum(terms + row * width, (size_t)width, &staged[row]);
        if (status != OK)
            goto done;
    }
    /* Atomic with respect to status refusal, not a concurrent-memory promise.
     * The caller supplies fresh private output; disjoint input is checked. */
    memcpy(output.buf, staged, (size_t)output_bytes);
done:
    PyBuffer_Release(&output);
    PyBuffer_Release(&packed);
    return PyLong_FromLong(status);
}

static PyMethodDef methods[] = {
    {"reduce_rows", reduce_rows, METH_VARARGS,
     "reduce_rows(packed, output) -> status; ordered finite binary64 reduction.\n"
     "All nonzero statuses leave output unchanged. Export exceptions propagate.\n"
     "Stable input and fresh private output are caller obligations."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_ordered_reduction_accel",
    "Private strict-arithmetic ordered reduction for feature diagnostics.",
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__ordered_reduction_accel(void)
{
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
#define ADD_CONSTANT(name, value) \
    if (PyModule_AddIntConstant(m, name, value) < 0) { Py_DECREF(m); return NULL; }
    ADD_CONSTANT("OK", OK)
    ADD_CONSTANT("ARGUMENT", ARGUMENT)
    ADD_CONSTANT("ARITHMETIC_MODE", ARITHMETIC_MODE)
    ADD_CONSTANT("NONFINITE", NONFINITE)
    ADD_CONSTANT("MEMORY", MEMORY)
    ADD_CONSTANT("PARTIAL_CAPACITY", PARTIAL_CAPACITY)
    ADD_CONSTANT("SUM_OVERFLOW", SUM_OVERFLOW)
    ADD_CONSTANT("API_VERSION", 1)
    ADD_CONSTANT("MAX_ROWS", PROBE_MAX_ROWS)
    ADD_CONSTANT("MAX_TERMS", PROBE_MAX_TERMS)
#undef ADD_CONSTANT
    return m;
}
