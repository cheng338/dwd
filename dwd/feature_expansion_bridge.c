#define PY_SSIZE_T_CLEAN
#ifdef Py_LIMITED_API
#error This isolated experiment requires exact CPython 3.12 full ABI
#endif
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include "feature_expansion_core.c"

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 12
#error Build this experiment only with CPython 3.12 headers
#endif

/* Context is a finite, source-owned description of the callbacks bypassed by
 * this gradient. It is not a general security/attestation API. All references
 * are held by the exact tuple throughout the GIL-held call. No foreign writes
 * to the arrays are permitted. Context shape and family are built in the facade.
 * Dictionary-slot acquisition MAY allocate; it occurs before the final barrier.
 * The final barrier only reads previously acquired slots and borrowed fields.
 */
#define MAX_DICTIONARIES 64
#define MAX_BINDINGS 256
#define MAX_FUNCTIONS 64
#define MAX_TYPES 16
#define MAX_INSTANCES 16
#define MAX_VIEWS 16

static int tuple_size(PyObject *o, Py_ssize_t size)
{ return PyTuple_CheckExact(o) && PyTuple_GET_SIZE(o) == size; }

static int bounded_tuple(PyObject *o, Py_ssize_t maximum)
{ return PyTuple_CheckExact(o) && PyTuple_GET_SIZE(o) <= maximum; }

static int string_dictionary(PyObject *o)
{
    if (!PyDict_CheckExact(o)) return 0;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(o, &pos, &key, &value))
        if (!PyUnicode_CheckExact(key)) return 0;
    return 1;
}

static int dictionary_binding(PyObject *dict, PyObject *key, PyObject *value)
{
    /* Every key in dict has already been checked exact Unicode. Consequently
     * dictionary lookup cannot dispatch custom equality/hash callbacks. */
    return PyUnicode_CheckExact(key) && PyDict_GetItemWithError(dict, key) == value;
}

static int function_token(PyObject *t)
{
    if (!tuple_size(t, 8)) return 0;
    PyObject *f = PyTuple_GET_ITEM(t, 0);
    if (!PyFunction_Check(f)) return 0;
    PyFunctionObject *fn = (PyFunctionObject *)f;
    if (fn->func_code != PyTuple_GET_ITEM(t, 1)
        || (fn->func_defaults ? fn->func_defaults : Py_None) != PyTuple_GET_ITEM(t, 2)
        || fn->func_globals != PyTuple_GET_ITEM(t, 4)
        || fn->func_module != PyTuple_GET_ITEM(t, 5)
        || fn->func_name != PyTuple_GET_ITEM(t, 6)
        || fn->func_qualname != PyTuple_GET_ITEM(t, 7)) return 0;
    PyObject *expected = PyTuple_GET_ITEM(t, 3), *actual = fn->func_kwdefaults;
    if (expected == Py_None) {
        if (actual != NULL) return 0;
    } else {
        if (!bounded_tuple(expected, 8) || !string_dictionary(actual)
            || PyDict_Size(actual) != PyTuple_GET_SIZE(expected)) return 0;
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(expected); ++i) {
            PyObject *pair = PyTuple_GET_ITEM(expected, i);
            if (!tuple_size(pair, 2) || !dictionary_binding(actual,
                    PyTuple_GET_ITEM(pair, 0), PyTuple_GET_ITEM(pair, 1))) return 0;
        }
    }
    if (fn->func_dict != NULL) {
        if (!string_dictionary(fn->func_dict)) return 0;
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(fn->func_dict, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "__wrapped__") == 0) return 0;
    }
    return 1;
}

static int prepare_context(PyObject *context, PyObject ***slots)
{
    if (!tuple_size(context, 6)) return 0;
    const Py_ssize_t limits[5] = {MAX_DICTIONARIES, MAX_BINDINGS,
                                  MAX_FUNCTIONS, MAX_TYPES, MAX_INSTANCES};
    for (int i = 0; i < 5; ++i)
        if (!bounded_tuple(PyTuple_GET_ITEM(context, i), limits[i])) return 0;
    PyObject *instances = PyTuple_GET_ITEM(context, 4);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(instances); ++i) {
        PyObject *t = PyTuple_GET_ITEM(instances, i);
        if (!tuple_size(t, 3) || !PyType_CheckExact(PyTuple_GET_ITEM(t, 1))
            || (PyObject *)Py_TYPE(PyTuple_GET_ITEM(t, 0)) != PyTuple_GET_ITEM(t, 1)
            || !PyDict_CheckExact(PyTuple_GET_ITEM(t, 2))) return 0;
        /* Exact, prequalified cached-property / NumPy dispatcher objects only.
         * 3.12.15 object.c1188: this returns the object's stable dictionary slot,
         * materializing inline attributes if needed. Never call it after gate. */
        slots[i] = _PyObject_GetDictPtr(PyTuple_GET_ITEM(t, 0));
        if (slots[i] == NULL || *slots[i] != PyTuple_GET_ITEM(t, 2)) return 0;
    }
    return 1;
}

static int final_context(PyObject *context, PyObject ***slots)
{
    PyThreadState *thread = PyThreadState_Get();
    if (thread->c_profilefunc != NULL || thread->c_tracefunc != NULL) return 0;
    PyObject *dicts = PyTuple_GET_ITEM(context, 0);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(dicts); ++i)
        if (!string_dictionary(PyTuple_GET_ITEM(dicts, i))) return 0;
    PyObject *bindings = PyTuple_GET_ITEM(context, 1);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(bindings); ++i) {
        PyObject *t = PyTuple_GET_ITEM(bindings, i);
        if (!tuple_size(t, 3)) return 0;
        PyObject *d = PyTuple_GET_ITEM(t, 0);
        int present = 0;
        for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(dicts); ++j)
            if (d == PyTuple_GET_ITEM(dicts, j)) { present = 1; break; }
        if (!present || !dictionary_binding(d, PyTuple_GET_ITEM(t, 1),
                                              PyTuple_GET_ITEM(t, 2))) return 0;
    }
    PyObject *functions = PyTuple_GET_ITEM(context, 2);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(functions); ++i)
        if (!function_token(PyTuple_GET_ITEM(functions, i))) return 0;
    PyObject *types = PyTuple_GET_ITEM(context, 3);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(types); ++i) {
        PyObject *t = PyTuple_GET_ITEM(types, i);
        if (!tuple_size(t, 3) || !PyType_CheckExact(PyTuple_GET_ITEM(t, 0))) return 0;
        PyTypeObject *cls = (PyTypeObject *)PyTuple_GET_ITEM(t, 0);
        PyObject *snapshot = PyTuple_GET_ITEM(t, 1), *live = cls->tp_dict;
        if (cls->tp_bases != PyTuple_GET_ITEM(t, 2)
            || !string_dictionary(snapshot) || !string_dictionary(live)
            || PyDict_Size(snapshot) != PyDict_Size(live)) return 0;
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(snapshot, &pos, &key, &value))
            if (!dictionary_binding(live, key, value)) return 0;
    }
    PyObject *instances = PyTuple_GET_ITEM(context, 4);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(instances); ++i) {
        PyObject *t = PyTuple_GET_ITEM(instances, i);
        if ((PyObject *)Py_TYPE(PyTuple_GET_ITEM(t, 0)) != PyTuple_GET_ITEM(t, 1)
            || *slots[i] != PyTuple_GET_ITEM(t, 2)) return 0;
    }
    PyObject *anchor = PyTuple_GET_ITEM(context, 5);
    if (!tuple_size(anchor, 2) || !string_dictionary(PyTuple_GET_ITEM(anchor, 0))
        || !dictionary_binding(PyTuple_GET_ITEM(anchor, 0),
                               PyTuple_GET_ITEM(anchor, 1), context)) return 0;
    return 1;
}

static int native_format(const Py_buffer *v, char kind, Py_ssize_t size)
{
    if (v->itemsize != size || v->format == NULL) return 0;
    const char *f = v->format;
    const uint16_t one = 1;
    int little = *((const unsigned char *)&one) == 1;
    if (*f == '@' || *f == '=' || (*f == '<' && little)
        || ((*f == '>' || *f == '!') && !little)) ++f;
    return f[0] == kind && f[1] == '\0';
}

static int shaped(const Py_buffer *v, int ndim, Py_ssize_t a, Py_ssize_t b,
                  char kind, Py_ssize_t size)
{
    if (v->buf == NULL || v->ndim != ndim || v->shape == NULL
        || v->strides == NULL || v->suboffsets != NULL
        || !native_format(v, kind, size) || ((uintptr_t)v->buf % (uintptr_t)size))
        return 0;
    if (v->shape[0] != a || (ndim == 2 && v->shape[1] != b)
        || v->len != a * b * size) return 0;
    return PyBuffer_IsContiguous(v, 'C');
}

static int disjoint(const Py_buffer *a, const Py_buffer *b)
{
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
    if ((uintptr_t)a->len > UINTPTR_MAX - x || (uintptr_t)b->len > UINTPTR_MAX - y)
        return 0;
    return x + (uintptr_t)a->len <= y || y + (uintptr_t)b->len <= x;
}

static PyObject *gradient(PyObject *bundle, int split_interface)
{
    Py_buffer views[MAX_VIEWS] = {{0}};
    PyObject **slots[MAX_INSTANCES] = {0};
    double staged[FEATURE_MAX_ROWS], *scratch = NULL;
    int32_t *rhs_exponents = NULL;
    int status = ARGUMENT, acquired = 0;
    if (Py_Version != PY_VERSION_HEX) return PyLong_FromLong(FEATURE_GUARD);
    /* bundle=(context, exact memoryview tuple, penalty exponent). */
    if (!tuple_size(bundle, 3)) return PyLong_FromLong(ARGUMENT);
    PyObject *context = PyTuple_GET_ITEM(bundle, 0), *buffers = PyTuple_GET_ITEM(bundle, 1);
    const int count = split_interface ? 16 : 7;
    if (!tuple_size(buffers, count) || !PyLong_CheckExact(PyTuple_GET_ITEM(bundle, 2)))
        return PyLong_FromLong(ARGUMENT);
    int overflow = 0;
    long long penalty = PyLong_AsLongLongAndOverflow(PyTuple_GET_ITEM(bundle, 2), &overflow);
    if (overflow) return PyLong_FromLong(FEATURE_EXPONENT);
    if (PyErr_Occurred()) return NULL;
    for (int i = 0; i < count; ++i) {
        PyObject *view = PyTuple_GET_ITEM(buffers, i);
        /* Never acquire an unknown exporter's buffer. Exact memoryviews were
         * constructed only after exact-base-array checks in the facade. */
        if (!PyMemoryView_Check(view)) goto done;
        if (PyObject_GetBuffer(view, &views[i], PyBUF_FORMAT | PyBUF_STRIDES) < 0)
            goto error;
        ++acquired;
    }
    const Py_buffer *tile = &views[0], *output = &views[count-1];
    if (tile->ndim != 2 || tile->shape == NULL) goto done;
    Py_ssize_t rows = tile->shape[0], width = tile->shape[1];
    if (rows < 2 || rows > FEATURE_MAX_ROWS || width < 1
        || width > FEATURE_MAX_WIDTH || width > FEATURE_MAX_ENTRIES / rows) goto done;
    if (!shaped(tile, 2, rows, width, 'd', 8)
        || !shaped(output, 1, rows, 1, 'd', 8) || output->readonly) goto done;
    for (int i = 1; i < count-1; ++i) {
        int integer = split_interface ? (i == 2 || i == 6 || i == 10 || i >= 13) : i >= 4;
        if (!shaped(&views[i], 1, width, 1, integer ? 'i' : 'd', integer ? 4 : 8)) goto done;
    }
    for (int i = 0; i < count-1; ++i)
        if (!disjoint(&views[i], output)) goto done;
    if (!prepare_context(context, slots)) { status = FEATURE_GUARD; goto done; }
    scratch = PyMem_Malloc((size_t)(split_interface ? 6 : 15) * (size_t)width * sizeof(double));
    if (scratch == NULL) { PyErr_NoMemory(); goto error; }
    if (!split_interface) {
        rhs_exponents = PyMem_Malloc(3 * (size_t)width * sizeof(int32_t));
        if (rhs_exponents == NULL) { PyErr_NoMemory(); goto error; }
    }
    /* FINAL BARRIER. All allocations, exports and dictionary materialization
     * precede it. Until staged publication there are NO Python calls capable
     * of allocation, callbacks, finalization, audit, descriptors or GIL release. */
    if (!final_context(context, slots)) { status = FEATURE_GUARD; goto done; }
    if (!supported_arithmetic()) { status = ARITHMETIC_MODE; goto done; }
    feature_right_split right[3];
    if (split_interface) {
        for (int k = 0; k < 3; ++k) {
            right[k].mantissa = views[1+4*k].buf;
            right[k].exponent = views[2+4*k].buf;
            right[k].high = views[3+4*k].buf;
            right[k].low = views[4+4*k].buf;
        }
    } else {
        const double *raw[3] = {views[1].buf, views[2].buf, views[3].buf};
        status = feature_prepare_right(raw, (size_t)width, scratch + 6*width, rhs_exponents, right);
        if (status != OK) goto done;
    }
    status = feature_gradient(tile->buf, (size_t)rows, (size_t)width, right,
        views[count-3].buf, views[count-2].buf, penalty, scratch, staged);
    if (status == OK) memcpy(output->buf, staged, (size_t)rows * sizeof(double));
done:
    PyMem_Free(rhs_exponents);
    PyMem_Free(scratch);
    for (int i = acquired; i-- > 0;) PyBuffer_Release(&views[i]);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLong(status);
error:
    PyMem_Free(rhs_exponents);
    PyMem_Free(scratch);
    for (int i = acquired; i-- > 0;) PyBuffer_Release(&views[i]);
    return NULL;
}

static PyObject *gradient_from_splits(PyObject *self, PyObject *bundle)
{ (void)self; return gradient(bundle, 1); }
static PyObject *gradient_from_prepared(PyObject *self, PyObject *bundle)
{ (void)self; return gradient(bundle, 0); }

static PyObject *beta_from_split(PyObject *self, PyObject *bundle)
{
    (void)self;
    Py_buffer views[7] = {{0}};
    PyObject **slots[MAX_INSTANCES] = {0};
    double staged[FEATURE_MAX_ROWS], *scratch = NULL;
    int status = ARGUMENT, acquired = 0;
    if (Py_Version != PY_VERSION_HEX) return PyLong_FromLong(FEATURE_GUARD);
    /* Same already anchored context. No context creation or import here. */
    if (!tuple_size(bundle, 2)) return PyLong_FromLong(ARGUMENT);
    PyObject *context = PyTuple_GET_ITEM(bundle, 0), *buffers = PyTuple_GET_ITEM(bundle, 1);
    if (!tuple_size(buffers, 7)) return PyLong_FromLong(ARGUMENT);
    for (int i = 0; i < 7; ++i) {
        PyObject *view = PyTuple_GET_ITEM(buffers, i);
        if (!PyMemoryView_Check(view)) goto done;
        if (PyObject_GetBuffer(view, &views[i], PyBUF_FORMAT | PyBUF_STRIDES) < 0)
            goto error;
        ++acquired;
    }
    const Py_buffer *tile = &views[0], *output = &views[6];
    if (tile->ndim != 2 || tile->shape == NULL) goto done;
    Py_ssize_t rows = tile->shape[0], width = tile->shape[1];
    if (rows < 2 || rows > FEATURE_MAX_ROWS || width < 1
        || width > FEATURE_MAX_WIDTH || width > FEATURE_MAX_ENTRIES / rows) goto done;
    if (!shaped(tile, 2, rows, width, 'd', 8)
        || !shaped(output, 1, rows, 1, 'd', 8) || output->readonly) goto done;
    for (int i = 1; i < 6; ++i) {
        int integer = i == 2 || i == 5;
        if (!shaped(&views[i], 1, width, 1, integer ? 'i' : 'd', integer ? 4 : 8)) goto done;
    }
    for (int i = 0; i < 6; ++i)
        if (!disjoint(&views[i], output)) goto done;
    if (!prepare_context(context, slots)) { status = FEATURE_GUARD; goto done; }
    scratch = PyMem_Malloc(2 * (size_t)width * sizeof(double));
    if (scratch == NULL) { PyErr_NoMemory(); goto error; }
    /* Same final barrier, after all allocation/export/materialization. */
    if (!final_context(context, slots)) { status = FEATURE_GUARD; goto done; }
    if (!supported_arithmetic()) { status = ARITHMETIC_MODE; goto done; }
    feature_right_split right = {views[1].buf, views[2].buf, views[3].buf, views[4].buf};
    status = feature_beta(tile->buf, (size_t)rows, (size_t)width, &right,
                          views[5].buf, scratch, staged);
    if (status == OK) memcpy(output->buf, staged, (size_t)rows * sizeof(double));
done:
    PyMem_Free(scratch);
    for (int i = acquired; i-- > 0;) PyBuffer_Release(&views[i]);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLong(status);
error:
    PyMem_Free(scratch);
    for (int i = acquired; i-- > 0;) PyBuffer_Release(&views[i]);
    return NULL;
}

static PyMethodDef methods[] = {
    {"gradient_from_splits", gradient_from_splits, METH_O, "Private G-S; complete staged gradient or refusal."},
    {"gradient_from_prepared", gradient_from_prepared, METH_O, "Private G-R; complete staged gradient or refusal."},
    {"beta_from_split", beta_from_split, METH_O, "Private warm Q; complete staged coefficient tile or refusal."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_feature_expansion_accel",
    "Exact-build CPython 3.12 experimental gradient and warm beta expansion.", -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__feature_expansion_accel(void)
{
    if (Py_Version != PY_VERSION_HEX) {
        PyErr_SetString(PyExc_ImportError, "Fused experiment requires the exact CPython header build.");
        return NULL;
    }
    PyObject *module = PyModule_Create(&definition);
    if (module == NULL) return NULL;
#define CONSTANT(name, value) if (PyModule_AddIntConstant(module, name, value) < 0) { Py_DECREF(module); return NULL; }
    CONSTANT("API_VERSION", 2)
    CONSTANT("BETA_API_VERSION", 1)
    CONSTANT("COMPILED_PY_HEX", PY_VERSION_HEX)
    CONSTANT("MAX_ROWS", FEATURE_MAX_ROWS)
    CONSTANT("MAX_ENTRIES", FEATURE_MAX_ENTRIES)
    CONSTANT("MAX_WIDTH", FEATURE_MAX_WIDTH)
    CONSTANT("OK", OK)
    CONSTANT("ARGUMENT", ARGUMENT)
    CONSTANT("ARITHMETIC_MODE", ARITHMETIC_MODE)
    CONSTANT("NONFINITE", NONFINITE)
    CONSTANT("PARTIAL_CAPACITY", PARTIAL_CAPACITY)
    CONSTANT("SUM_OVERFLOW", SUM_OVERFLOW)
    CONSTANT("RESTORATION", FEATURE_RESTORATION)
    CONSTANT("EXPONENT", FEATURE_EXPONENT)
    CONSTANT("GUARD", FEATURE_GUARD)
#undef CONSTANT
    return module;
}
