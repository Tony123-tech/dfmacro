#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject* zero_copy_filter_gt(PyObject* self, PyObject* args) {
    PyObject* input_array_obj;
    double threshold;

    if (!PyArg_ParseTuple(args, "Od", &input_array_obj, &threshold)) {
        return NULL;
    }

    if (!PyArray_Check(input_array_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array");
        return NULL;
    }

    PyArrayObject* input_array = (PyArrayObject*)input_array_obj;

    if (PyArray_NDIM(input_array) != 1 || PyArray_TYPE(input_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Expected 1D float64 array");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(input_array);
    double* data = (double*)PyArray_DATA(input_array);

    npy_intp count = 0;
    for (npy_intp i = 0; i < size; i++) {
        if (data[i] > threshold) {
            count++;
        }
    }

    npy_intp dims[1] = {count};
    PyArrayObject* result_indices = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_INT64, 0);
    npy_int64* out_data = (npy_int64*)PyArray_DATA(result_indices);

    npy_intp out_idx = 0;
    for (npy_intp i = 0; i < size; i++) {
        if (data[i] > threshold) {
            out_data[out_idx++] = i;
        }
    }

    return (PyObject*)result_indices;
}

static PyMethodDef DfmacroMethods[] = {
    {"zero_copy_filter_gt", zero_copy_filter_gt, METH_VARARGS, "Zero-copy filter returning indices."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dfmacro_c_module = {
    PyModuleDef_HEAD_INIT,
    "dfmacro_c",
    "C extension for dfmacro",
    -1,
    DfmacroMethods
};

PyMODINIT_FUNC PyInit_dfmacro_c(void) {
    import_array();
    return PyModule_Create(&dfmacro_c_module);
}
