
// you must include Python.h before ANY other header files!
#include <Python.h>
#include <structmember.h>  // need this for PyMemberDef

// use the following if extension is composed of multiple files
//#define PY_ARRAY_UNIQUE_SYMBOL xx_numpy_xx
#include <numpy/arrayobject.h>

#include "gem.h"

typedef struct {
    PyObject_HEAD
    gemID gid;
} gemObject;


static void
gemObject_dealloc(gemObject* self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
gemObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    gemObject* self = (gemObject*)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->gid.index = 0;
        self->gid.ident.tag = 0;
    }
    return (PyObject*) self;
}

static int
gemObject_init(gemObject *self, PyObject *args, PyObject *kwds)
{
    if (! PyArg_ParseTuple(args, "")) {
        return -1; 
    }
    return 0;
}


/* Here's how you add attributes to your type... */
static PyMemberDef gemObject_members[] = {
    {"index", T_INT, offsetof(gemObject, gid.index), READONLY, "gem id index"},
    {"tag", T_INT, offsetof(gemObject, gid.ident.tag), READONLY, "gem tag"},
    {"ptr", T_OBJECT, offsetof(gemObject, gid.ident.ptr), READONLY, "gem entity ptr"},
    {NULL}  /* Sentinel */
};


// here's how you add functions to your type
// fields are: method name, ptr to c impl, flags indicating how call should be constructed, doc string
// flags are:
//    METH_VARARGS:   expects (PyObject* self_or_module, PyObject *args)
//    METH_KEYWORDS:  expects (PyObject* self_or_module, PyObject *args, PyObject *kwds)
//    METH_NOARGS:    expects (PyObject* self_or_module, PyObject *always_null)
//    METH_O:         expects (PyObject* self_or_module, PyObject *some_object)
//
static PyMethodDef gemObject_methods[] = {
    {NULL, NULL, 0, NULL}        // Sentinel
};

static PyTypeObject gemObjectType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "GEMid",                  /*tp_name*/
    sizeof(gemObject),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)gemObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
//    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "gem object",              /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    gemObject_methods,         /* tp_methods */
    gemObject_members,         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)gemObject_init,  /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
    0,                         /* tp_free */
    0,                         /* tp_is_gc */
    0,                         /* tp_bases */
    0,                         /* tp_mro - method resolution order */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
    0                          /* tp_del */
};

static PyObject* pygem_getEdge(PyObject* module, PyObject* args) {
    gemObject* pybrep;
    int edge;
    
    if (!PyArg_ParseTuple(args, "Oi", &pybrep, &edge)) {
        PyErr_SetString(PyExc_TypeError, "bad args passed to getedge()");
        return NULL;
    }
    
    // FIXME: need to determine if ptr is actually a gemBRep
    gemBRep* brep = pybrep->gid.ident.ptr;
    double tlimit[2];
    int nodes[2];
    int faces[2];
    int num_attrs;
    
    // call actual gem function
    if (gem_getEdge(brep, edge, tlimit, nodes, faces, &num_attrs) != GEM_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "getEdge call failed");
        return NULL;
    }
    
    PyObject* tlimit_tuple = Py_BuildValue("(dd)", tlimit[0], tlimit[1]);
    PyObject* nodes_tuple = Py_BuildValue("(ii)", nodes[0], nodes[1]);
    PyObject* faces_tuple = Py_BuildValue("(ii)", faces[0], faces[1]);
    
    return Py_BuildValue("(OOOi)", tlimit_tuple, nodes_tuple, faces_tuple, num_attrs); 
}


static PyMethodDef pygem_methods[] = {
    {"get_edge",  (PyCFunction)pygem_getEdge, METH_VARARGS,
     "for a given edge in a BRep, return nodes, faces, and number of attributes"},
    {NULL, NULL, 0, NULL}        // Sentinel
};


#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initgem(void) 
{
    PyObject* module;

    gemObjectType.tp_new = gemObject_new;
    if (PyType_Ready(&gemObjectType) < 0)
        return;

    module = Py_InitModule3("gem", pygem_methods,
                            "Module to interface with NPSS");

    import_array(); // loads the numpy C API

    Py_INCREF(&gemObjectType);
    PyModule_AddObject(module, "GEMid", (PyObject *)&gemObjectType);
}


