#ifndef __SVMAPI_H
#define __SVMAPI_H

#include <Python.h>
#include "pyobjs.h"
#include "svmapi_globals.h"

PyMODINIT_FUNC PyInit_svmapi(void);

extern PyObject *svmapi_usermodule;
extern PyObject *svmapi_thismodule;

#endif // __SVMAPI_METHODS_H
