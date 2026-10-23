#include <Python.h>

extern PyObject* InitMusaModule(void);

PyMODINIT_FUNC PyInit__MUSAC(void) {
  return InitMusaModule();
}
