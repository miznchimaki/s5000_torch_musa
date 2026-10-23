#ifndef THCP_EVENT_INC
#define THCP_EVENT_INC

#include <ATen/musa/MUSAEvent.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/python_headers.h>

struct THCPEvent : THPEvent {
  at::musa::CUDAEvent cuda_event;
};
extern PyObject* THCPEventClass;

void THCPEvent_init(PyObject* module);

inline bool THCPEvent_Check(PyObject* obj) {
  return THCPEventClass && PyObject_IsInstance(obj, THCPEventClass);
}

#endif // THCP_EVENT_INC
