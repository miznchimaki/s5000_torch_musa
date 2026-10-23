#include <ATen/musa/nvrtc_stub/ATenNVRTC.h>

namespace at::musa {

NVRTC* load_nvrtc() {
  auto self = new NVRTC();
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_NVRTC(CREATE_ASSIGN)
  return self;
}

} // at::musa
