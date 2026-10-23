#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/record_stream_native.h>
#endif

namespace at::native {
void record_stream_cuda(Tensor& self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  c10::musa::MUSACachingAllocator::recordStream(self.storage().data_ptr(), at::musa::MUSAStream::unpack3(data.stream_id, data.device_index, data.device_type));
}
}  // namespace at::native
