#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include <c10/util/CallOnce.h>

#include <deque>
#include <vector>

namespace at::musa {

namespace {

DeviceIndex num_gpus = -1;
std::deque<c10::once_flag> device_flags;
std::vector<musaDeviceProp> device_properties;

void initCUDAContextVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    num_gpus = c10::musa::device_count();
    device_flags.resize(num_gpus);
    device_properties.resize(num_gpus);
    return true;
  }();
}

void initDeviceProperty(DeviceIndex device_index) {
  musaDeviceProp device_prop{};
  AT_MUSA_CHECK(musaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

musaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::musa::current_device();
  return getDeviceProperties(device);
}

musaDeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  initCUDAContextVectors();
  if (device == -1)
    device = c10::musa::current_device();
  AT_ASSERT(
      device >= 0 && device < num_gpus,
      "device=",
      static_cast<int>(device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device) {
  initCUDAContextVectors();
  if (device == -1)
    device = c10::musa::current_device();
  AT_ASSERT(
      device >= 0 && device < num_gpus,
      "device=",
      static_cast<int>(device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  AT_ASSERT(
      peer_device >= 0 && peer_device < num_gpus,
      "peer_device=",
      static_cast<int>(peer_device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  int can_access = 0;
  AT_MUSA_CHECK(musaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getMUSADeviceAllocator() {
  return c10::musa::MUSACachingAllocator::get();
}

} // namespace at::musa
