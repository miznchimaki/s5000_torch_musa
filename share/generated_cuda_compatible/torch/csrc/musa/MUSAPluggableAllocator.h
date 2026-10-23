#pragma once

#include <c10/core/Allocator.h>
#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"
#include <c10/musa/MUSAMacros.h>
#include "torch_musa/csrc/core/MUSAStream.h"

#include "torch_musa/csrc/core/MUSACachingAllocator.h"

#include <mutex>

namespace torch::musa::CUDAPluggableAllocator {

TORCH_CUDA_CPP_API std::shared_ptr<
    c10::musa::MUSACachingAllocator::CUDAAllocator>
getCurrentAllocator();
TORCH_CUDA_CPP_API std::shared_ptr<
    c10::musa::MUSACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, musaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, musaStream_t)> free_fn);
TORCH_CUDA_CPP_API void changeCurrentAllocator(
    const std::shared_ptr<c10::musa::MUSACachingAllocator::CUDAAllocator>&
        allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      musaStream_t stream);
  size_t size;
  c10::DeviceIndex device_idx;
  musaStream_t stream{};
};

struct TORCH_CUDA_CPP_API CUDAPluggableAllocator
    : public c10::musa::MUSACachingAllocator::CUDAAllocator {
  CUDAPluggableAllocator(
      std::function<void*(size_t, int, musaStream_t)> alloc_fn,
      std::function<void(void*, size_t, int, musaStream_t)> free_fn);

  CUDAPluggableAllocator(CUDAPluggableAllocator& other);
  CUDAPluggableAllocator(CUDAPluggableAllocator&& other) = delete;
  CUDAPluggableAllocator& operator=(const CUDAPluggableAllocator& other) =
      delete;
  CUDAPluggableAllocator& operator=(CUDAPluggableAllocator&& other) = delete;
  ~CUDAPluggableAllocator() override = default;

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(
      std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(
      std::function<void(void* ptr, musaStream_t stream)> record_stream_fn);

  void set_begin_allocate_to_pool(
      std::function<
          void(int, c10::musa::MempoolId_t, std::function<bool(musaStream_t)>)>
          capture_begin_fn);

  void set_end_allocate_to_pool_fn(
      std::function<void(int, c10::musa::MempoolId_t)> capture_about_to_end_fn);

  void set_release_pool(
      std::function<void(int, c10::musa::MempoolId_t)> capture_destroy_fn);

  void* malloc(size_t size, c10::DeviceIndex device, musaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, musaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  double getMemoryFraction(c10::DeviceIndex device) override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  std::vector<c10::musa::MUSACachingAllocator::StreamSegmentSize>
  getExpandableSegmentSizes(c10::DeviceIndex device) override;
  void emptyCache(c10::musa::MempoolId_t mempool_id = {0, 0}) override;
  void enable(bool /*value*/) override {}
  bool isEnabled() const override {
    return true;
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr& /*ptr*/, c10::musa::MUSAStream stream)
      override;

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  c10::musa::MUSACachingAllocator::SnapshotInfo snapshot(
      c10::musa::MempoolId_t mempool,
      bool include_traces = true) override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      c10::musa::MempoolId_t mempool_id,
      std::function<bool(musaStream_t)> /*filter*/) override;
  void endAllocateToPool(
      c10::DeviceIndex device,
      c10::musa::MempoolId_t mempool_id) override;
  void releasePool(c10::DeviceIndex device, c10::musa::MempoolId_t mempool_id)
      override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  c10::musa::MUSACachingAllocator::ShareableHandle shareIpcHandle(
      void* /*ptr*/) override;
  void recordHistory(
      bool enabled,
      c10::musa::MUSACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::musa::MUSACachingAllocator::RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) override;
  void attachOutOfMemoryObserver(
      c10::musa::MUSACachingAllocator::OutOfMemoryObserver observer) override;
  void attachAllocatorTraceTracker(
      c10::musa::MUSACachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<c10::musa::MUSACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, at::musa::MempoolId_t id)
      override;
  c10::musa::MUSACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::musa::MUSACachingAllocator::AllocatorState> pps)
      override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override;
  musaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      musaStream_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

 protected:
  std::function<void*(size_t, int, musaStream_t)> alloc_fn_;
  std::function<void(void*, size_t, int, musaStream_t)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, musaStream_t stream)> record_stream_fn_;
  std::function<
      void(int, c10::musa::MempoolId_t, std::function<bool(musaStream_t)>)>
      begin_allocate_to_pool_fn_;
  std::function<void(int, c10::musa::MempoolId_t)> end_allocate_to_pool_fn_;
  std::function<void(int, c10::musa::MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookkeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch::musa::CUDAPluggableAllocator
