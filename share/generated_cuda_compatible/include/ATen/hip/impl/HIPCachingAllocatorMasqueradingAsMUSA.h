#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace c10 {
// forward declaration
class DataPtr;
namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

C10_MUSA_API c10::musa::MUSACachingAllocator::CUDAAllocator* get();
C10_MUSA_API void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream);

using c10::musa::MUSACachingAllocator::raw_alloc;
using c10::musa::MUSACachingAllocator::raw_alloc_with_stream;
using c10::musa::MUSACachingAllocator::raw_delete;
using c10::musa::MUSACachingAllocator::init;
using c10::musa::MUSACachingAllocator::getMemoryFraction;
using c10::musa::MUSACachingAllocator::setMemoryFraction;
using c10::musa::MUSACachingAllocator::emptyCache;
using c10::musa::MUSACachingAllocator::enable;
using c10::musa::MUSACachingAllocator::isEnabled;
using c10::musa::MUSACachingAllocator::cacheInfo;
using c10::musa::MUSACachingAllocator::getBaseAllocation;
using c10::musa::MUSACachingAllocator::getDeviceStats;
using c10::musa::MUSACachingAllocator::resetAccumulatedStats;
using c10::musa::MUSACachingAllocator::resetPeakStats;
using c10::musa::MUSACachingAllocator::snapshot;
using c10::musa::MUSACachingAllocator::getCheckpointState;
using c10::musa::MUSACachingAllocator::setCheckpointPoolState;
using c10::musa::MUSACachingAllocator::beginAllocateToPool;
using c10::musa::MUSACachingAllocator::endAllocateToPool;
using c10::musa::MUSACachingAllocator::recordHistory;
using c10::musa::MUSACachingAllocator::recordAnnotation;
using c10::musa::MUSACachingAllocator::pushCompileContext;
using c10::musa::MUSACachingAllocator::popCompileContext;
using c10::musa::MUSACachingAllocator::isHistoryEnabled;
using c10::musa::MUSACachingAllocator::checkPoolLiveAllocations;
using c10::musa::MUSACachingAllocator::attachOutOfMemoryObserver;
using c10::musa::MUSACachingAllocator::attachAllocatorTraceTracker;
using c10::musa::MUSACachingAllocator::releasePool;
using c10::musa::MUSACachingAllocator::createOrIncrefPool;
using c10::musa::MUSACachingAllocator::setUseOnOOM;
using c10::musa::MUSACachingAllocator::setNoSplit;
using c10::musa::MUSACachingAllocator::getPoolUseCount;
using c10::musa::MUSACachingAllocator::getIpcDevPtr;
using c10::musa::MUSACachingAllocator::shareIpcHandle;
using c10::musa::MUSACachingAllocator::name;
using c10::musa::MUSACachingAllocator::memcpyAsync;
using c10::musa::MUSACachingAllocator::enablePeerAccess;
} // namespace HIPCachingAllocatorMasqueradingAsCUDA
} // namespace hip
} // namespace c10

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
