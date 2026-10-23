#include <gtest/gtest.h>

#include <c10/musa/impl/CUDATest.h>

using namespace c10::musa::impl;

TEST(CUDATest, SmokeTest) {
  c10_cuda_test();
}
