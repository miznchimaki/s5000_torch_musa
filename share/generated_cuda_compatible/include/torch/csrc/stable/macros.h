#include <torch/csrc/stable/c/shim.h>

#include <stdexcept>
#include <string>

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

// Users of this macro are expected to include musa_runtime.h
#define STD_MUSA_CHECK(EXPR)                      \
  do {                                            \
    const musaError_t __err = EXPR;               \
    char* __error_msg = nullptr;                  \
    torch_c10_musa_check_msg(                     \
        static_cast<int32_t>(__err),              \
        __FILE__,                                 \
        __func__,                                 \
        static_cast<uint32_t>(__LINE__),          \
        true,                                     \
        &__error_msg);                            \
    if (__error_msg != nullptr) {                 \
      std::string __msg(__error_msg);             \
      torch_c10_musa_free_error_msg(__error_msg); \
      throw std::runtime_error(__msg);            \
    }                                             \
  } while (0)

// Users of this macro are expected to include musa_runtime.h
#define STD_MUSA_KERNEL_LAUNCH_CHECK() STD_MUSA_CHECK(musaGetLastError())

#endif
