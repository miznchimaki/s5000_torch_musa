#pragma once

#include <mutlass/arch/arch.h>
#include <mutlass/arch/mma.h>
#include <mutlass/bfloat16.h>
#include <mutlass/mutlass.h>
#include <mutlass/gemm/gemm.h>
#include <mutlass/layout/matrix.h>

#include <ATen/native/musa/cutlass_extensions/arch/mma.h>
#include <ATen/native/musa/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h>

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename TypeA, typename TypeB, typename arch, typename Enable = void>
struct MixedGemmArchTraits {
};

template<typename arch>
struct MixedGemmArchTraits<float, float, arch> {
    static constexpr int Stages = 2;
    using OperatorClass         = mutlass::arch::OpClassSimt;
    using AccType               = float;
    using LayoutB               = mutlass::layout::RowMajor;

    static constexpr int ElementsPerAccessA = 1;
    static constexpr int ElementsPerAccessB = 1;
    static constexpr int ElementsPerAccessC = 1;
    static constexpr int ThreadblockK       = 8;
    using InstructionShape                  = mutlass::gemm::GemmShape<1, 1, 1>;

    using Operator = mutlass::arch::OpMultiplyAdd;
};

// ========================= Volta Traits ===========================
// Volta will always dequantize after the global memory load.
// This will instantiate any HMMA tensorcore kernels for Volta.
// Note that volta does not have native bfloat support so weights and activations will be casted to fp16
// and compute will happen in fp16 then will be converted for bf16 output.
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    mutlass::arch::Sm70,
    typename mutlass::platform::enable_if<mutlass::platform::is_same<TypeA, mutlass::half_t>::value
                                          || mutlass::platform::is_same<TypeA, mutlass::bfloat16_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<TypeB, mutlass::arch::Sm70>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = mutlass::arch::OpClassTensorOp;
    using AccType       = float;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / mutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / mutlass::sizeof_bits<TypeA>::value;
    using InstructionShape                  = mutlass::gemm::GemmShape<8, 8, 4>;

    using Operator = typename LayoutDetails::Operator;
};

// ======================= Turing Traits ==============================
// Note that turing does not have native bfloat support so weights and activations will be casted to fp16
// and compute will happen in fp16 then will be converted for bf16 output.
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    mutlass::arch::Sm75,
    typename mutlass::platform::enable_if<mutlass::platform::is_same<TypeA, mutlass::half_t>::value
                                          || mutlass::platform::is_same<TypeA, mutlass::bfloat16_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<TypeB, mutlass::arch::Sm75>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = mutlass::arch::OpClassTensorOp;
    using AccType       = float;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / mutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / mutlass::sizeof_bits<TypeA>::value;
    using InstructionShape                  = mutlass::gemm::GemmShape<16, 8, 8>;

    using Operator = typename LayoutDetails::Operator;
};

// ======================= Ampere Traits ==============================
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    mutlass::arch::Sm80,
    typename mutlass::platform::enable_if<mutlass::platform::is_same<TypeA, mutlass::half_t>::value
                                          || mutlass::platform::is_same<TypeA, mutlass::bfloat16_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<TypeB, mutlass::arch::Sm80>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = mutlass::arch::OpClassTensorOp;
    using AccType       = float;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / mutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / mutlass::sizeof_bits<TypeA>::value;
    using InstructionShape                  = mutlass::gemm::GemmShape<16, 8, 16>;

    using Operator = typename LayoutDetails::Operator;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
