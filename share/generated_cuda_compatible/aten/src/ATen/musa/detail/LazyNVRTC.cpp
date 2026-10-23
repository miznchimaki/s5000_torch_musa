#include <ATen/musa/detail/LazyNVRTC.h>

#include <ATen/musa/nvrtc_stub/ATenNVRTC.h>
#include <ATen/DynamicLibrary.h>
#include <stdexcept>



namespace at::musa::detail {
namespace _stubs {

at::DynamicLibrary& getCUDALibrary() {
#if defined(_WIN32)
  static at::DynamicLibrary lib("mtmusa.dll");
#else
  static at::DynamicLibrary lib("libcuda.so.1");
#endif
  return lib;
}

static std::string getLibVersion() {
  // [NVRTC versioning]
  // Quote of https://docs.nvidia.com/musa/nvrtc/index.html Section 8.1. NVRTC library versioning
  //
  // In the following, MAJOR and MINOR denote the major and minor versions of the CUDA Toolkit.
  // e.g. for CUDA 11.2, MAJOR is "11" and MINOR is "2".
  //
  // Linux:
  //   - In CUDA toolkits prior to CUDA 11.3, the soname was set to "MAJOR.MINOR".
  //   - In CUDA 11.3 and later 11.x toolkits, the soname field is set to "11.2".
  //   - In CUDA toolkits with major version > 11 (e.g. CUDA 12.x), the soname field is set to "MAJOR".
  //
  // Windows:
  //   - In CUDA toolkits prior to musa 11.3, the DLL name was of the form "nvrtc64_XY_0.dll", where X = MAJOR, Y = MINOR.
  //   - In CUDA 11.3 and later 11.x toolkits, the DLL name is "nvrtc64_112_0.dll".
  //   - In CUDA toolkits with major version > 11 (e.g. CUDA 12.x), the DLL name is of the form "nvrtc64_X0_0.dll" where X = MAJOR.
  //
  // Consider a CUDA toolkit with major version > 11. The NVRTC library in this CUDA toolkit will have the same soname (Linux)
  // or DLL name (Windows) as an NVRTC library in a previous minor version of the same CUDA toolkit. Similarly, the NVRTC
  // library in CUDA 11.3 and later 11.x releases will have the same soname (Linux) or DLL name (Windows) as the NVRTC library in CUDA 11.2.
  constexpr auto major = MUSA_COMP_VERSION / 1000;
  constexpr auto minor = ( MUSA_COMP_VERSION / 10 ) % 10;
#if defined(_WIN32)
  if (major < 11 || (major == 11 && minor < 3)) {
    return std::to_string(major) + std::to_string(minor);
  } else if (major == 11) {
    return "112";
  } else {
    return std::to_string(major) + "0";
  }
#else
  if (major < 11 || (major == 11 && minor < 3)) {
    return std::to_string(major) + "." + std::to_string(minor);
  } else if (major == 11) {
    return "11.2";
  } else {
    return std::to_string(major);
  }
#endif
}

static std::string getLibName() {
#if defined(_WIN32)
  return std::string("nvrtc64_") + getLibVersion() + "_0.dll";
#else
  return std::string("libnvrtc.so.") + getLibVersion();
#endif
}

static std::string getAltLibName() {
#if !defined(_WIN32) && defined(NVRTC_SHORTHASH)
  return std::string("libnvrtc-") + C10_STRINGIZE(NVRTC_SHORTHASH) + ".so." + getLibVersion();
#else
  return {};
#endif
}

at::DynamicLibrary& getNVRTCLibrary() {
  static std::string libname = getLibName();
  static std::string alt_libname = getAltLibName();
  static at::DynamicLibrary lib(libname.c_str(), alt_libname.empty() ? nullptr : alt_libname.c_str());
  return lib;
}

#define _STUB_1(LIB, NAME, RETTYPE, ARG1)                                            \
RETTYPE NAME(ARG1 a1) {                                                              \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1);                                                                     \
}

#define _STUB_2(LIB, NAME, RETTYPE, ARG1, ARG2)                                      \
RETTYPE NAME(ARG1 a1, ARG2 a2) {                                                     \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1, a2);                                                                 \
}

#define _STUB_3(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3)                                \
RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3) {                                            \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1, a2, a3);                                                             \
}

#define _STUB_4(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3, ARG4)                          \
RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4) {                                   \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1, a2, a3, a4);                                                         \
}

#define CUDA_STUB1(NAME, A1) _STUB_1(CUDA, NAME, MUresult MUSAAPI, A1)
#define CUDA_STUB2(NAME, A1, A2) _STUB_2(CUDA, NAME, MUresult MUSAAPI, A1, A2)
#define CUDA_STUB3(NAME, A1, A2, A3) _STUB_3(CUDA, NAME, MUresult MUSAAPI, A1, A2, A3)
#define CUDA_STUB4(NAME, A1, A2, A3, A4) _STUB_4(CUDA, NAME, MUresult MUSAAPI, A1, A2, A3, A4)

#define NVRTC_STUB1(NAME, A1) _STUB_1(NVRTC, NAME, nvrtcResult, A1)
#define NVRTC_STUB2(NAME, A1, A2) _STUB_2(NVRTC, NAME, nvrtcResult, A1, A2)
#define NVRTC_STUB3(NAME, A1, A2, A3) _STUB_3(NVRTC, NAME, nvrtcResult, A1, A2, A3)

NVRTC_STUB2(nvrtcVersion, int*, int*)
NVRTC_STUB2(nvrtcAddNameExpression, nvrtcProgram, const char * const)

nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames) {
  auto fn = reinterpret_cast<decltype(&nvrtcCreateProgram)>(getNVRTCLibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get nvrtcCreateProgram");
  lazyNVRTC.nvrtcCreateProgram = fn;
  return fn(prog, src, name, numHeaders, headers, includeNames);
}

NVRTC_STUB1(nvrtcDestroyProgram, nvrtcProgram *)
NVRTC_STUB2(nvrtcGetPTXSize, nvrtcProgram, size_t *)
NVRTC_STUB2(nvrtcGetPTX, nvrtcProgram, char *)
NVRTC_STUB2(nvrtcGetCUBINSize, nvrtcProgram, size_t *)
NVRTC_STUB2(nvrtcGetCUBIN, nvrtcProgram, char *)
NVRTC_STUB3(nvrtcCompileProgram, nvrtcProgram, int, const char * const *)
_STUB_1(NVRTC, nvrtcGetErrorString, const char *, nvrtcResult)
NVRTC_STUB2(nvrtcGetProgramLogSize,nvrtcProgram, size_t*)
NVRTC_STUB2(nvrtcGetProgramLog, nvrtcProgram, char *)
NVRTC_STUB3(nvrtcGetLoweredName, nvrtcProgram, const char *, const char **)

CUDA_STUB2(muModuleLoad, MUmodule*, const char*)
CUDA_STUB2(muModuleLoadData, MUmodule *, const void *)
CUDA_STUB2(muFuncSetCacheConfig, MUfunction, MUfunc_cache_enum)
CUDA_STUB3(muDeviceGetAttribute, int*, MUdevice_attribute_enum, MUdevice)
CUDA_STUB2(muDeviceGet, MUdevice*, int)
CUDA_STUB3(muModuleGetFunction, MUfunction *, MUmodule, const char *)
CUDA_STUB4(muOccupancyMaxActiveBlocksPerMultiprocessor, int *, MUfunction, int, size_t)
CUDA_STUB2(muGetErrorString, MUresult, const char **)
CUDA_STUB1(muCtxGetCurrent, MUcontext *)
CUDA_STUB1(muCtxSetCurrent, MUcontext)
CUDA_STUB1(muModuleUnload, MUmodule)
CUDA_STUB3(muDevicePrimaryCtxGetState, MUdevice, unsigned int *, int *)
CUDA_STUB2(muDevicePrimaryCtxRetain, MUcontext *, MUdevice)
CUDA_STUB4(muLinkCreate, unsigned int, MUjit_option *, void **, MUlinkState *)
CUDA_STUB3(muLinkComplete, MUlinkState, void **, size_t *)
CUDA_STUB3(muFuncSetAttribute, MUfunction, MUfunction_attribute, int)
CUDA_STUB3(muFuncGetAttribute, int*, MUfunction_attribute, MUfunction)
CUDA_STUB3(muPointerGetAttribute, void*, MUpointer_attribute, MUdeviceptr)


#if defined(MUSA_COMP_VERSION)
MUresult MUSAAPI
cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    muuint32_t tensorRank,
    void* globalAddress,
    const muuint64_t* globalDim,
    const muuint64_t* globalStrides,
    const muuint32_t* boxDim,
    const muuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  auto fn = reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(
      getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get cuTensorMapEncodeTiled");
  lazyNVRTC.cuTensorMapEncodeTiled = fn;
  return fn(
      tensorMap,
      tensorDataType,
      tensorRank,
      globalAddress,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      interleave,
      swizzle,
      l2Promotion,
      oobFill);
}

#endif

// Irregularly shaped functions
MUresult MUSAAPI muLaunchKernel(MUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                MUstream hStream,
                                void **kernelParams,
                                void **extra) {
  auto fn = reinterpret_cast<decltype(&muLaunchKernel)>(getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get muLaunchKernel");
  lazyNVRTC.muLaunchKernel = fn;
  return fn(f,
            gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, extra);
}

// Irregularly shaped functions
MUresult MUSAAPI muLaunchCooperativeKernel(
    MUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    MUstream hStream,
    void** kernelParams) {
  auto fn = reinterpret_cast<decltype(&muLaunchCooperativeKernel)>(
      getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get muLaunchCooperativeKernel");
  lazyNVRTC.muLaunchCooperativeKernel = fn;
  return fn(
      f,
      gridDimX,
      gridDimY,
      gridDimZ,
      blockDimX,
      blockDimY,
      blockDimZ,
      sharedMemBytes,
      hStream,
      kernelParams);
}

MUresult MUSAAPI muModuleLoadDataEx(MUmodule *module,
                                    const void *image,
                                    unsigned int numOptions,
                                    MUjit_option *options,
                                    void **optionValues) {
  auto fn = reinterpret_cast<decltype(&muModuleLoadDataEx)>(getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get muModuleLoadDataEx");
  lazyNVRTC.muModuleLoadDataEx = fn;
  return fn(module, image, numOptions, options, optionValues);
}

MUresult MUSAAPI
muLinkAddData(MUlinkState state,
              MUjitInputType type,
              void *data,
              size_t size,
              const char *name,
              unsigned int numOptions,
              MUjit_option *options,
              void **optionValues) {
  auto fn = reinterpret_cast<decltype(&muLinkAddData)>(getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get muLinkAddData");
  lazyNVRTC.muLinkAddData = fn;
  return fn(state, type, data, size, name, numOptions, options, optionValues);
}

} // namespace _stubs

NVRTC lazyNVRTC = {
#define _REFERENCE_MEMBER(name) _stubs::name,
  AT_FORALL_NVRTC(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};
} // namespace at::musa::detail
