c10/musa is a core library with CUDA functionality.  It is distinguished
from c10 in that it links against the CUDA library, but like c10 it doesn't
contain any kernels, and consists solely of core functionality that is generally
useful when writing CUDA code; for example, C++ wrappers for the CUDA C API.

**Important notes for developers.** If you want to add files or functionality
to this folder, TAKE NOTE.  The code in this folder is very special,
because on our AMD GPU build, we transpile it into c10/hip to provide a
ROCm environment.  Thus, if you write:

```
// c10/musa/CUDAFoo.h
namespace c10 { namespace musa {

void my_func();

}}
```

this will get transpiled into:


```
// c10/hip/HIPFoo.h
namespace c10 { namespace hip {

void my_func();

}}
```

Thus, if you add new functionality to c10, you must also update `C10_MAPPINGS`
`torch/utils/hipify/cuda_to_hip_mappings.py` to transpile
occurrences of `musa::my_func` to `hip::my_func`.  (At the moment,
we do NOT have a catch all `musa::` to `hip::` namespace conversion,
as not all `musa` namespaces are converted to `hip::`, even though
c10's are.)

Transpilation inside this folder is controlled by `CAFFE2_SPECIFIC_MAPPINGS`
(oddly enough.)  `C10_MAPPINGS` apply to ALL source files.

If you add a new directory to this folder, you MUST update both
c10/musa/CMakeLists.txt and c10/hip/CMakeLists.txt
