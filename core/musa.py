"""musa related functions and attributes"""

# pylint: disable=C0103

import torch_musa


class muBLASModule:
    """
    This class name matched with cuBLASModule, while in torch_musa
    most of matmul kernels are implemented in muDNN.
    """

    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch_musa._MUSAC._get_mublas_allow_tf32()
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch_musa._MUSAC._set_mublas_allow_tf32(value)
        raise AttributeError("Unknown attribute " + name)


matmul = muBLASModule()
