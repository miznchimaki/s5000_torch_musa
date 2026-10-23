"""define torch musa storage methods """

# pylint: disable=unused-argument,E0202,W0236,I1101
import torch
from torch._utils import _get_async_or_non_blocking
from torch import classproperty
from torch.storage import _LegacyStorage, _warn_typed_storage_removal

from torch_musa import _MUSAC


class _MusaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for MUSA storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for MUSA storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for MUSA storage")


class ByteStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_MusaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


@property
def _is_musa(self):
    return self.device.type == "musa"


def _musa(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in MUSA memory.

    If this object is already in MUSA memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    non_blocking = _get_async_or_non_blocking("musa", non_blocking, kwargs)
    if self.is_musa:
        if device is None:
            device = torch.musa.current_device()
        if self.get_device() == device:
            return self
    else:
        if device is None:
            device = -1
    with torch.musa.device(device):
        if self.is_sparse:
            new_type = getattr(torch.musa.sparse, self.__class__.__name__)
            indices = torch.Tensor._indices(self).musa(device, non_blocking)
            values = torch.Tensor._values(self).musa(device, non_blocking)
            return new_type(indices, values, self.size())

        untyped_storage = torch.UntypedStorage(self.size(), device=torch.device("musa"))
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage


def _edit_device(self, device):
    """
    edit device property of storage.

    Args:
        device (torch.device): the destination device
    """
    _MUSAC._edit_device_musa(self, device)


@classmethod
def _new_with_weak_ptr(cls, *args, **kwargs):
    return _MUSAC._new_with_weak_ptr_musa(*args, **kwargs)


@classmethod
def _new_shared_musa(cls, *args, **kwargs):
    return _MUSAC._new_shared_musa(*args, **kwargs)


def _share_musa(self, *args, **kwargs):
    return _MUSAC._share_musa_(self, *args, **kwargs)


@classmethod
def _release_ipc_counter_musa(cls, *args, device=None, **kwargs):
    return _MUSAC._release_ipc_counter_musa(*args, **kwargs)


def _is_shared(self):
    if self.is_musa:
        return True
    return self.is_shared_old()


def set_storage_attributes():
    """overwrite torch UntypedStorage methods."""
    torch.UntypedStorage._share_musa_ = _share_musa
    torch.UntypedStorage._share_cuda_ = _share_musa

    torch.UntypedStorage._new_shared_musa = _new_shared_musa
    torch.UntypedStorage._new_shared_cuda = _new_shared_musa

    torch.UntypedStorage.musa = _musa
    torch.UntypedStorage.is_musa = _is_musa
    torch.UntypedStorage.is_cuda = _is_musa

    torch.UntypedStorage.is_shared_old = torch.UntypedStorage.is_shared
    torch.UntypedStorage.is_shared = _is_shared

    torch.UntypedStorage._release_ipc_counter_musa = _release_ipc_counter_musa
    torch.UntypedStorage._release_ipc_counter_cuda = _release_ipc_counter_musa

    torch.UntypedStorage.edit_device = _edit_device

    torch.UntypedStorage._new_with_weak_ptr_musa = _new_with_weak_ptr
