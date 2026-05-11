import atexit
import logging
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.distributed.naive_distributed import get_naive_distributed
from sglang.srt.utils import check_cuda_result

logger = logging.getLogger(__name__)


SHM_ATTACH_TIMEOUT_S = 5.0
SHM_ATTACH_RETRY_INTERVAL_S = 0.01
SHM_CREATE_RETRY_ATTEMPTS = 3


class HostSharedMemoryManager:
    def __init__(self, base_name: str):
        self._base_name = Path(base_name)
        self._operation_index = 0
        self._records: List[_Record] = []
        self._closed = False
        atexit.register(self.close)

    def malloc(self, *, shape, dtype):
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self._malloc_raw(num_bytes=meta_tensor.nbytes)
        return raw.view(dtype).view(*shape)

    def _malloc_raw(self, *, num_bytes: int) -> torch.Tensor:
        self._operation_index += 1
        shm_name = f"{self._base_name}_op{self._operation_index}"

        if get_naive_distributed().get_rank() == 0:
            shm = self._create_rank0_shm(shm_name, num_bytes)

        get_naive_distributed().barrier()

        if get_naive_distributed().get_rank() != 0:
            shm = self._open_non_owner_shm(shm_name)

        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        tensor = torch.from_numpy(np_array)

        self._cuda_host_register(tensor.data_ptr(), num_bytes)

        get_naive_distributed().barrier()

        self._records.append(
            _Record(
                shm=shm,
                np_array=np_array,
                tensor=tensor,
            )
        )
        return tensor

    def _create_rank0_shm(self, shm_name: str, num_bytes: int) -> shared_memory.SharedMemory:
        for attempt in range(1, SHM_CREATE_RETRY_ATTEMPTS + 1):
            try:
                return shared_memory.SharedMemory(
                    name=shm_name, create=True, size=num_bytes
                )
            except FileExistsError:
                logger.warning(
                    "Found stale host shared memory segment '%s'; unlinking and retrying create.",
                    shm_name,
                )
                self._cleanup_stale_shm(shm_name)
                if attempt == SHM_CREATE_RETRY_ATTEMPTS:
                    raise
                time.sleep(SHM_ATTACH_RETRY_INTERVAL_S)

    def _open_non_owner_shm(self, shm_name: str) -> shared_memory.SharedMemory:
        deadline = time.monotonic() + SHM_ATTACH_TIMEOUT_S
        while True:
            try:
                return shared_memory.SharedMemory(name=shm_name)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(SHM_ATTACH_RETRY_INTERVAL_S)

    def _cleanup_stale_shm(self, shm_name: str) -> bool:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            return False

        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                return False
        return True

    def _cuda_host_register(self, data_ptr: int, num_bytes: int) -> None:
        import cuda.bindings.runtime as cuda_rt

        check_cuda_result(
            cuda_rt.cudaHostRegister(
                data_ptr, num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )

    def _cuda_host_unregister(self, data_ptr: int) -> None:
        import cuda.bindings.runtime as cuda_rt

        check_cuda_result(cuda_rt.cudaHostUnregister(data_ptr))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        should_unlink = False
        try:
            should_unlink = get_naive_distributed().get_rank() == 0
        except Exception:
            # Best-effort fallback for shutdown paths where the distributed context
            # is already torn down: close local handles and try unlinking.
            should_unlink = True

        for record in reversed(self._records):
            try:
                self._cuda_host_unregister(record.tensor.data_ptr())
            except Exception:
                logger.exception("Failed to unregister host shared memory buffer.")

            try:
                record.shm.close()
            except Exception:
                logger.exception("Failed to close host shared memory buffer.")

            if should_unlink:
                try:
                    record.shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.exception("Failed to unlink host shared memory buffer.")

        self._records.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass
class _Record:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray
    tensor: torch.Tensor


# Can have multi instances if needed
_instance: Optional[HostSharedMemoryManager] = None


def get_host_shared_memory_manager():
    assert _instance is not None
    return _instance


def set_host_shared_memory_manager(instance: HostSharedMemoryManager):
    global _instance
    assert _instance is None
    _instance = instance
