"""Record-sized DCP reads sharing a temporary-memory budget in one pod."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import io
import math
from pathlib import Path, PurePosixPath
import tempfile
import time

from loguru import logger
from torch.distributed.checkpoint import FileSystemReader, LoadPlan, LoadPlanner
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, TensorStorageMetadata
from torch.futures import Future


class PodCheckpointReadBudget:
    """Admit decoded checkpoint records across processes sharing a pod's /tmp.

    Reservations release after failure or process exit. Oversized records fail
    before reading. All readers sharing the directory must use the same budget.
    """

    def __init__(self, memory_bytes: int, directory: Path | None = None) -> None:
        if memory_bytes <= 0:
            raise ValueError("Megatron checkpoint load memory budget must be positive")
        self.memory_bytes = memory_bytes
        self.directory = directory or Path(tempfile.gettempdir()) / "skyrl-megatron-dcp-read-memory"
        self.directory.mkdir(exist_ok=True)
        self.admission_path = self.directory / "admission.lock"

    def _reserved_bytes(self, reservation_path: Path) -> int:
        used_bytes = 0
        for path in self.directory.glob("*.reservation"):
            if path == reservation_path:
                continue
            with path.open("r+") as peer:
                try:
                    fcntl.flock(peer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    contents = peer.read().split()
                    if contents:
                        budget, amount = map(int, contents)
                        if budget != self.memory_bytes:
                            raise ValueError("Megatron checkpoint readers in one pod need the same budget")
                        used_bytes += amount
                else:
                    path.unlink()
        return used_bytes

    @contextmanager
    def reserve(self, memory_bytes: int) -> Generator[None, None, None]:
        if memory_bytes > self.memory_bytes:
            raise ValueError(
                f"DCP record needs {memory_bytes} temporary bytes, exceeding the pod budget "
                f"of {self.memory_bytes}; increase trainer.distributed.megatron_checkpoint_load_memory_gib"
            )
        with self.admission_path.open("a+") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            # Publish only locked files: a peer must not reclaim a new reservation
            # between its creation and acquiring the process-lifetime lock.
            reservation = tempfile.NamedTemporaryFile(dir=self.directory, suffix=".reservation", delete=False)
            fcntl.flock(reservation, fcntl.LOCK_EX)
        with reservation:
            reservation_path = Path(reservation.name)
            try:
                while True:
                    with self.admission_path.open("a+") as admission:
                        fcntl.flock(admission, fcntl.LOCK_EX)
                        used_bytes = self._reserved_bytes(reservation_path)
                        if used_bytes + memory_bytes <= self.memory_bytes:
                            reservation.write(f"{self.memory_bytes} {memory_bytes}".encode())
                            reservation.flush()
                            break
                    time.sleep(0.05)
                yield
            finally:
                with self.admission_path.open("a+") as admission:
                    fcntl.flock(admission, fcntl.LOCK_EX)
                    reservation_path.unlink()


class BudgetedCheckpointReader(FileSystemReader):
    """Use PyTorch's reader and planner with admission around each record copy."""

    def __init__(self, path: str, budget: PodCheckpointReadBudget) -> None:
        super().__init__(path)
        self.budget = budget
        self.record_memory: dict[MetadataIndex, int] = {}
        self.storage_read_seconds = 0.0
        self.decode_copy_seconds = 0.0
        self.admission_wait_seconds = 0.0

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool, *args, **kwargs) -> None:
        super().set_up_storage_reader(metadata, is_coordinator, *args, **kwargs)
        # The supported writer emits uncompressed torch.save tensor records and
        # raw BYTE_IO records. Transforms can expand beyond their saved length.
        chunk_bytes = {
            (name, chunk.offsets): math.prod(chunk.sizes) * description.properties.dtype.itemsize
            for name, description in metadata.state_dict_metadata.items()
            if isinstance(description, TensorStorageMetadata)
            for chunk in description.chunks
        }
        for index, storage in self.storage_data.items():
            if not isinstance(storage, _StorageInfo) or storage.transform_descriptors:
                raise ValueError("Megatron restore requires untransformed torch_dist DCP records")
            path = PurePosixPath(storage.relative_path)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".distcp":
                raise ValueError(f"Unsupported DCP record path: {storage.relative_path}")
            tensor_metadata = metadata.state_dict_metadata[index.fqn]
            decoded_bytes = storage.length
            if isinstance(tensor_metadata, TensorStorageMetadata):
                decoded_bytes = max(decoded_bytes, chunk_bytes[index.fqn, index.offset])
            # Allow a wire/read buffer, a decoded storage, and a temporary copy.
            # Use the saved length too: torch.save can include a larger backing
            # storage than the logical chunk. Destination tensors are resident.
            self.record_memory[index] = storage.length + 2 * decoded_bytes

    def _slice_file(self, file, storage: _StorageInfo):
        # One bounded range read avoids many tiny S3 requests from torch.load's
        # zip-file seeks. Its buffer stays covered by the record reservation.
        started = time.monotonic()
        buffer = io.BytesIO(super()._slice_file(file, storage).read())
        self.storage_read_seconds += time.monotonic() - started
        return buffer

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        serialized_bytes = 0
        for item in plan.items:
            waiting_since = time.monotonic()
            with self.budget.reserve(self.record_memory[item.storage_index]):
                admitted_at = time.monotonic()
                self.admission_wait_seconds += admitted_at - waiting_since
                storage_before = self.storage_read_seconds
                # Returning from the parent releases its last decoded tensor
                # before this reservation ends or the next record is admitted.
                super().read_data(replace(plan, items=[item]), planner).wait()
                self.decode_copy_seconds += (
                    time.monotonic() - admitted_at - (self.storage_read_seconds - storage_before)
                )
                serialized_bytes += self.storage_data[item.storage_index].length
        logger.info(
            "DCP records={} serialized_bytes={} storage_read={:.3f}s decode_copy={:.3f}s admission_wait={:.3f}s",
            len(plan.items),
            serialized_bytes,
            self.storage_read_seconds,
            self.decode_copy_seconds,
            self.admission_wait_seconds,
        )
        future: Future[None] = Future()
        future.set_result(None)
        return future
