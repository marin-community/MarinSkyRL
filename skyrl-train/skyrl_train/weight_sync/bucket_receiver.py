"""Storage-preserving bucket install/replay for qualified TP1 TRITON Grug models.

This layer owns no CUDA allocation, communication or model discovery. The native
caller must first qualify the actual model/backend, expert maps and headroom,
then supply its two transfer buffers and independent installed-parameter map.
"""

from collections.abc import Mapping, Sequence

import torch

from skyrl_train.weight_sync.byte_replay import ByteComparison, ReceiverByteCoverage, compare_installed_views
from skyrl_train.weight_sync.expert_scatter import grug_expert_views
from skyrl_train.weight_sync.manifest import ManifestEntry, PublicationManifest, unpack_bucket


class GrugBucketReceiver:
    def __init__(
        self,
        manifest: PublicationManifest,
        parameters: Mapping[str, torch.Tensor],
        expert_maps: Mapping[str, Sequence[int]],
        buffers: tuple[torch.Tensor, torch.Tensor],
        *,
        backend: str,
        tensor_parallel_size: int,
    ):
        if backend != "TRITON" or tensor_parallel_size != 1:
            raise ValueError("Bucket receiver requires qualified unquantized TP1 TRITON Grug layout")
        if not manifest.entries or len(buffers) != 2:
            raise ValueError("Bucket receiver requires a nonempty manifest and two buffers")
        self.manifest = manifest
        self.parameters = dict(parameters)
        self.expert_maps = {name: tuple(mapping) for name, mapping in expert_maps.items()}
        self.buffers = buffers
        self.backend = backend
        self._next_install = 0
        self._next_replay = 0
        self._replay_mismatches = 0
        self._replay_bytes = 0
        self._replay_coverage = ReceiverByteCoverage(self.parameters)
        inventory = ReceiverByteCoverage(self.parameters)
        for buffer in buffers:
            if (
                buffer.dtype != torch.uint8
                or buffer.ndim != 1
                or not buffer.is_contiguous()
                or buffer.numel() != manifest.bucket_bytes
            ):
                raise ValueError("Transfer buffers must exactly match the uint8 bucket capacity")
        storages = (*buffers, *self.parameters.values())
        for index, left in enumerate(storages):
            for right in storages[index + 1 :]:
                if left.device != right.device:
                    raise ValueError("Transfer buffers and installed parameters must share one device")
                if max(left.data_ptr(), right.data_ptr()) < min(
                    left.data_ptr() + left.numel() * left.element_size(),
                    right.data_ptr() + right.numel() * right.element_size(),
                ):
                    raise ValueError("Transfer buffers and installed parameter storage must not overlap")
        # Views of the uninitialized transfer buffers suffice for metadata checks.
        # No source byte is read and no installed byte is written during this pass.
        self._bucket_local_bytes = []
        for bucket_id in range(manifest.bucket_count):
            local_bytes = 0
            for source, installed in self._pairs(bucket_id):
                inventory.observe(installed)
                local_bytes += source.numel() * source.element_size()
            self._bucket_local_bytes.append(local_bytes)
        self.expected_bytes = inventory.finish()

    def _entry_pairs(self, entry: ManifestEntry, source: torch.Tensor):
        if entry.expert_start is not None:
            prefix = entry.hf_name.rsplit(".experts.", 1)[0] + ".experts.routed_experts"
            return grug_expert_views(
                entry,
                source,
                self.parameters[prefix + ".w13_weight"],
                self.parameters[prefix + ".w2_weight"],
                self.expert_maps[prefix],
                self.backend,
            )
        installed = self.parameters[entry.hf_name]
        if installed.shape != source.shape or installed.dtype != source.dtype or not installed.is_contiguous():
            raise ValueError("Dense installed parameter must match the complete wire tensor exactly")
        return ((source, installed),)

    def _pairs(self, bucket_id: int):
        buffer = self.buffers[bucket_id % 2]
        for entry, source in unpack_bucket(self.manifest, bucket_id, buffer):
            yield from self._entry_pairs(entry, source)

    def install_bucket(self, bucket_id: int) -> int:
        self.validate_next_bucket(bucket_id, replay=False)
        with torch.no_grad():
            for source, installed in self._pairs(bucket_id):
                installed.copy_(source)
        self._next_install += 1
        return self._bucket_local_bytes[bucket_id]

    def replay_bucket(self, bucket_id: int, scratch: torch.Tensor) -> ByteComparison:
        self.validate_next_bucket(bucket_id, replay=True)
        if (
            scratch.dtype != torch.bool
            or scratch.ndim != 1
            or not scratch.is_contiguous()
            or not 0 < scratch.numel() <= 512 * 1024
            or scratch.device != self.buffers[0].device
        ):
            raise ValueError("Receiver replay reserves at most 512 KiB of independent boolean scratch")
        # Check the entire buffers and inventory before any comparison can write
        # scratch, including bytes belonging to later entries or the other slot.
        for tensor in (*self.buffers, *self.parameters.values()):
            if max(scratch.data_ptr(), tensor.data_ptr()) < min(
                scratch.data_ptr() + scratch.numel(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            ):
                raise ValueError("Replay scratch overlaps transfer or installed storage")
        pairs = tuple(self._pairs(bucket_id))
        result = compare_installed_views(pairs, scratch, expected_bytes=self._bucket_local_bytes[bucket_id])
        for _, installed in pairs:
            self._replay_coverage.observe(installed)
        self._replay_bytes += result.compared_bytes
        self._replay_mismatches += result.mismatches
        self._next_replay += 1
        return result

    def validate_next_bucket(self, bucket_id: int, *, replay: bool) -> None:
        if type(bucket_id) is not int or type(replay) is not bool:
            raise ValueError("Bucket identity and replay phase must be typed explicitly")
        if not replay:
            if bucket_id != self._next_install or bucket_id >= self.manifest.bucket_count:
                raise ValueError("Install must consume every bucket once in manifest order")
            return
        if self._next_install != self.manifest.bucket_count:
            raise ValueError("Frozen replay starts only after every install bucket")
        if bucket_id != self._next_replay or bucket_id >= self.manifest.bucket_count:
            raise ValueError("Replay must consume every bucket once in manifest order")

    def validate_install_complete(self) -> None:
        if self._next_install != self.manifest.bucket_count:
            raise ValueError("Install is missing manifest buckets")

    def finish_replay(self) -> ByteComparison:
        if self._next_replay != self.manifest.bucket_count:
            raise ValueError("Replay is missing manifest buckets")
        if self._replay_coverage.finish() != self._replay_bytes:
            raise ValueError("Replay byte count differs from independent installed coverage")
        return ByteComparison(self._replay_bytes, self._replay_mismatches)
