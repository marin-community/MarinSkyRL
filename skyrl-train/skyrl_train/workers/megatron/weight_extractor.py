"""Megatron-to-Hugging Face weight extraction."""

from collections.abc import Iterator

import torch

from skyrl_train.weight_sync import WeightChunk, WeightExtractor
from skyrl_train.weight_sync.weight_extractor import weight_sync_dtype


def _mapping_hf_names(mapping) -> set[str]:
    names = mapping.hf_param
    if isinstance(names, dict):
        names = names.values()
    elif isinstance(names, str):
        names = (names,)
    return {str(name) for name in names}


class MegatronWeightExtractor(WeightExtractor):
    """Export Megatron weights and optionally pack the exported tensors for transport.

    Megatron Bridge conversions may accumulate several conversion tasks before emitting one
    Hugging Face tensor. The bridge must therefore receive the complete conversion stream in one
    invocation. Bucketing happens after conversion so it cannot split grouped expert exports.
    """

    def __init__(
        self,
        bridge,
        actor_module,
        model_type: str,
        enable_bucketing: bool = False,
        bucket_size_threshold_GB: float = 1.0,
    ):
        self.bridge = bridge
        self.actor_module = actor_module
        self.model_type = model_type
        self.enable_bucketing = enable_bucketing
        self.bucket_size_threshold_bytes = int(bucket_size_threshold_GB * 1024**3)

    def _wire_tensor(self, name: str, tensor: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
        return tensor.to(device=device, dtype=weight_sync_dtype(self.model_type, name, dtype), non_blocking=True)

    @staticmethod
    def _chunk(entries: list[tuple[str, torch.Tensor]]) -> WeightChunk:
        return WeightChunk(
            names=[name for name, _ in entries],
            dtypes=[str(tensor.dtype) for _, tensor in entries],
            shapes=[list(tensor.shape) for _, tensor in entries],
            tensors=[tensor for _, tensor in entries],
        )

    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Export every weight and yield transport-sized chunks."""

        device = torch.cuda.current_device()
        if not self.enable_bucketing:
            exported_weights = self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            )
            for name, tensor in exported_weights:
                yield self._chunk([(name, self._wire_tensor(name, tensor, dtype, device))])
            return

        conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)
        expected_grouped_names = set()
        for task in conversion_tasks:
            if getattr(task.mapping, "is_grouped_export", False):
                expected_grouped_names.update(_mapping_hf_names(task.mapping))
        emitted_grouped_names = set()
        exported_weights = self.bridge.export_hf_weights(
            self.actor_module,
            show_progress=False,
            conversion_tasks=conversion_tasks,
        )

        entries: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        for name, tensor in exported_weights:
            if name in expected_grouped_names:
                emitted_grouped_names.add(name)
            tensor = self._wire_tensor(name, tensor, dtype, device)
            tensor_bytes = tensor.numel() * tensor.element_size()
            uses_special_dtype = tensor.dtype != dtype

            if entries and (
                uses_special_dtype
                or tensor.dtype != entries[0][1].dtype
                or bucket_bytes + tensor_bytes > self.bucket_size_threshold_bytes
            ):
                yield self._chunk(entries)
                entries = []
                bucket_bytes = 0

            if uses_special_dtype:
                yield self._chunk([(name, tensor)])
                continue

            entries.append((name, tensor))
            bucket_bytes += tensor_bytes
            if bucket_bytes >= self.bucket_size_threshold_bytes:
                yield self._chunk(entries)
                entries = []
                bucket_bytes = 0

        if entries:
            yield self._chunk(entries)

        missing_grouped_names = expected_grouped_names - emitted_grouped_names
        if missing_grouped_names:
            missing = ", ".join(sorted(missing_grouped_names))
            raise RuntimeError(f"Megatron Bridge omitted grouped weight exports: {missing}")
