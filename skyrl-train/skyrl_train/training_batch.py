"""Training batch containers and iteration."""

import io
import math
import pickle
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, TypedDict

import torch
from jaxtyping import Float, Integer

from skyrl_train.dataset.replay_buffer import Experience

DictType = TypeVar("DictType")
GLOBAL_LOSS_DENOM_METADATA_KEY = "global_loss_denom"


def per_data_parallel_batch_size(mini_batch_size: int, samples_per_prompt: int, data_parallel_size: int) -> int:
    """Return a role's per-data-parallel-rank mini-batch size."""
    if data_parallel_size < 1:
        raise ValueError(f"data_parallel_size must be positive, got {data_parallel_size}")
    return mini_batch_size * samples_per_prompt // data_parallel_size


def gradient_accumulation_steps(per_rank_mini_batch_size: int, micro_batch_size: int) -> int:
    """Return the number of microbatches in one optimizer update."""
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    steps, remainder = divmod(per_rank_mini_batch_size, micro_batch_size)
    if steps < 1 or remainder:
        raise ValueError(
            f"per-rank mini-batch size {per_rank_mini_batch_size} must be a positive multiple "
            f"of micro-batch size {micro_batch_size}"
        )
    return steps


# NOTE (sumanthrh): This is inspired by `TensorDict` but is much simpler.
class TensorBatch(dict, Generic[DictType]):
    """Base class for training batches

    This defines a generic container for a batch of training data (inputs or outputs).
    Consists of a dictionary of tensors along with some metadata.
    """

    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = None
        self._device = None
        self._check_consistency()

    def select(self, keys: List[str], metadata_keys: Optional[List[str]] = None) -> "TensorBatch[DictType]":
        """Select a subset of the data batch.

        Args:
            keys: The keys to select
            metadata_keys: The metadata keys to select

        Returns:
            A new `TensorBatch` object with the selected keys and metadata
        """
        selected_batch_data = {}
        for key in keys:
            selected_batch_data[key] = self[key]
        selected_metadata = {}
        if metadata_keys is None:
            selected_metadata = self.metadata
        else:
            selected_metadata = {}
            for key in metadata_keys:
                selected_metadata[key] = self.metadata[key]
        new_batch = self.__class__(selected_batch_data)
        new_batch.metadata = selected_metadata
        return new_batch

    def _check_consistency(self):
        """Check consistency of all present fields"""
        keys = list(self.keys())
        if len(keys) == 0:
            return

        batch_size = len(self[keys[0]])
        self._batch_size = batch_size
        for key in keys:
            value = self[key]
            if value is None:
                continue
            self._device = value.device if self._device is None else self._device
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Field {key} must be a tensor, got {type(value)}")
            if len(value) != batch_size:
                raise ValueError(f"Batch size mismatch in {key}")
            if value.device != self._device:
                raise ValueError(f"Device mismatch in {key}. Expected {self._device}, got {value.device}")

    def __getitem__(self, index) -> "TensorBatch[DictType]":
        if isinstance(index, slice):
            return self.slice(index.start, index.stop, index.step)
        elif isinstance(index, int):
            return self.slice(index, index + 1)
        else:
            return super().__getitem__(index)

    def __setitem__(self, key: str, value: Optional[torch.Tensor]) -> None:
        if value is None:
            super().__setitem__(key, value)
            return

        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Field {key} must be a tensor, got {type(value)}")

        if hasattr(self, "_batch_size") and self._batch_size is not None and len(value) != self._batch_size:
            raise ValueError(
                f"Batch size mismatch in {key}. Expected tensor to be of size {self._batch_size}, got {len(value)}."
            )

        super().__setitem__(key, value)

        if hasattr(self, "_batch_size") and self._batch_size is None:
            self._batch_size = len(value)

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None, *, non_blocking: bool = False
    ) -> "TensorBatch":
        """Move tensors to device and/or cast to dtype.

        Args:
            device: The device to move the tensors to
            dtype: The dtype to cast the tensors to
            non_blocking: Whether the operation should be non-blocking
        """
        for key, value in self.items():
            if value is None:
                continue
            assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
            self[key] = value.to(device, dtype, non_blocking=non_blocking)
        return self

    @property
    def batch_size(self) -> int:
        """Batch size for the tensors"""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """Get the device for the tensors"""
        return self._device

    def __getstate__(self):
        """Serialize the `TensorBatch` object for pickle protocol"""
        if self._device is not None:
            assert self._device == torch.device("cpu"), "Tensors must be on CPU before serialization"
        batch_dict = {}
        for key, value in self.items():
            value_to_save = value
            if value is not None:
                logical_bytes = value.numel() * value.element_size()
                if value.storage_offset() != 0 or value.untyped_storage().nbytes() > logical_bytes:
                    value_to_save = value.clone(memory_format=torch.contiguous_format)
            buffer = io.BytesIO()
            torch.save(value_to_save, buffer)
            batch_dict[key] = buffer.getvalue()

        return {
            "batch_dict": batch_dict,
            "batch_size": self._batch_size,
            "device": self._device,
            "metadata": self.metadata,
        }

    def __reduce_ex__(self, _protocol: int):
        """Serialize only the compact custom state, not the inherited dict payload."""
        return self.__class__, (), self.__getstate__()

    def __setstate__(self, state):
        """Deserialize the `TensorBatch` object and load it into memory"""
        for key, value in state["batch_dict"].items():
            buffer = io.BytesIO(value)
            self[key] = torch.load(buffer)

        self._batch_size = state["batch_size"]
        self._device = state["device"]
        self.metadata = state["metadata"]
        self._check_consistency()
        return self

    def repeat(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat` (and `numpy.tile`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            else:
                assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat(repeats)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def repeat_interleave(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat_interleave` (and `numpy.repeat`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            else:
                assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat_interleave(repeats)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def chunk(self, chunk_size: int) -> List["TensorBatch[DictType]"]:
        """Split into smaller chunks"""
        chunks = []
        for i in range(0, self.batch_size, chunk_size):
            chunk_data = {}
            for key, value in self.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        chunk_data[key] = value[i : i + chunk_size]
                    else:
                        raise ValueError(f"Unsupported type {type(value)} for key {key}")
                else:
                    # `None` values are not chunked
                    chunk_data[key] = value
            chunk = self.__class__(chunk_data)
            chunk.metadata = self.metadata
            chunks.append(chunk)
        return chunks

    def slice(self, start: int, end: int, step: int = 1) -> "TensorBatch[DictType]":
        """Slice the data batch.

        Args:
            start: The start index
            end: The end index
            step: The step size

        Returns:
            A new `TensorBatch` object with the view of the specified slice.
        """
        slice_obj = slice(start, end, step)
        sliced_data = {}
        for key, value in self.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    sliced_data[key] = value[slice_obj]
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                # `None` values are not sliced
                sliced_data[key] = value
        sliced_batch = self.__class__(sliced_data)
        sliced_batch.metadata = self.metadata
        return sliced_batch

    def save(self, path: str):
        """Save the data to a pickle file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the data from a pickle file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def cat(cls, shards: List["TensorBatch[DictType]"]) -> "TensorBatch[DictType]":
        """Concatenate shards.

        Args:
            shards: The list of `TensorBatch` objects to cat

        Returns:
            A new `TensorBatch` object with the concatenated data
        """
        cat_data = {}
        assert len(shards) > 0, "Cannot cat an empty list of shards"
        for key, value in shards[0].items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    cat_data[key] = torch.cat([shard[key] for shard in shards])
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                # `None` values are not cat'd
                cat_data[key] = value
        metadata = shards[0].metadata
        cat_batch = cls(cat_data)
        cat_batch.metadata = metadata
        return cat_batch

    def __len__(self) -> int:
        """Length of the batch.

        Note that this is the same as the batch size rather than the number of keys in the batch.
        """
        return self._batch_size

    def __eq__(self, other: Any) -> bool:
        """Check if two `TensorBatch` objects are equal"""
        if not isinstance(other, TensorBatch):
            return False
        if self.metadata != other.metadata:
            return False
        if len(self) != len(other):
            return False
        if len(self.items()) != len(other.items()):
            return False
        for k, v in self.items():
            if k not in other or not torch.equal(v, other[k]):
                return False
        return True

    def __str__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return f"TensorBatch(batch_size={self.batch_size}, device={self.device}, metadata={self.metadata}), items={self.items()}"

    def __repr__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return self.__str__()


class TrainingInput(TypedDict, total=False):
    """Schema for training input batch"""

    sequences: Integer[torch.Tensor, "batch_size seq_len"]
    attention_mask: Integer[torch.Tensor, "batch_size seq_len"]
    loss_mask: Integer[torch.Tensor, "batch_size seq_len"]
    response_mask: Integer[torch.Tensor, "batch_size seq_len"]
    action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    base_action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    values: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    returns: Float[torch.Tensor, "batch_size seq_len"]
    advantages: Float[torch.Tensor, "batch_size seq_len"]
    kl: Float[torch.Tensor, "batch_size seq_len"]
    rewards: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    # Teacher distillation fields (populated by DistillationTrainer when teacher engine is configured)
    teacher_top_k_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len K"]]
    teacher_top_k_indices: Optional[Integer[torch.Tensor, "batch_size seq_len K"]]
    # MoE router-replay capture rail (Stage 1): per-token expert-selection indices
    # captured from vLLM, [batch, response_len, L, K] (L = MoE layers, K = top-k).
    # Present only when trainer.policy.fsdp_config.moe_router_replay is True. >2-D
    # per-token precedent: teacher_top_k_* above; TensorBatch._check_consistency
    # only validates dim-0, so 4-D is accepted. Consumed in Stage 2 (replay), not here.
    rollout_routed_experts: Optional[Integer[torch.Tensor, "batch_size seq_len L K"]]
    # Loop-behavior reward shaping (Stage B / F5): per-token additive shaping
    # channel, SEPARATE from `rewards` (the RLOO-N outcome term). Default all-zeros
    # and present ONLY when trainer.algorithm.enable_token_reward_channel is True,
    # so the flag-off TrainingInputBatch keyset is byte-identical to today. The
    # combiner that ADDS this into the advantage is registered in Stage C; Stage B
    # only makes it flow as zeros (no-op).
    token_level_shaping: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    # Loop-behavior reward shaping (Stage B / F4): per-token span tags
    # {OTHER=0, THINK=1, ACTION=2, EDIT=3}, aligned 1:1 with the response tokens
    # (same exact-token-id layout TIS uses). Present only when the channel is on.
    response_span_tags: Optional[Integer[torch.Tensor, "batch_size seq_len"]]


class TrainingInputBatch(TensorBatch[TrainingInput]):
    """Training input data"""

    pass


class TrainingBatchIterator(Iterator[Experience]):
    """Yield reusable ``Experience`` microbatches from a training batch."""

    def __init__(self, data: TrainingInputBatch, sample_batch_size: int):
        if sample_batch_size < 1:
            raise ValueError(f"sample_batch_size must be positive, got {sample_batch_size}")
        self._chunks = data.chunk(sample_batch_size)
        self._iterator = iter(self._chunks)
        self._length = math.ceil(data.batch_size / sample_batch_size)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> "TrainingBatchIterator":
        return self

    def __next__(self) -> Experience:
        try:
            return self._experience(next(self._iterator))
        except StopIteration:
            self._iterator = iter(self._chunks)
            raise

    @staticmethod
    def _experience(batch: TrainingInputBatch) -> Experience:
        return Experience(
            sequences=batch["sequences"],
            action_log_probs=batch["action_log_probs"],
            base_action_log_probs=batch["base_action_log_probs"],
            values=batch["values"],
            returns=batch["returns"],
            advantages=batch["advantages"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            action_mask=batch["response_mask"],
            num_actions=batch.metadata["response_length"],
            rollout_logprobs=batch.get("rollout_logprobs"),
            rollout_routed_experts=batch.get("rollout_routed_experts"),
            response_span_tags=batch.get("response_span_tags"),
            info={},
            metadata=batch.metadata,
        )


class TrainingOutputBatch(TensorBatch[Dict[str, torch.Tensor]]):
    """Training output data"""

    pass
