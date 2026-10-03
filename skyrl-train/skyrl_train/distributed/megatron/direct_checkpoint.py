from pathlib import Path
from dataclasses import replace
from urllib.parse import urlparse

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.dist_checkpointing.strategies.checkpointable import (
    CheckpointableShardedTensor,
    LocalShardsContainer,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    MCoreSavePlanner,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
    _replace_sharded_keys_with_state_dict_keys,
    _replace_state_dict_keys_with_sharded_keys,
    _restore_dict_types,
    _unwrap_pyt_sharded_tensor,
    mcore_to_pyt_state_dict,
)
from torch.distributed import checkpoint
from torch.distributed.checkpoint import SavePlan, SavePlanner
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.metadata import Metadata

from marinskyrl.remote_io import create_s3_filesystem
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader


class _SchemaGuardedMCoreSavePlanner(MCoreSavePlanner):
    """Reuse DCP plans only when all rank-local write properties are identical."""

    def __init__(self, *, cache_key: str, **kwargs) -> None:
        super().__init__(enable_plan_caching=True, **kwargs)
        self._cached_plans_key = cache_key

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        cached = SavePlanner._cached_save_plan.get(self._cached_plans_key)
        # PyTorch's comparator omits dtype and other TensorProperties.
        if cached is not None and plan == cached:
            return SavePlan([], usable=False)
        self._pending_local_plan = plan
        return plan

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        changed = any(plan.usable for plan in all_plans)
        delta_plans, metadata = super().create_global_plan(all_plans)
        if changed:
            # Scatter every fresh plan if any rank's full schema changed.
            return self.global_plan, metadata
        return delta_plans, metadata


def invalidate_checkpoint_plan_cache(cache_key: str) -> None:
    """Drop every DCP plan/metadata cache entry for one Megatron strategy."""
    for cache in (
        SavePlanner._cached_save_plan,
        SavePlanner._cached_all_plans,
        SavePlanner._cached_global_plan,
        SavePlanner._cached_metadata,
        SavePlanner._cached_final_save_plan,
    ):
        cache.pop(cache_key, None)


def _mcore_to_pyt_save_state_dict(state_dict: dict) -> dict:
    """Use DCP's checkpointable shard path for tensors with prepended axes.

    MCore 0.18 sends these regular-grid tensors through legacy PyTorch
    ShardedTensor construction, which validates a large synthetic global grid
    on every save. The checkpointable path describes only real local shards.
    The expanded local view has the same shape and offsets as the legacy path.
    """
    legacy_keys = {
        key
        for key, shards in state_dict.items()
        if isinstance(shards[0], ShardedTensor)
        and shards[0].prepend_axis_num > 0
        and shards[0].flattened_range is None
        and shards[0].has_regular_grid
    }
    if not legacy_keys:
        return mcore_to_pyt_state_dict(state_dict, False)

    converted = mcore_to_pyt_state_dict(
        {key: shards for key, shards in state_dict.items() if key not in legacy_keys}, False
    )
    for key in legacy_keys:
        normalized_shards = []
        for shard in state_dict[key]:
            if shard.data is None:
                raise ValueError(f"Missing checkpoint tensor data for {key}")
            if shard.prepend_axis_num != state_dict[key][0].prepend_axis_num:
                raise ValueError(f"Inconsistent prepended axes for {key}")
            data = shard.data.detach()
            if not data.is_contiguous():
                data = data.contiguous()
            data = data.view((1,) * shard.prepend_axis_num + shard.local_shape)
            normalized = replace(shard, data=data, local_shape=tuple(data.shape), prepend_axis_num=0)
            normalized_shards.append(CheckpointableShardedTensor.from_sh_ten(normalized))
        converted[key] = LocalShardsContainer(normalized_shards) if len(normalized_shards) > 1 else normalized_shards[0]
    return {key: converted[key] for key in state_dict}


class DirectS3TorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Save MCore torch-dist shards directly to S3 through PyTorch DCP.

    MCore 0.18 has no storage-writer hook, so this adapter preserves its state
    translation and planner while replacing only the writer.
    """

    def __init__(self, checkpoint_dir: str, *, plan_cache_key: str | None = None) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.plan_cache_key = plan_cache_key

    def save(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path) -> None:
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pytorch_state_dict = _mcore_to_pyt_save_state_dict(sharded_state_dict)
        filesystem = create_s3_filesystem()
        writer = StreamingFsspecWriter(self.checkpoint_dir, filesystem=filesystem)
        planner_kwargs = {
            "dedup_replicated_tensors": not self.keep_only_main_replica,
            "flatten_state_dict": False,
            "flatten_sharded_tensors": False,
        }
        planner = (
            _SchemaGuardedMCoreSavePlanner(cache_key=self.plan_cache_key, **planner_kwargs)
            if self.plan_cache_key is not None
            else MCoreSavePlanner(**planner_kwargs)
        )
        try:
            checkpoint.save(pytorch_state_dict, storage_writer=writer, planner=planner)
        except BaseException:
            if self.plan_cache_key is not None:
                invalidate_checkpoint_plan_cache(self.plan_cache_key)
            raise


class DirectS3TorchDistLoadShardedStrategy(TorchDistLoadShardedStrategy):
    """Read the DCP records needed by this rank directly from S3."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__()
        source = urlparse(checkpoint_dir)
        if source.scheme != "s3" or not source.netloc or not source.path.strip("/") or source.query or source.fragment:
            raise ValueError("Megatron restore requires a CoreWeave S3 training checkpoint")
        self.filesystem = create_s3_filesystem(default_cache_type="none")
        endpoint = self.filesystem.endpoint_url or self.filesystem.client_kwargs.get("endpoint_url", "")
        resolved = urlparse(endpoint)
        if (
            resolved.scheme not in ("http", "https")
            or resolved.hostname not in ("cwobject.com", "cwlota.com")
            or resolved.path not in ("", "/")
            or resolved.username
            or resolved.query
            or resolved.fragment
        ):
            raise ValueError("Megatron restore requires a resolved cwobject.com or cwlota.com storage endpoint")
        self.checkpoint_dir = checkpoint_dir

    def load(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path, async_strategy: str = "mcore"):
        del async_strategy  # Required by the Megatron sharded-load strategy interface.
        original = sharded_state_dict
        tensors = [value for value in nested_values(original) if isinstance(value, ShardedTensor)]
        converted, flat_mapping, rename_mapping = _replace_state_dict_keys_with_sharded_keys(original)
        pytorch_state_dict = mcore_to_pyt_state_dict(converted, True)

        reader = RecordCheckpointReader(self.checkpoint_dir)
        reader.fs = FsspecFileSystem()
        reader.fs.fs = self.filesystem
        reader.path = self.checkpoint_dir
        checkpoint.load(
            pytorch_state_dict,
            storage_reader=reader,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=[value for value in tensors if not value.allow_shape_mismatch],
                allow_shape_mismatch_sharded_tensors={
                    value.key: value for value in tensors if value.allow_shape_mismatch
                },
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
            no_dist=True,
        )
        restored = {name: _unwrap_pyt_sharded_tensor(value) for name, value in pytorch_state_dict.items()}
        restored = _replace_sharded_keys_with_state_dict_keys(restored, flat_mapping, rename_mapping)
        _restore_dict_types(restored, original)
        return restored
