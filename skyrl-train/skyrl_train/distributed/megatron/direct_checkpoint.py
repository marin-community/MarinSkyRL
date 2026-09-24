import cProfile
import json
import os
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from loguru import logger
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
from torch import distributed as dist
from torch.distributed.checkpoint import FileSystemReader, SavePlan, SavePlanner
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.metadata import Metadata

from skyrl_train.io.s3fs import get_s3_fs, s3_refresh_if_expiring
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter
from skyrl_train.checkpoint_listing import extract_step_from_path
from skyrl_train.timing_observability import checkpoint_phase


def _profile_mcore_conversion(rank: int, step: int | None) -> cProfile.Profile | None:
    """Opt-in diagnostic for one warm save; never enabled in production by default."""
    requested_step = os.environ.get("CHECKPOINT_MCORE_PROFILE_STEP")
    requested_ranks = os.environ.get("CHECKPOINT_MCORE_PROFILE_RANKS", "0")
    if requested_step is None or str(step) != requested_step:
        return None
    if str(rank) not in requested_ranks.split(","):
        return None
    return cProfile.Profile()


def _log_mcore_conversion_profile(
    profile: cProfile.Profile,
    rank: int,
    step: int | None,
    wall_seconds: float,
    process_cpu_seconds: float,
    converted_state_dict: dict,
) -> None:
    def describe(entry):
        code = entry.code
        name = f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}" if hasattr(code, "co_filename") else str(code)
        return {
            "function": name,
            "calls": entry.callcount,
            "self_seconds": round(entry.inlinetime, 4),
            "cumulative_seconds": round(entry.totaltime, 4),
        }

    entries = profile.getstats()
    logger.info(
        "checkpoint_mcore_profile {}",
        json.dumps(
            {
                "rank": rank,
                "step": step,
                "wall_seconds": round(wall_seconds, 4),
                "process_cpu_seconds": round(process_cpu_seconds, 4),
                "converted_key_count": len(converted_state_dict),
                "byte_object_count": sum(isinstance(value, BytesIO) for value in converted_state_dict.values()),
                "byte_object_bytes": sum(
                    value.getbuffer().nbytes for value in converted_state_dict.values() if isinstance(value, BytesIO)
                ),
                "top_self": [
                    describe(entry) for entry in sorted(entries, key=lambda entry: entry.inlinetime, reverse=True)[:25]
                ],
                "top_cumulative": [
                    describe(entry) for entry in sorted(entries, key=lambda entry: entry.totaltime, reverse=True)[:25]
                ],
            },
            sort_keys=True,
        ),
    )


# MCore 0.18 does not expose a storage-writer hook. Keep the adapter narrow: it
# preserves MCore's state translation and planner and replaces only the writer.
class _SchemaGuardedMCoreSavePlanner(MCoreSavePlanner):
    """Reuse DCP plans only when all rank-local write properties are identical."""

    def __init__(self, *, cache_key: str, **kwargs) -> None:
        super().__init__(enable_plan_caching=True, **kwargs)
        self._cached_plans_key = cache_key

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        cached = SavePlanner._cached_save_plan.get(self._cached_plans_key)
        # PyTorch's comparator omits dtype and other TensorProperties. Dataclass
        # equality also checks those properties and MCore's planner data.
        if cached is not None and plan == cached:
            return SavePlan([], usable=False)
        self._pending_local_plan = plan
        return plan

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        changed = any(plan.usable for plan in all_plans)
        delta_plans, metadata = super().create_global_plan(all_plans)
        if changed:
            # The upstream delta comparator also omits TensorProperties. Scatter
            # every fresh plan if any rank's full schema changed.
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
    """Save MCore torch-dist shards directly to S3 through PyTorch DCP."""

    def __init__(self, checkpoint_dir: str, *, plan_cache_key: str | None = None) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.plan_cache_key = plan_cache_key

    def save(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path) -> None:
        rank = dist.get_rank()
        step = extract_step_from_path(os.path.dirname(self.checkpoint_dir.rstrip("/")))
        conversion_profile = _profile_mcore_conversion(rank, step)
        with checkpoint_phase("megatron", "save", "dcp_translate", rank=rank, step=step):
            sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
                sharded_state_dict, self.keep_only_main_replica
            )
            if conversion_profile is not None:
                conversion_started = time.perf_counter()
                conversion_cpu_started = time.process_time()
                conversion_profile.enable()
            try:
                pytorch_state_dict = _mcore_to_pyt_save_state_dict(sharded_state_dict)
            finally:
                if conversion_profile is not None:
                    conversion_profile.disable()
                    conversion_wall_seconds = time.perf_counter() - conversion_started
                    conversion_process_cpu_seconds = time.process_time() - conversion_cpu_started
        if conversion_profile is not None:
            _log_mcore_conversion_profile(
                conversion_profile,
                rank,
                step,
                conversion_wall_seconds,
                conversion_process_cpu_seconds,
                pytorch_state_dict,
            )
        filesystem = get_s3_fs()
        s3_refresh_if_expiring(filesystem)
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
    """Read only the DCP tensor byte ranges assigned to this rank from S3."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def load(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path, async_strategy: str = "mcore"):
        del async_strategy  # Required by the Megatron sharded-load strategy interface.
        original = sharded_state_dict
        tensors = [value for value in nested_values(original) if isinstance(value, ShardedTensor)]
        converted, flat_mapping, rename_mapping = _replace_state_dict_keys_with_sharded_keys(original)
        pytorch_state_dict = mcore_to_pyt_state_dict(converted, True)

        filesystem = get_s3_fs()
        s3_refresh_if_expiring(filesystem)
        reader = FileSystemReader(self.checkpoint_dir)
        reader.fs = FsspecFileSystem()
        reader.fs.fs = filesystem
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
