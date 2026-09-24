from pathlib import Path
import os

from loguru import logger
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
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
from torch.distributed.checkpoint import FileSystemReader, LoadPlan, LoadPlanner
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.planner import LoadItemType

from marinskyrl.remote_io import create_s3_filesystem
from skyrl_train.checkpoint_listing import extract_step_from_path
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter
from skyrl_train.timing_observability import checkpoint_phase


# MCore 0.18 does not expose a storage-writer hook. Keep the adapter narrow: it
# preserves MCore's state translation and planner and replaces only the writer.
class DirectS3TorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Save MCore torch-dist shards directly to S3 through PyTorch DCP."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def save(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path) -> None:
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pytorch_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        filesystem = create_s3_filesystem()
        writer = StreamingFsspecWriter(self.checkpoint_dir, filesystem=filesystem)
        checkpoint.save(
            pytorch_state_dict,
            storage_writer=writer,
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
        )


class _ObservedFileSystemReader(FileSystemReader):
    """Count logical DCP read items without changing storage calls."""

    def __init__(self, checkpoint_dir: str, *, rank: int, step: int, operation: str) -> None:
        super().__init__(checkpoint_dir)
        self.observation_rank = rank
        self.observation_step = step
        self.observation_operation = operation

    def read_data(self, plan: LoadPlan, planner: LoadPlanner):
        with checkpoint_phase(
            "megatron",
            self.observation_operation,
            "dcp_read_data",
            rank=self.observation_rank,
            step=self.observation_step,
        ) as sample:
            try:
                storage_items = [self.storage_data[item.storage_index] for item in plan.items]
                sample.counters["logical_read_items"] = len(plan.items)
                sample.counters["logical_tensor_reads"] = sum(item.type == LoadItemType.TENSOR for item in plan.items)
                sample.counters["logical_byte_reads"] = sum(item.type == LoadItemType.BYTE_IO for item in plan.items)
                sample.counters["logical_read_bytes"] = sum(item.length for item in storage_items)
                sample.counters["storage_files_touched"] = len({item.relative_path for item in storage_items})
            except Exception:
                logger.opt(exception=True).warning("Could not count logical checkpoint reads")
            return super().read_data(plan, planner)


class DirectS3TorchDistLoadShardedStrategy(TorchDistLoadShardedStrategy):
    """Read only the DCP tensor byte ranges assigned to this rank from S3."""

    def __init__(self, checkpoint_dir: str, *, operation: str = "resume") -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.operation = operation

    def load(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path, async_strategy: str = "mcore"):
        del async_strategy  # Required by the Megatron sharded-load strategy interface.
        original = sharded_state_dict
        tensors = [value for value in nested_values(original) if isinstance(value, ShardedTensor)]
        converted, flat_mapping, rename_mapping = _replace_state_dict_keys_with_sharded_keys(original)
        pytorch_state_dict = mcore_to_pyt_state_dict(converted, True)

        filesystem = create_s3_filesystem()
        reader = _ObservedFileSystemReader(
            self.checkpoint_dir,
            rank=dist.get_rank(),
            step=extract_step_from_path(os.path.dirname(self.checkpoint_dir.rstrip("/"))),
            operation=self.operation,
        )
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
