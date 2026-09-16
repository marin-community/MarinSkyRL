from pathlib import Path

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from torch.distributed import checkpoint

from skyrl_train.io.s3fs import get_s3_fs, s3_refresh_if_expiring
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter


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
        filesystem = get_s3_fs()
        s3_refresh_if_expiring(filesystem)
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
