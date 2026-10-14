"""Load Megatron factories without duplicating their existing destinations."""

from dataclasses import replace
from functools import partial

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensorFactory
from megatron.core.transformer.utils import cat_with_oom_fallback

from skyrl_train.io.checkpoint_buffers import merge_adjacent_checkpoint_shards


def reuse_checkpoint_factory_buffers(sharded_state_dict: ShardedStateDict) -> None:
    """Reuse SwiGLU destinations during load without changing the saved layout."""

    # Core 0.19.2's SwiGLU merge allocates another full tensor through torch.cat:
    # https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/transformer/utils.py#L80
    # DCP already loaded adjacent gate/up views into their destination. Reuse it;
    # other layouts retain native merging. Remove when native restore reuses the
    # destination; verify the BF16/FP32 DCP value/alias tests and full save/resume.
    def reuse(value):
        if isinstance(value, ShardedTensorFactory) and value.merge_fn is cat_with_oom_fallback:
            return replace(value, merge_fn=partial(merge_adjacent_checkpoint_shards, fallback=value.merge_fn))
        return value

    dict_list_map_inplace(reuse, sharded_state_dict)
