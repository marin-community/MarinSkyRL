"""Load Megatron factories without duplicating their existing destinations."""

from dataclasses import replace
from functools import partial

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensorFactory
from megatron.core.transformer.utils import cat_with_oom_fallback

from skyrl_train.io.checkpoint_buffers import merge_adjacent_checkpoint_shards


def reuse_checkpoint_factory_buffers(sharded_state_dict: ShardedStateDict) -> None:
    """Reuse SwiGLU destinations during load without changing the saved layout."""

    def reuse(value):
        if isinstance(value, ShardedTensorFactory) and value.merge_fn is cat_with_oom_fallback:
            return replace(value, merge_fn=partial(merge_adjacent_checkpoint_shards, fallback=value.merge_fn))
        return value

    dict_list_map_inplace(reuse, sharded_state_dict)
