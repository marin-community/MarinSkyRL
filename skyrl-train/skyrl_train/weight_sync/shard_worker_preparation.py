"""Native model and identity adapters for explicit shard preparation RPCs."""

from functools import partial

from skyrl_train.weight_sync.bucket_identity import bucket_identity
from skyrl_train.weight_sync.shard_observations import hardware_identity
from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.shard_preparation import (
    bind_live_preparation,
    close_live_preparation,
    collect_policy_preparation,
    collect_receiver_preparation,
)


def collect_policy(worker, preparation_id, geometry, output_uri):
    from megatron.core import parallel_state

    device = next(worker.actor_module[0].parameters()).device
    capture = partial(persist_readback, output_uri, f"shard-source-{preparation_id}")
    return collect_policy_preparation(
        worker,
        preparation_id,
        geometry,
        parallel_state,
        {**bucket_identity(device), **hardware_identity(device)},
        capture,
    )


def collect_receiver(worker, preparation_id, geometry, replica):
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    return collect_receiver_preparation(
        worker,
        preparation_id,
        geometry,
        replica,
        ep.rank_in_group,
        ep.world_size,
        {**bucket_identity(worker.device), **hardware_identity(worker.device)},
    )


def bind_policy(worker, plan, output_uri):
    from megatron.core import parallel_state

    return bind_live_preparation(
        worker,
        plan,
        parallel_state,
        partial(persist_readback, output_uri, f"shard-bind-{plan.preparation_id}"),
        proof_capture=partial(persist_readback, output_uri, f"shard-source-proof-{plan.preparation_id}"),
    )


def bind_receiver(worker, plan, output_uri):
    return bind_live_preparation(
        worker, plan, None, partial(persist_readback, output_uri, f"shard-bind-{plan.preparation_id}")
    )


def close_preparation(worker, preparation_id):
    return close_live_preparation(worker, preparation_id)
