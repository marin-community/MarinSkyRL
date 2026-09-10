"""Bind observed model tensors and explicit rendezvous endpoints to K10 sessions.

The coordinator must gather exact typed inventories and assign global identities
before calling this preparation. No default model, inferred rank, address lookup
or automatic GPU allocation is provided.
"""

import torch.distributed as dist

from skyrl_train.weight_sync.byte_replay import ReceiverByteCoverage
from skyrl_train.weight_sync.frozen_source_views import local_source_slices
from skyrl_train.weight_sync.shard_group_factory import prepare_rank_groups
from skyrl_train.weight_sync.shard_group_ready import warm_owned_groups
from skyrl_train.weight_sync.shard_replica_proof import FullReplicaComparator, local_replica_catalogue
from skyrl_train.weight_sync.shard_session import bind_worker_shard_stream
from skyrl_train.weight_sync.shard_stream import ShardStreamRank


def native_policy_sources(worker, global_rank):
    """Acquire the policy lease while resolving actual Bridge source views."""
    with worker._policy_weight_access.hold("shard-source-preparation"):
        slices, sources = local_source_slices(worker.bridge.get_conversion_tasks(worker.actor_module), worker.provider)
    return slices, sources, local_replica_catalogue(global_rank, slices, sources)


def native_receiver_sources(worker):
    config = worker.vllm_config
    hf, parallel = config.model_config.hf_config, config.parallel_config
    if (
        hf.model_type != "grug_moe"
        or config.model_config.quantization is not None
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_eplb
    ):
        raise ValueError("Native shard receiver requires unquantized TP1 PP1 Grug without expert rebalancing")
    model = worker.model_runner.model
    maps = {}
    for name, module in model.named_modules():
        if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            continue
        backend = getattr(getattr(module.quant_method, "unquantized_backend", None), "name", None)
        if backend != "TRITON":
            raise ValueError("Native shard expert destinations require the observed TRITON backend")
        maps[name] = tuple(
            int(module._map_global_expert_id_to_local_expert_id(expert)) for expert in range(hf.num_experts)
        )
    if len(maps) != hf.num_hidden_layers:
        raise ValueError("Native shard expert module inventory misses configured layers")
    parameters = dict(model.named_parameters())
    coverage = ReceiverByteCoverage(parameters)
    return parameters, maps, coverage.expected_bytes


def required_group_memberships(schedule, dense_plan, replica_plan, *, create_replica_groups=True):
    memberships = {f"stream-expert-{item.ep}": item.members for item in schedule.groups}
    receiver_global = dict(schedule.receiver_global_ranks)
    if not dense_plan:
        raise ValueError("Native shard factory requires the complete dense schedule")
    fanout = dense_plan[0].replica_fanout
    if any(item.replica_fanout != fanout for item in dense_plan):
        raise ValueError("Dense entries disagree on receiver replica membership")
    for index, native_ranks in enumerate(fanout):
        memberships[f"stream-local-{index}"] = tuple(receiver_global[rank] for rank in native_ranks)
    for item in replica_plan.groups if create_replica_groups else ():
        if item.name in memberships:
            raise ValueError("Proof and installation group namespaces overlap")
        memberships[item.name] = item.members
    return memberships


def prepare_native_shard_worker(
    worker,
    rank,
    schedule,
    expert_views,
    dense_plan,
    replica_plan,
    endpoints,
    *,
    sources,
    parameters,
    expert_maps,
    transfer_workspace,
    comparison_workspace,
    dense_chunk_bytes,
    policy_access,
    borrowed_groups=None,
    borrowed_source_ranks=None,
):
    """Prepare actual groups, validate live storage, then expose the bound session.

    Allocation owners provide premeasured workspace tensors explicitly. All ranks
    consume identical source-bound endpoint order; only members construct each
    communicator. Native caller retains device/stream/peak receipts separately.
    """
    if getattr(worker, "_shard_stream_session", None) is not None:
        raise ValueError("Close previous native shard preparation before creating groups")
    required = required_group_memberships(
        schedule, dense_plan, replica_plan, create_replica_groups=borrowed_groups is None
    )
    if tuple(item.name for item in endpoints) != tuple(required) or any(
        item.members != required[item.name] for item in endpoints
    ):
        raise ValueError("Native endpoints differ from complete ordered source/group membership")
    trainer = rank in dict(schedule.trainer_global_ranks).values()
    if trainer and (policy_access is None or comparison_workspace is None):
        raise ValueError("Policy preparation requires its actual weight lease and comparison workspace")
    if not trainer and (policy_access is not None or comparison_workspace is not None):
        raise ValueError("Receiver preparation cannot claim policy proof workspace or lease")
    groups = prepare_rank_groups(rank, endpoints)
    try:
        local = [
            groups[name] for name, members in required.items() if name.startswith("stream-local-") and rank in members
        ]
        runner = ShardStreamRank(
            rank,
            schedule,
            expert_views,
            dense_plan,
            sources,
            parameters,
            expert_maps,
            transfer_workspace,
            {item.ep: groups[f"stream-expert-{item.ep}"] for item in schedule.groups if rank in item.members},
            local[0] if local else None,
            dense_chunk_bytes=dense_chunk_bytes,
        )
        group_ready = warm_owned_groups(runner, groups, required)
        runner.replica_plan_id = replica_plan.identity
        comparator = (
            FullReplicaComparator(
                rank,
                replica_plan,
                groups if borrowed_groups is None else borrowed_groups,
                transfer_workspace,
                comparison_workspace,
                broadcast_source_ranks=borrowed_source_ranks,
            )
            if trainer
            else None
        )
        expected_receiver_bytes = None if trainer else ReceiverByteCoverage(parameters).expected_bytes
        manifest_id = bind_worker_shard_stream(
            worker,
            runner,
            policy_access=policy_access,
            replica_verifier=comparator,
            owned_groups=tuple(groups.values()),
            retained_proof_workspace_bytes=comparison_workspace.numel() if trainer else 0,
        )
        return {
            "rank": rank,
            "manifest_id": manifest_id,
            "replica_plan_id": replica_plan.identity,
            "group_memberships": {name: required[name] for name in groups},
            "group_readiness": group_ready,
            "dense_chunk_bytes": dense_chunk_bytes,
            "expected_receiver_bytes": expected_receiver_bytes,
            "transfer_workspace_bytes": transfer_workspace.numel(),
            "comparison_workspace_bytes": comparison_workspace.numel() if trainer else 0,
        }
    except BaseException as primary:
        for group in reversed(tuple(groups.values())):
            try:
                dist.destroy_process_group(group)
            except Exception as error:
                primary.add_note(f"Native shard preparation cleanup: {type(error).__name__}: {error}")
        raise
