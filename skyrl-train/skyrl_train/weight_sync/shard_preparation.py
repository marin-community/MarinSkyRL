"""Gather live model metadata and bind one explicitly owned shard preparation.

Only tensor metadata crosses the coordinator boundary. Current source views and
installed parameters stay on their native worker; preparation owns the learner
lease until binding or cleanup completes. Native allocation peaks are separate
from the later source-replica comparison's proof-memory gate.
"""

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
import time

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.frozen_source_views import local_source_slices
from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint
from skyrl_train.weight_sync.shard_group_schedule import (
    TrainerRank,
    UnequalExpertParallelism,
    ReceiverRank,
    ShardGroupSchedule,
    build_shard_group_schedule,
)
from skyrl_train.weight_sync.shard_native_factory import (
    native_receiver_sources,
    prepare_native_shard_worker,
    required_group_memberships,
)
from skyrl_train.weight_sync.shard_replica_proof import (
    ReplicaPlan,
    local_replica_catalogue,
    build_replica_plan,
    borrowed_policy_groups,
)
from skyrl_train.weight_sync.shard_session import storage_versions
from skyrl_train.weight_sync.shard_source_inventory import (
    DenseSourceRank,
    DenseTransfer,
    LocalExpertSource,
    local_shard_inventory,
    dense_stream_plan,
)


@dataclass(frozen=True)
class ShardGeometry:
    policy_ranks: int
    receiver_replicas: int
    expert_parallel_size: int
    layers_by_pp: tuple[tuple[int, ...], ...]
    num_experts: int
    hidden_size: int
    intermediate_size: int

    def validate(self):
        values = (
            self.policy_ranks,
            self.receiver_replicas,
            self.expert_parallel_size,
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        if any(type(value) is not int or value <= 0 for value in values) or not self.layers_by_pp:
            raise ValueError("Shard geometry requires complete positive model and rank dimensions")
        layers = tuple(layer for stage in self.layers_by_pp for layer in stage)
        if not all(self.layers_by_pp) or layers != tuple(range(len(layers))):
            raise ValueError("Shard PP ownership must cover the ordered complete model layers")
        if self.policy_ranks % (len(self.layers_by_pp) * self.expert_parallel_size):
            raise ValueError("Shard policy ranks do not form complete PP/EP replicas")
        if self.num_experts % self.expert_parallel_size:
            raise ValueError("Shard experts must split evenly across EP owners")


@dataclass(frozen=True)
class PreparationOptions:
    transfer_bytes: int
    comparison_bytes: int
    dense_chunk_bytes: int
    minimum_free_bytes: int

    def validate(self):
        if (
            any(type(value) is not int for value in asdict(self).values())
            or self.transfer_bytes <= 0
            or not 1 <= self.comparison_bytes <= 65536
            or not 4 <= self.dense_chunk_bytes <= self.transfer_bytes
            or self.minimum_free_bytes < 0
        ):
            raise ValueError("Shard preparation workspace/headroom limits are invalid")


class PreparationPhase(StrEnum):
    COLLECTED = "collected"
    BINDING = "binding"
    BOUND = "bound"
    FAILED = "failed"


@dataclass
class LocalPreparation:
    preparation_id: str
    geometry: ShardGeometry
    rank: int
    sources: dict
    parameters: dict
    expert_maps: dict
    source_versions: dict
    token: str | None
    metadata: dict
    phase: PreparationPhase = PreparationPhase.COLLECTED


def _fresh(worker, preparation_id, geometry):
    geometry.validate()
    if not isinstance(preparation_id, str) or not preparation_id:
        raise ValueError("Shard preparation needs an explicit identity")
    if (
        getattr(worker, "_shard_preparation", None) is not None
        or getattr(worker, "_shard_stream_session", None) is not None
    ):
        raise ValueError("Previous shard preparation/session must be closed first")


def collect_policy_preparation(worker, preparation_id, geometry, parallel_state, identity, capture):
    _fresh(worker, preparation_id, geometry)
    rank = dist.get_rank()
    trainer = TrainerRank(
        rank,
        parallel_state.get_expert_data_parallel_rank(),
        parallel_state.get_pipeline_model_parallel_rank(),
        parallel_state.get_expert_model_parallel_rank(),
    )
    actual_ep = parallel_state.get_expert_model_parallel_world_size()
    if actual_ep != geometry.expert_parallel_size:
        raise UnequalExpertParallelism(
            f"K10_EQUAL_EP_REQUIRED policy_ep={actual_ep} prepared_ep={geometry.expert_parallel_size}"
        )
    if dist.get_world_size() != geometry.policy_ranks or parallel_state.get_tensor_model_parallel_rank() != 0:
        raise ValueError("Observed policy world differs from TP1 shard geometry")
    if parallel_state.get_tensor_model_parallel_world_size() != 1:
        raise ValueError("Observed policy TP degree differs from shard geometry")
    if not callable(capture):
        raise ValueError("Live source preparation requires a durable allocation receipt sink")
    token = worker._policy_weight_access.acquire("shard-native-preparation")
    device, before = None, None
    started = time.monotonic()
    try:
        device = next(worker.actor_module[0].parameters()).device
        before = _allocation_state(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        slices, sources = local_source_slices(worker.bridge.get_conversion_tasks(worker.actor_module), worker.provider)
        inventory = local_shard_inventory(
            slices,
            sources,
            trainer,
            layers=geometry.layers_by_pp[trainer.pp],
            num_experts=geometry.num_experts,
            expert_parallel_size=geometry.expert_parallel_size,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
        )
        groups = {
            "expert": tuple(dist.get_process_group_ranks(parallel_state.get_expert_data_parallel_group())),
            "dense": tuple(dist.get_process_group_ranks(parallel_state.get_data_parallel_group())),
        }
        metadata = {
            "preparation_id": preparation_id,
            "role": "policy",
            "rank": rank,
            "identity": identity,
            "trainer": trainer,
            "inventory": inventory,
            "catalogue": local_replica_catalogue(rank, slices, sources),
            "native_groups": groups,
            "geometry": geometry,
        }
        metadata["source_setup_seconds"] = time.monotonic() - started
        metadata["source_setup_allocation"] = {"before": before, "after": _allocation_state(device)}
        capture(
            {
                "preparation_id": preparation_id,
                "rank": rank,
                "identity": identity,
                "phase": "source-metadata",
                "source_setup_allocation": metadata["source_setup_allocation"],
                "source_setup_seconds": metadata["source_setup_seconds"],
            }
        )
        worker._shard_preparation = LocalPreparation(
            preparation_id, geometry, rank, sources, {}, {}, storage_versions(sources), token, metadata
        )
        return metadata
    except BaseException as primary:
        failure = {
            "preparation_id": preparation_id,
            "rank": rank,
            "identity": identity,
            "phase": "source-metadata-failed",
            "source_setup_seconds": time.monotonic() - started,
            "error_type": type(primary).__name__,
            "error": str(primary)[:4096],
            "allocation_before": before,
        }
        try:
            if device is not None:
                try:
                    failure["allocation_after"] = _allocation_state(device)
                except BaseException as error:
                    failure["allocation_readback_error"] = f"{type(error).__name__}: {error}"
                    primary.add_note(failure["allocation_readback_error"])
            try:
                capture(failure)
            except BaseException as error:
                primary.add_note(f"Source preparation receipt: {type(error).__name__}: {error}")
        finally:
            worker._policy_weight_access.release(token)
        raise


def collect_receiver_preparation(worker, preparation_id, geometry, replica, ep_rank, ep_size, identity):
    _fresh(worker, preparation_id, geometry)
    native_rank, world = dist.get_rank(), dist.get_world_size()
    if ep_size != geometry.expert_parallel_size:
        raise UnequalExpertParallelism(
            f"K10_EQUAL_EP_REQUIRED receiver_ep={ep_size} prepared_ep={geometry.expert_parallel_size}"
        )
    if (
        type(replica) is not int
        or not 0 <= replica < geometry.receiver_replicas
        or world != geometry.expert_parallel_size
        or native_rank != ep_rank
    ):
        raise ValueError("Observed receiver world/EP mapping differs from TP1 shard geometry")
    hf = worker.vllm_config.model_config.hf_config
    expected = (
        sum(map(len, geometry.layers_by_pp)),
        geometry.num_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
    )
    if (hf.num_hidden_layers, hf.num_experts, hf.hidden_size, hf.moe_intermediate_size) != expected:
        raise ValueError("Actual receiver model dimensions differ from source geometry")
    parameters, maps, expected_bytes = native_receiver_sources(worker)
    rank = replica * world + native_rank
    receiver = ReceiverRank(rank, replica, ep_rank)
    dense = {
        name: (tuple(value.shape), str(value.dtype).removeprefix("torch."))
        for name, value in parameters.items()
        if not name.endswith((".routed_experts.w13_weight", ".routed_experts.w2_weight"))
    }
    metadata = {
        "preparation_id": preparation_id,
        "role": "receiver",
        "rank": rank,
        "identity": identity,
        "receiver": receiver,
        "dense_parameters": dense,
        "expected_bytes": expected_bytes,
        "geometry": geometry,
    }
    worker._shard_preparation = LocalPreparation(
        preparation_id, geometry, geometry.policy_ranks + rank, {}, parameters, maps, {}, None, metadata
    )
    return metadata


@dataclass(frozen=True)
class PreparedShardPlan:
    preparation_id: str
    geometry: ShardGeometry
    options: PreparationOptions
    schedule: ShardGroupSchedule
    expert_views: tuple[LocalExpertSource, ...]
    dense_plan: tuple[DenseTransfer, ...]
    replica_plan: ReplicaPlan
    endpoints: tuple[GroupEndpoint, ...]
    expected_receiver_bytes: tuple[tuple[int, int], ...]
    identity_rows: tuple[dict, ...]

    @property
    def plan_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_shard_preparation(policy, receivers, geometry, options, endpoint_factory):
    geometry.validate()
    options.validate()
    if len(policy) != geometry.policy_ranks or sorted(row["rank"] for row in policy) != list(
        range(geometry.policy_ranks)
    ):
        raise ValueError("Policy preparation omits or duplicates native ranks")
    expected_receivers = geometry.receiver_replicas * geometry.expert_parallel_size
    if len(receivers) != expected_receivers or sorted(row["rank"] for row in receivers) != list(
        range(expected_receivers)
    ):
        raise ValueError("Receiver preparation omits or duplicates native ranks")
    rows = tuple(sorted(policy, key=lambda row: row["rank"])) + tuple(sorted(receivers, key=lambda row: row["rank"]))
    preparation_ids = {row["preparation_id"] for row in rows}
    if len(preparation_ids) != 1 or any(row["geometry"] != geometry for row in rows):
        raise ValueError("Native preparation identity or model geometry differs")
    for role, role_rows, key in (("policy", policy, "trainer"), ("receiver", receivers, "receiver")):
        if any(
            row["role"] != role or type(row["rank"]) is not int or row[key].rank != row["rank"] for row in role_rows
        ):
            raise ValueError("Native preparation role and observed rank mapping disagree")
    trainers = tuple(row["trainer"] for row in policy)
    receiver_ranks = tuple(row["receiver"] for row in receivers)
    if any(type(row["expected_bytes"]) is not int or row["expected_bytes"] <= 0 for row in receivers):
        raise ValueError("Native receiver storage needs a positive exact byte inventory")
    views = {}
    for row in policy:
        for view in row["inventory"].experts:
            if view.entry.name in views and views[view.entry.name] != view:
                raise ValueError("Policy replicas disagree on exact expert source metadata")
            views[view.entry.name] = view
    ordered_views = tuple(views[name] for name in sorted(views))
    schedule = build_shard_group_schedule(
        trainers,
        receiver_ranks,
        tuple(view.entry for view in ordered_views),
        trainer_ep=geometry.expert_parallel_size,
        receiver_ep=geometry.expert_parallel_size,
        layers_by_pp=geometry.layers_by_pp,
        num_experts=geometry.num_experts,
    )
    source_dtypes = {}
    for row in policy:
        for item in row["inventory"].dense:
            if item.hf_name in source_dtypes and source_dtypes[item.hf_name] != item.wire_dtype:
                raise ValueError("Policy copies disagree on dense wire dtype")
            source_dtypes[item.hf_name] = item.wire_dtype
    installed = receivers[0]["dense_parameters"]
    if any(row["dense_parameters"] != installed for row in receivers) or set(installed) != set(source_dtypes):
        raise ValueError("Dense native receiver/source names or storage geometry differ")
    for name, (_, dtype) in installed.items():
        if dtype != source_dtypes[name] and not (
            name.endswith(".mlp.router.weight") and dtype == "float32" and source_dtypes[name] == "bfloat16"
        ):
            raise ValueError("Dense native wire/storage conversion is unqualified")
    dense = dense_stream_plan(
        tuple(DenseSourceRank(row["trainer"], row["inventory"].dense) for row in policy),
        receiver_ranks,
        {name: (shape, source_dtypes[name]) for name, (shape, _) in installed.items()},
        expert_parallel_size=geometry.expert_parallel_size,
    )
    replica = build_replica_plan(trainers, schedule, tuple(row["catalogue"] for row in policy))
    native_for_global = {global_rank: native for native, global_rank in schedule.trainer_global_ranks}
    for row in policy:
        global_rank = dict(schedule.trainer_global_ranks)[row["rank"]]
        for group in replica.groups:
            if global_rank in group.members:
                expected = tuple(native_for_global[member] for member in group.members)
                if tuple(row["native_groups"]["expert" if group.primary else "dense"]) != expected:
                    raise ValueError("Observed borrowed groups differ from exact source-copy membership")
    memberships = required_group_memberships(schedule, dense, replica, create_replica_groups=False)
    endpoints = tuple(endpoint_factory(name, members) for name, members in memberships.items())
    return PreparedShardPlan(
        next(iter(preparation_ids)),
        geometry,
        options,
        schedule,
        ordered_views,
        dense,
        replica,
        endpoints,
        tuple(
            (geometry.policy_ranks + row["rank"], row["expected_bytes"])
            for row in sorted(receivers, key=lambda row: row["rank"])
        ),
        tuple({"role": row["role"], "rank": row["rank"], "identity": row["identity"]} for row in rows),
    )


def _allocation_state(device):
    if device.type != "cuda":
        return {"device": str(device), "cuda_measured": False}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "cuda_measured": True,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_bytes": free,
        "total_bytes": total,
    }


def bind_live_preparation(worker, plan, parallel_state, capture, *, proof_capture=None):
    state = getattr(worker, "_shard_preparation", None)
    if state is None or state.preparation_id != plan.preparation_id or state.geometry != plan.geometry:
        raise ValueError("Native binding does not match retained preparation")
    if (
        state.phase is not PreparationPhase.COLLECTED
        or getattr(worker, "_shard_stream_session", None) is not None
        or not callable(capture)
    ):
        raise ValueError("Native binding must be fresh and have a durable receipt sink")
    plan.options.validate()
    if storage_versions(state.sources) != state.source_versions:
        raise ValueError("Policy sources changed while the preparation lease was held")
    identity_row = {key: state.metadata[key] for key in ("role", "rank", "identity")}
    if identity_row not in plan.identity_rows:
        raise ValueError("Native worker identity was not included in the prepared plan")
    device = next(iter(state.sources.values() if state.sources else state.parameters.values())).device
    before = _allocation_state(device)
    required = plan.options.transfer_bytes + (plan.options.comparison_bytes if state.token is not None else 0)
    if device.type == "cuda" and before["free_bytes"] < required + plan.options.minimum_free_bytes:
        raise ValueError("Actual native free memory does not satisfy prepared workspace headroom")
    state.phase = PreparationPhase.BINDING
    started = time.monotonic()
    receipt = {
        "preparation_id": plan.preparation_id,
        "plan_id": plan.plan_id,
        "rank": state.rank,
        "identity": state.metadata["identity"],
        "allocation_before": before,
        "workspace_requested_bytes": required,
        "phase": "binding",
        "source_lease_held": state.token is not None,
    }
    primary = None
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        transfer = torch.empty(plan.options.transfer_bytes, dtype=torch.uint8, device=device)
        comparison = (
            torch.empty(plan.options.comparison_bytes, dtype=torch.bool, device=device)
            if state.token is not None
            else None
        )
        if state.token is not None:
            groups, roots, group_receipts = borrowed_policy_groups(
                parallel_state, state.rank, plan.schedule, plan.replica_plan
            )
        else:
            groups, roots, group_receipts = {}, {}, ()
        receipt["borrowed_replica_groups"] = group_receipts
        bound = prepare_native_shard_worker(
            worker,
            state.rank,
            plan.schedule,
            plan.expert_views,
            plan.dense_plan,
            plan.replica_plan,
            plan.endpoints,
            sources=state.sources,
            parameters=state.parameters,
            expert_maps=state.expert_maps,
            transfer_workspace=transfer,
            comparison_workspace=comparison,
            dense_chunk_bytes=plan.options.dense_chunk_bytes,
            policy_access=worker._policy_weight_access if state.token is not None else None,
            borrowed_groups=groups,
            borrowed_source_ranks=roots,
            proof_capture=proof_capture,
        )
        receipt.update(bound, phase="prepared")
        if storage_versions(state.sources) != state.source_versions:
            raise ValueError("Policy sources changed during native binding")
        if state.token is not None:
            worker._shard_stream_session.adopt_preparation_lease(state.token, state.source_versions)
            state.token = None
            receipt["source_lease_transferred"] = True
    except BaseException as error:
        primary = error
        receipt.update(phase="failed", error_type=type(error).__name__, error=str(error)[:4096])
        raise
    finally:
        try:
            try:
                receipt["allocation_after"] = _allocation_state(device)
            except BaseException as error:
                receipt["allocation_readback_error"] = f"{type(error).__name__}: {error}"
                if primary is None:
                    primary = error
                else:
                    primary.add_note(f"Preparation allocation readback: {type(error).__name__}: {error}")
            receipt["binding_seconds"] = time.monotonic() - started
            try:
                capture(receipt)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note(f"Preparation receipt capture: {type(error).__name__}: {error}")
        finally:
            state.phase = PreparationPhase.BOUND if primary is None else PreparationPhase.FAILED
            if state.token is not None:
                worker._policy_weight_access.release(state.token)
                state.token = None
        if primary is not None:
            raise primary
    return receipt


def close_live_preparation(worker, preparation_id):
    """Dispose only unstarted bindings, after all sibling preparation calls settle."""
    state = getattr(worker, "_shard_preparation", None)
    if state is None:
        return {"preparation_id": preparation_id, "phase": "closed", "state_was_present": False}
    if state.preparation_id != preparation_id or state.phase is PreparationPhase.BINDING:
        raise ValueError("Cannot close a different or active native preparation")
    session = getattr(worker, "_shard_stream_session", None)
    if session is not None and session.publication_id is not None:
        raise ValueError("An active publication must use its versioned session close")
    errors = []
    if session is not None:
        if session.versions is not None and storage_versions(session.runner.sources) != session.versions:
            errors.append("Frozen learner source changed before preparation close")
        for group in reversed(session.owned_groups):
            try:
                dist.destroy_process_group(group)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        if session.token is not None:
            session.policy_access.release(session.token)
            session.token = None
        del worker._shard_stream_session
    if state.token is not None:
        worker._policy_weight_access.release(state.token)
        state.token = None
    receipt = {
        "preparation_id": preparation_id,
        "rank": state.rank,
        "phase": "closed",
        "state_was_present": True,
        "cleanup_errors": errors,
    }
    del worker._shard_preparation
    if errors:
        raise RuntimeError(f"Native preparation cleanup failed: {errors}")
    return receipt
