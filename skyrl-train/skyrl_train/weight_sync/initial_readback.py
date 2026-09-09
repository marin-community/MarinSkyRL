"""Zero-update initial-sync diagnostic lifecycle, independent of the training loop."""

import asyncio
import json
import time

from skyrl_train.weight_sync.readback_diagnostics import receipt_chunks, validate_replica_digests


ALIGNED_ENVIRONMENT_KEYS = (
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_NTHREADS",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
)


def validate_communicator_environment(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Missing communicator environment")
    reference = rows[0]["environment"]["values"]
    for row in rows:
        values = row["environment"]["values"]
        if values["VLLM_BATCH_INVARIANT"] not in (None, "0"):
            raise ValueError("Weight-sync diagnostics require batch invariance off")
        if any(values[key] != reference[key] for key in ALIGNED_ENVIRONMENT_KEYS):
            raise ValueError("Weight-sync communicator environment mismatch")


def validate_initial_readback(policy: list[dict], receivers: list[list[dict]]) -> list[dict]:
    ranks = [row["rank"] for row in policy]
    if sorted(ranks) != list(range(len(policy))) or not policy:
        raise ValueError("Incomplete policy rank coverage")
    roots = [row for row in policy if row["all_rank_digests"] is not None]
    if len(roots) != 1 or roots[0]["rank"] != 0:
        raise ValueError("Missing policy digest allgather")
    gathered = roots[0]["all_rank_digests"]
    if len(gathered) != len(policy):
        raise ValueError("Incomplete digest allgather")
    for row, actual in zip(sorted(policy, key=lambda item: item["rank"]), gathered, strict=True):
        if any(row[key] != actual[key] for key in actual):
            raise ValueError("Allgather differs from originating policy receipt")
    comparisons = validate_replica_digests(gathered)
    if not receivers or any(not engine for engine in receivers):
        raise ValueError("Missing receiver engine")
    for engine in receivers:
        if sorted(row["rank"] for row in engine) != list(range(engine[0]["world_size"])):
            raise ValueError("Incomplete receiver rank coverage")
    all_rows = [*policy, *(row for engine in receivers for row in engine)]
    validate_communicator_environment(all_rows)
    return comparisons


async def run_initial_readback(trainer) -> dict:
    """Initialize and sync once; never start evaluation, producers, or PPO updates."""
    trainer.global_step = 0
    trainer._weight_sync_owner = asyncio.current_task()
    policy_environment = await asyncio.gather(
        *trainer.policy_model.async_run_ray_method("pass_through", "read_weight_sync_environment")
    )
    receiver_environment = await trainer.inference_engine_client.read_weight_sync_environment()
    pre_sync_environment = {"policy": policy_environment, "receivers": receiver_environment}
    for chunk in receipt_chunks(pre_sync_environment):
        print("WEIGHT_SYNC_PRE_GROUP_CHUNK " + json.dumps(chunk, sort_keys=True), flush=True)
    validate_communicator_environment(
        [*policy_environment, *(row for engine in receiver_environment for row in engine)]
    )
    started = time.monotonic()
    trainer.init_weight_sync_state()
    initialized = time.monotonic()
    await trainer.async_sync_policy_weights_to_inference_engines()
    await trainer._drain_policy_event_loops()
    synced = time.monotonic()
    policy = await asyncio.gather(
        *trainer.policy_model.async_run_ray_method("pass_through", "read_weight_sync_policy_state")
    )
    receivers = await trainer.inference_engine_client.read_publication_receiver_state()
    observed = time.monotonic()
    comparisons = validate_initial_readback(policy, receivers)
    return {
        "schema": "initial_weight_sync_readback_v1",
        "updates": 0,
        "initial_syncs": 1,
        "init_group_seconds": initialized - started,
        "initial_sync_seconds": synced - initialized,
        "post_sync_readback_seconds": observed - synced,
        "policy": policy,
        "receivers": receivers,
        "replica_digest_comparisons": comparisons,
        "pre_sync_environment": pre_sync_environment,
        "digest_scope": "diagnostic SHA-256; not K14 every-byte equality proof",
    }
