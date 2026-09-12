"""Durable per-worker replay receipts and a complete awaited driver callback."""

from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.shard_interval import receipt_rows, settled, validate_rows
from skyrl_train.weight_sync.shard_session import storage_versions, worker_shard_call


def replay_worker_call(worker, method, manifest_id, publication_id, output_uri):
    state = getattr(worker, "_shard_stream_session", None)
    try:
        if state is not None and state.runner.rank in state.runner.receivers:
            current = dict(worker.model_runner.model.named_parameters())
            if storage_versions(current) != storage_versions(state.runner.parameters):
                raise ValueError("Actual model parameter inventory differs from the prepared receiver")
        result = worker_shard_call(worker, method, manifest_id, publication_id)
    except BaseException as primary:
        failed = {
            "manifest_id": manifest_id,
            "publication_id": publication_id,
            "phase": "failed",
            "rank": state.runner.rank if state is not None else None,
            "method": method,
            "error_type": type(primary).__name__,
            "error": str(primary)[:4096],
        }
        if state is not None and state.replay_state is not None:
            failed["memory_before"] = state.replay_state.memory_before
            try:
                failed["memory_after"] = state.replay_state.memory_snapshot()
            except BaseException as error:
                failed["memory_readback_error"] = f"{type(error).__name__}: {error}"
        try:
            persist_readback(output_uri, f"shard-{method}-failed-{manifest_id}-{publication_id}", failed)
        except BaseException as error:
            primary.add_note(f"Shard replay failure receipt: {type(error).__name__}: {error}")
        raise
    binding = persist_readback(output_uri, f"shard-{method}-{manifest_id}-{publication_id}", result)
    return {**result, "durable_receipt": binding}


async def replay_prepared_shards(
    driver,
    manifest_id,
    publication_id,
    *,
    policy_ranks,
    expected_receiver_bytes,
    expected_device_type,
    output_uri,
    capture,
):
    """Precheck all receivers, replay all ranks, persist raw rows, then apply the gate.

    Production packets must explicitly require CUDA measurements. CPU is solely
    for the real-Gloo fixture and cannot satisfy the native memory gate.
    """
    if expected_device_type not in ("cpu", "cuda") or not callable(capture):
        raise ValueError("Replay needs an explicit device gate and durable coordinator sink")
    expected_receiver_bytes = dict(expected_receiver_bytes)
    policy_ranks = tuple(policy_ranks)
    receiver_ranks = tuple(expected_receiver_bytes)
    if (
        not policy_ranks
        or not receiver_ranks
        or any(type(rank) is not int or rank < 0 for rank in (*policy_ranks, *receiver_ranks))
        or len(set(policy_ranks)) != len(policy_ranks)
        or set(policy_ranks) & set(receiver_ranks)
        or any(type(count) is not int or count <= 0 for count in expected_receiver_bytes.values())
    ):
        raise ValueError("Replay requires exact disjoint participants and complete installed byte counts")
    client = driver.inference_engine_client

    async def policy(method):
        refs = driver.policy_model.async_run_ray_method("pass_through", method, manifest_id, publication_id, output_uri)
        return await settled(*refs)

    prepared = await settled(
        policy("prepare_shard_publication_replay"), client.prepare_shard_replay(manifest_id, publication_id, output_uri)
    )
    capture({"phase": "replay-prepared", "rows": receipt_rows(prepared)})
    policy_ready = validate_rows(prepared[0], manifest_id, publication_id, policy_ranks, "replay-ready")
    receiver_ready = validate_rows(prepared[1], manifest_id, publication_id, receiver_ranks, "replay-ready")
    if any(row["expected_bytes"] != expected_receiver_bytes[row["rank"]] for row in receiver_ready):
        raise ValueError("Independent complete receiver inventory differs from the native plan")
    if any(
        row["memory_before"]["cuda_measured"] != (expected_device_type == "cuda")
        for row in policy_ready + receiver_ready
    ):
        raise ValueError("Actual replay device differs from the required native device gate")
    results = await settled(
        policy("replay_shard_publication"), client.replay_shard_stream(manifest_id, publication_id, output_uri)
    )
    capture({"phase": "replay-returned", "rows": receipt_rows(results)})
    policies = validate_rows(results[0], manifest_id, publication_id, policy_ranks, "verified")
    receivers = validate_rows(results[1], manifest_id, publication_id, receiver_ranks, "verified")
    if any(
        row["mismatches"] != 0
        or row["compared_bytes"] != expected_receiver_bytes[row["rank"]]
        or row["coverage"] != 1.0
        for row in receivers
    ):
        raise ValueError("Full installed-byte replay failed")
    if expected_device_type == "cuda" and any(
        row["replay_memory_within_limit"] is not True for row in policies + receivers
    ):
        raise ValueError("Native replay exceeded the complete additional scratch limit")
    return receivers
