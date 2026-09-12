"""Own a frozen K10 install/replay interval with explicit group lifetime."""

import asyncio
from enum import StrEnum
import time


class GenerationBoundary(StrEnum):
    INTERVAL = "interval"
    DRIVER = "driver"


class ShardLifecycle(StrEnum):
    CLOSE = "close"
    RETAIN = "retain"


async def settled(*calls):
    """Join every launched native call before allowing cleanup or resuming."""
    batch = asyncio.gather(*calls, return_exceptions=True)
    cancellation = None
    while not batch.done():
        try:
            await asyncio.shield(batch)
        except asyncio.CancelledError as error:
            # Every cancellation request must leave the native batch protected.
            # A second cancel during the join must not release its weight lease.
            if cancellation is None:
                cancellation = error
    results = batch.result()
    if cancellation is not None:
        for value in results:
            if isinstance(value, BaseException):
                cancellation.add_note(f"Joined shard call: {type(value).__name__}: {value}")
        raise cancellation
    failures = [value for value in results if isinstance(value, BaseException)]
    if failures:
        for extra in failures[1:]:
            failures[0].add_note(f"Concurrent shard call: {type(extra).__name__}: {extra}")
        raise failures[0]
    return results


def receipt_rows(value):
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, (tuple, list)):
        raise ValueError("Shard calls must retain per-rank dictionary receipts")
    return [row for item in value for row in receipt_rows(item)]


def validate_rows(value, manifest_id, publication_id, expected_ranks, phase):
    rows = receipt_rows(value)
    if any(type(row["rank"]) is not int for row in rows) or sorted(row["rank"] for row in rows) != sorted(
        expected_ranks
    ):
        raise ValueError("Shard receipt is missing or duplicates a native participant")
    if any(
        row["manifest_id"] != manifest_id
        or type(row["publication_id"]) is not int
        or row["publication_id"] != publication_id
        or row["phase"] != phase
        for row in rows
    ):
        raise ValueError("Shard receipt has the wrong manifest, version or phase")
    return rows


async def run_shard_interval(
    driver,
    manifest_id,
    publication_id,
    *,
    replay,
    policy_ranks,
    receiver_ranks,
    expected_receiver_bytes,
    proofs=True,
    lifecycle=ShardLifecycle.CLOSE,
    generation_boundary=GenerationBoundary.INTERVAL,
    capture=None,
    observe=None,
):
    """Pause, freeze and install before releasing ownership, with optional full-byte proofs.

    With proofs enabled, the replay callback must execute the complete native byte proof.
    Disabling proofs retains version, storage and participant checks. Failed/partial
    installation leaves inference paused; the caller must terminate or recover
    the job rather than generate with a partial installation.
    """
    if capture is not None and not callable(capture):
        raise ValueError("Shard interval capture must be callable")
    if type(proofs) is not bool:
        raise ValueError("Shard interval proofs must be boolean")
    lifecycle = ShardLifecycle(lifecycle)
    generation_boundary = GenerationBoundary(generation_boundary)
    if not callable(replay) or not policy_ranks or not receiver_ranks:
        raise ValueError("Shard interval requires explicit complete proof and rank coverage")
    # Snapshot independently prepared installed storage counts before any RPC.
    # A callback's claimed coverage fraction cannot establish complete bytes.
    expected_receiver_bytes = dict(expected_receiver_bytes)
    if (
        any(type(rank) is not int for rank in expected_receiver_bytes)
        or set(expected_receiver_bytes) != set(receiver_ranks)
        or any(type(count) is not int or count <= 0 for count in expected_receiver_bytes.values())
    ):
        raise ValueError("Shard replay requires exact expected installed bytes for every receiver")
    client = driver.inference_engine_client

    async def policy(method, *args):
        refs = driver.policy_model.async_run_ray_method("pass_through", method, manifest_id, publication_id, *args)
        return await settled(*refs)

    timings = {}

    async def measured(name, call):
        started = time.perf_counter()
        try:
            return await call
        finally:
            timings[name] = time.perf_counter() - started

    result = {"phase_seconds": timings, "proofs": proofs}
    installation_complete = False
    primary = None
    try:
        if generation_boundary is GenerationBoundary.INTERVAL:
            await measured("pause", settled(client.pause_generation(settle_native_calls=True)))
        elif not client.generation_paused_event.is_set():
            raise ValueError("Driver-owned shard publication requires paused inference")
        result["policy_begin"] = validate_rows(
            await measured("freeze", policy("begin_shard_publication", proofs)),
            manifest_id,
            publication_id,
            policy_ranks,
            "frozen",
        )
        if proofs:
            result["source_proof"] = validate_rows(
                await measured("source_replica_proof", policy("verify_shard_publication")),
                manifest_id,
                publication_id,
                policy_ranks,
                "verified",
            )
        result["receiver_begin"] = validate_rows(
            (await measured("receiver_begin", settled(client.begin_shard_stream(manifest_id, publication_id, proofs))))[
                0
            ],
            manifest_id,
            publication_id,
            receiver_ranks,
            "frozen",
        )
        if observe is not None:
            result["physical_before"] = await measured("observation_before", observe("before"))
        installed = await measured(
            "install", settled(policy("run_shard_publication"), client.run_shard_stream(manifest_id, publication_id))
        )
        if observe is not None:
            result["physical_after"] = await measured("observation_after", observe("after"))
        result["policy_install"] = validate_rows(installed[0], manifest_id, publication_id, policy_ranks, "installed")
        result["receiver_install"] = validate_rows(
            installed[1], manifest_id, publication_id, receiver_ranks, "installed"
        )
        if capture is not None:
            capture(
                {
                    "phase": "installed-before-replay",
                    "manifest_id": manifest_id,
                    "publication_id": publication_id,
                    "result": result,
                }
            )
        if proofs:
            proof = (await measured("full_byte_replay", settled(replay(manifest_id, publication_id))))[0]
            proof_rows = validate_rows(proof, manifest_id, publication_id, receiver_ranks, "verified")
            if any(
                type(row["mismatches"]) is not int
                or row["mismatches"] != 0
                or row["coverage"] != 1.0
                or type(row["compared_bytes"]) is not int
                or row["compared_bytes"] != expected_receiver_bytes[row["rank"]]
                for row in proof_rows
            ):
                raise ValueError("Shard frozen replay did not prove complete installed bytes")
            result["replay"] = proof_rows
        if lifecycle is ShardLifecycle.RETAIN:
            finished = await measured(
                "finish",
                settled(policy("finish_shard_publication"), client.finish_shard_stream(manifest_id, publication_id)),
            )
            result["policy_finish"] = validate_rows(finished[0], manifest_id, publication_id, policy_ranks, "prepared")
            result["receiver_finish"] = validate_rows(
                finished[1], manifest_id, publication_id, receiver_ranks, "prepared"
            )
            if any(not row.get("groups_retained") for rows in finished for row in receipt_rows(rows)):
                raise ValueError("Persistent shard finish did not retain every participant's groups")
        installation_complete = True
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if lifecycle is ShardLifecycle.CLOSE or not installation_complete:
                cleanup = await settled(
                    policy("close_shard_publication"), client.close_shard_stream(manifest_id, publication_id)
                )
                result["policy_close"] = validate_rows(cleanup[0], manifest_id, publication_id, policy_ranks, "closed")
                result["receiver_close"] = validate_rows(
                    cleanup[1], manifest_id, publication_id, receiver_ranks, "closed"
                )
            if installation_complete and generation_boundary is GenerationBoundary.INTERVAL:
                await measured(
                    "resume", settled(client.resume_generation(policy_version=publication_id, settle_native_calls=True))
                )
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(f"Shard cleanup: {type(cleanup_error).__name__}: {cleanup_error}")
    return result
