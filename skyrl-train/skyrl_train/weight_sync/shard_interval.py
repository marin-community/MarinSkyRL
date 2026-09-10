"""Explicit diagnostic entrypoint for one frozen K10 install/replay interval."""

import asyncio


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
    driver, manifest_id, publication_id, *, replay, policy_ranks, receiver_ranks, expected_receiver_bytes
):
    """Pause, freeze, prove source copies, install, replay, close, then resume.

    The explicit replay callback must execute the complete native byte proof.
    No callback or replica verifier is installed by default. Failed/partial
    installation leaves inference paused; the caller must terminate or recover
    the diagnostic job, never continue generation with unverified weights.
    """
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

    async def policy(method):
        refs = driver.policy_model.async_run_ray_method("pass_through", method, manifest_id, publication_id)
        return await settled(*refs)

    result = {}
    installed_and_verified = False
    primary = None
    try:
        await settled(client.pause_generation(settle_native_calls=True))
        result["policy_begin"] = validate_rows(
            await policy("begin_shard_publication"), manifest_id, publication_id, policy_ranks, "frozen"
        )
        result["source_proof"] = validate_rows(
            await policy("verify_shard_publication"), manifest_id, publication_id, policy_ranks, "verified"
        )
        result["receiver_begin"] = validate_rows(
            (await settled(client.begin_shard_stream(manifest_id, publication_id)))[0],
            manifest_id,
            publication_id,
            receiver_ranks,
            "frozen",
        )
        installed = await settled(policy("run_shard_publication"), client.run_shard_stream(manifest_id, publication_id))
        result["policy_install"] = validate_rows(installed[0], manifest_id, publication_id, policy_ranks, "installed")
        result["receiver_install"] = validate_rows(
            installed[1], manifest_id, publication_id, receiver_ranks, "installed"
        )
        proof = (await settled(replay(manifest_id, publication_id)))[0]
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
        installed_and_verified = True
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            cleanup = await settled(
                policy("close_shard_publication"), client.close_shard_stream(manifest_id, publication_id)
            )
            result["policy_close"] = validate_rows(cleanup[0], manifest_id, publication_id, policy_ranks, "closed")
            result["receiver_close"] = validate_rows(cleanup[1], manifest_id, publication_id, receiver_ranks, "closed")
            if installed_and_verified:
                await settled(client.resume_generation(policy_version=publication_id, settle_native_calls=True))
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(f"Shard cleanup: {type(cleanup_error).__name__}: {cleanup_error}")
    return result
