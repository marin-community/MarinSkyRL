"""Prepare live shard participants and own their cleanup across one diagnostic."""

from contextlib import asynccontextmanager
from dataclasses import asdict

from skyrl_train.weight_sync.shard_interval import receipt_rows, settled
from skyrl_train.weight_sync.shard_preparation import plan_shard_preparation


def validate_bindings(value, plan):
    rows = receipt_rows(value)
    expected = set(
        range(plan.geometry.policy_ranks + plan.geometry.receiver_replicas * plan.geometry.expert_parallel_size)
    )
    if (
        len(rows) != len(expected)
        or any(type(row["rank"]) is not int for row in rows)
        or {row["rank"] for row in rows} != expected
    ):
        raise ValueError("Prepared shard bindings miss or duplicate native participants")
    if len({row["manifest_id"] for row in rows}) != 1 or any(
        row["phase"] != "prepared" or row["plan_id"] != plan.plan_id for row in rows
    ):
        raise ValueError("Native shard bindings disagree on the exact manifest or preparation phase")
    receiver_bytes = dict(plan.expected_receiver_bytes)
    for row in rows:
        if row["rank"] in receiver_bytes:
            if (
                type(row["expected_receiver_bytes"]) is not int
                or row["expected_receiver_bytes"] != receiver_bytes[row["rank"]]
            ):
                raise ValueError("Native bound receiver bytes differ from collected installed storage")
        elif row.get("source_lease_transferred") is not True:
            raise ValueError("Native policy preparation did not retain learner ownership")
    return rows


@asynccontextmanager
async def prepared_shard_diagnostic(
    driver, preparation_id, geometry, options, *, endpoint_factory, output_uri, capture
):
    """Keep preparation alive until the caller completes its versioned interval.

    Endpoint allocation is explicit and must be qualified by the native packet.
    All dispatched siblings settle before teardown. Preparation never resumes
    inference; only the complete versioned install/replay interval may do that.
    """
    geometry.validate()
    options.validate()
    if not callable(endpoint_factory) or not callable(capture) or not output_uri:
        raise ValueError("Shard coordinator needs explicit rendezvous and durable evidence sinks")
    client = driver.inference_engine_client

    async def policy(method, *args):
        refs = driver.policy_model.async_run_ray_method("pass_through", method, *args)
        return await settled(*refs)

    primary = None
    try:
        await settled(client.pause_generation(settle_native_calls=True))
        collected = await settled(
            policy("collect_shard_policy_preparation", preparation_id, geometry, output_uri),
            client.collect_shard_receiver_preparation(preparation_id, geometry),
        )
        plan = plan_shard_preparation(
            receipt_rows(collected[0]), receipt_rows(collected[1]), geometry, options, endpoint_factory
        )
        # Retain metadata/group membership before communicator creation can fail.
        capture({"phase": "planned", "plan_id": plan.plan_id, "plan": asdict(plan)})
        bindings = await settled(
            policy("bind_shard_policy_preparation", plan, output_uri),
            client.bind_shard_receiver_preparation(plan, output_uri),
        )
        capture({"phase": "bindings-returned", "plan_id": plan.plan_id, "bindings": receipt_rows(bindings)})
        rows = validate_bindings(bindings, plan)
        yield plan, rows
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            closed = await settled(
                policy("close_shard_policy_preparation", preparation_id),
                client.close_shard_receiver_preparation(preparation_id),
            )
            capture({"phase": "closed", "preparation_id": preparation_id, "receipts": receipt_rows(closed)})
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Shard preparation cleanup: {type(error).__name__}: {error}")
