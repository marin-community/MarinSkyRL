import pytest

# `prometheus_client` and `rigging.telemetry` both come from the optional `telemetry` extra.
# The documented CPU install (`uv sync --frozen --extra dev`) omits it, and an unguarded
# import here aborts collection for the whole suite rather than this module alone.
pytest.importorskip("prometheus_client")
pytest.importorskip("rigging.telemetry")

from prometheus_client.parser import text_string_to_metric_families
from rigging import telemetry

from skyrl_train.ray_metrics import transform_ray_metrics


def test_transform_ray_metrics_selects_logical_state_and_bounds_identity() -> None:
    scrape = """
# TYPE ray_tasks gauge
ray_tasks{State="RUNNING",JobId="A1",WorkerId="worker-1",Name="task-1",Source="executor",IsRetry="0"} 2
ray_tasks{State="RUNNING",JobId="A1",WorkerId="worker-2",Name="task-2",Source="executor",IsRetry="0"} 3
ray_tasks{State="FINISHED",JobId="",WorkerId="worker-2",Name="task-2",Source="owner",IsRetry="0"} 1
# TYPE ray_tasks_internal gauge
ray_tasks_internal{State="RUNNING",JobId="A1"} 99
# TYPE ray_resources gauge
ray_resources{Name="CPU",State="AVAILABLE",NodeAddress="10.0.0.1"} 6
ray_resources{Name="GPU",State="USED",NodeAddress="10.0.0.1"} 8
ray_resources{Name="node:10.0.0.1",State="AVAILABLE",NodeAddress="10.0.0.1"} 1
# TYPE ray_object_store_used_memory gauge
ray_object_store_used_memory{WorkerId="worker-1",NodeAddress="10.0.0.1"} 1024
# TYPE ray_scheduler_failed_worker_startup_total gauge
ray_scheduler_failed_worker_startup_total{Reason="REGISTRATION_TIMEOUT",WorkerId="worker-1"} 4
# TYPE ray_node_cpu_utilization gauge
ray_node_cpu_utilization{NodeAddress="10.0.0.1"} 75
"""

    records = transform_ray_metrics(tuple(text_string_to_metric_families(scrape)))
    by_name_and_attributes = {(record.name, tuple(sorted(record.attributes.items()))): record for record in records}

    task = by_name_and_attributes[
        (
            "ray_tasks",
            (
                ("is_retry", "0"),
                ("ray_job_id", "A1"),
                ("source", "executor"),
                ("state", "running"),
            ),
        )
    ]
    assert task.value == 5
    assert task.unit == "{task}"
    assert task.source_temporality == telemetry.CURRENT_SNAPSHOT

    owner = next(record for record in records if record.name == "ray_tasks" and record.attributes["source"] == "owner")
    assert owner.value == 1
    assert "ray_job_id" not in owner.attributes

    resources = {record.attributes["resource"]: record for record in records if record.name == "ray_resources"}
    assert {name: record.value for name, record in resources.items()} == {"cpu": 6, "gpu": 8}
    assert all(record.unit == "{resource}" for record in resources.values())

    object_store = next(record for record in records if record.name == "ray_object_store_used_memory")
    assert object_store.value == 1024
    assert object_store.unit == "By"
    assert object_store.attributes == {}

    failure = next(record for record in records if record.name == "ray_scheduler_failed_worker_startup_total")
    assert failure.attributes == {"reason": "registration_timeout"}
    assert failure.source_temporality == telemetry.CUMULATIVE_SNAPSHOT

    assert not {"ray_tasks_internal", "ray_node_cpu_utilization"}.intersection(record.name for record in records)
    assert not {"WorkerId", "Name", "NodeAddress", "SessionName"}.intersection(
        attribute for record in records for attribute in record.attributes
    )
    assert all(value for record in records for value in record.attributes.values())
