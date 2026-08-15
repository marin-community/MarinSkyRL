import json
import platform
import socket

import pytest
import ray

from cloud.iris.task_runtime import (
    RENDEZVOUS_FILENAME,
    RendezvousPayload,
    validate_rendezvous_runtime,
    write_rendezvous,
)


def _head_payload() -> RendezvousPayload:
    return RendezvousPayload(
        head_ip="10.0.0.1",
        head_node="head-node",
        port=6379,
        num_tasks=2,
        python_version="3.12.13",
        ray_version="2.51.1",
        written_at=1.0,
    )


def test_rendezvous_publishes_head_runtime_identity(tmp_path):
    write_rendezvous(str(tmp_path), "10.0.0.1", 6379)

    payload = json.loads((tmp_path / RENDEZVOUS_FILENAME).read_text())
    assert payload["head_node"] == socket.gethostname()
    assert payload["python_version"] == platform.python_version()
    assert payload["ray_version"] == ray.__version__


def test_matching_rendezvous_runtime_is_accepted():
    head = _head_payload()

    validated = validate_rendezvous_runtime(
        head,
        worker_node="worker-node",
        python_version="3.12.13",
        ray_version="2.51.1",
    )

    assert validated == head


@pytest.mark.parametrize(
    ("python_version", "ray_version", "expected_versions"),
    [
        ("3.12.14", "2.51.1", ("Python 3.12.13", "Python 3.12.14")),
        ("3.12.13", "2.52.0", ("Ray 2.51.1", "Ray 2.52.0")),
    ],
)
def test_runtime_skew_names_both_nodes_and_versions(python_version, ray_version, expected_versions):
    with pytest.raises(RuntimeError) as error:
        validate_rendezvous_runtime(
            _head_payload(),
            worker_node="worker-node",
            python_version=python_version,
            ray_version=ray_version,
        )

    message = str(error.value)
    assert "head-node" in message
    assert "worker-node" in message
    for version in expected_versions:
        assert version in message
