# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize the real stage-2 backend request without submitting a job."""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from cloud.iris import iris_backend as backend
from cloud.iris.protocol import LaunchMode, job_spec
from google.protobuf.json_format import MessageToDict
from iris.client.client import IrisClient


def serialize_packet(report: dict, output: Path) -> None:
    request = report["request"]
    spec = job_spec(
        {
            "schema_version": 2,
            "request": request,
            "execution": {
                **report["execution"],
                "job_name": request["run_id"].replace("/", "-") + "-" + request["attempt_id"],
            },
        }
    )
    seen = []

    class Sink:
        def submit_job(self, **kwargs):
            seen.append(kwargs)
            return kwargs["job_id"]

        def close(self):
            pass

        def shutdown(self, wait=True):
            pass

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config:
        config.write(request["config_yaml"])
        config.flush()
        args = backend.resolved_launch_args(backend.job_launch_argv(spec, config.name, mode=LaunchMode.DETACH))
        captured = io.StringIO()
        with (
            contextlib.redirect_stdout(captured),
            contextlib.redirect_stderr(captured),
            patch.object(backend, "_ambient_in_cluster_client", return_value=IrisClient(Sink())),
        ):
            outcome = backend.launch(args, report["msr"])
        assert len(seen) == 1
        native = seen[0]
        environment = native["environment"]
        env = MessageToDict(environment, preserving_proto_field_name=True)["env_vars"]
        expected = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
        assert all(env.get(k) == v for k, v in expected.items()), "Missing diagnostic environment on gang task spec"
        assert native["replicas"] == 5
        assert native["max_retries_failure"] == native["max_retries_preemption"] == native["max_task_failures"] == 1
        assert native["timeout"].to_seconds() == 5400
        resources = MessageToDict(native["resources"], preserving_proto_field_name=True)
        assert resources["device"]["gpu"] == {"count": 8, "variant": "H100"}
        result = {
            "diagnostic_environment": expected,
            "resources": resources,
            "replicas": native["replicas"],
            "max_retries_failure": native["max_retries_failure"],
            "max_retries_preemption": native["max_retries_preemption"],
            "max_task_failures": native["max_task_failures"],
            "timeout_seconds": native["timeout"].to_seconds(),
            "coscheduling": MessageToDict(native["coscheduling"], preserving_proto_field_name=True),
            "request_hash": report["request_hash"],
            "msr": report["msr"],
            "mode": str(outcome.job_state),
        }
        output.write_text(json.dumps(result, indent=2))
        print("E61_STAGE2_ACTUAL_NATIVE_SERIALIZATION_PASS", report["arm"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serialize_packet(json.loads(args.packet.read_bytes()), args.output)
