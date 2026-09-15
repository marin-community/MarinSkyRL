import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SUBMITTER_PATH = Path(__file__).parents[3] / "skyrl-train" / "ci" / "opd" / "tinker_repro" / "submit_iris.py"
SPEC = importlib.util.spec_from_file_location("tinker_opd_aime24_submitter", SUBMITTER_PATH)
assert SPEC is not None and SPEC.loader is not None
submitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submitter
SPEC.loader.exec_module(submitter)


def test_submit_sends_secret_only_in_environment_and_requests_cpu_worker(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Client:
        def submit(self, **kwargs: Any) -> SimpleNamespace:
            captured["submit"] = kwargs
            return SimpleNamespace(job_id="/ben/tinker-opd-aime24")

    @contextmanager
    def open_iris_client(*, cluster_name: str, workspace: Path) -> Iterator[Client]:
        captured["cluster_name"] = cluster_name
        captured["workspace"] = workspace
        yield Client()

    monkeypatch.setattr(submitter, "open_iris_client", open_iris_client)
    api_key = "sentinel-tinker-api-key"
    job_id = submitter.submit(
        submitter.SubmissionConfig(
            checkpoint="tinker://released/sampler_weights/final",
            save_dir="s3://evaluation/results",
            max_examples=1,
        ),
        tinker_api_key=api_key,
    )

    assert job_id == "/ben/tinker-opd-aime24"
    assert captured["cluster_name"] == "cw-rno2a"
    assert captured["workspace"] == SUBMITTER_PATH.parents[4]

    request = captured["submit"]
    assert request["name"] == "tinker-opd-aime24"
    assert request["environment"].env_vars == {"TINKER_API_KEY": api_key}
    assert api_key not in request["entrypoint"].command

    resources = request["resources"].to_proto()
    assert resources.cpu_millicores == 2_000
    assert resources.memory_bytes == 8 * 1024**3
    assert resources.disk_bytes == 20 * 1024**3
    assert not resources.HasField("device")

    constraint = request["constraints"][0].to_proto()
    assert constraint.key == "preemptible"
    assert constraint.value.string_value == "false"
    assert submitter.job_pb2.PriorityBand.Name(request["priority_band"]) == "PRIORITY_BAND_INTERACTIVE"
    assert request["replicas"] == 1
    assert request["max_retries_failure"] == 0
    assert request["max_task_failures"] == 0
