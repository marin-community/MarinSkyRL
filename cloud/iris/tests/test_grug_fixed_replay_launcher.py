from __future__ import annotations

from argparse import Namespace

import pytest

from cloud.iris.grug_fixed_replay_launcher import main, task_command, validate_request


def request(**overrides):
    values = {
        "runtime_commit": "a" * 40,
        "task_image": "registry/image@sha256:" + "b" * 64,
        "model": "org/model",
        "model_revision": "c" * 40,
        "nodes": 4,
        "rendezvous_dir": "s3://bucket/run",
    }
    values.update(overrides)
    return Namespace(**values)


def benchmark_argv(**overrides):
    values = {
        "--source-revision": "a" * 40,
        "--image": "registry/image@sha256:" + "b" * 64,
        "--model": "org/model",
        "--model-revision": "c" * 40,
        "--sample": "1",
        "--mode": "headline",
    }
    values.update(overrides)
    return [item for pair in values.items() for item in pair]


def test_request_binds_runtime_image_model_sample_and_shape(monkeypatch):
    args = request()
    monkeypatch.setattr(
        "cloud.iris.grug_fixed_replay_launcher.resolve_launcher_source",
        lambda: Namespace(commit=args.runtime_commit),
    )

    validate_request(args, benchmark_argv())

    command = task_command(args, benchmark_argv())
    shell = command[-1]
    assert "--prestage-model-revision " + args.model_revision in shell
    assert "/skyrl-train/scripts/grug_fixed_replay_benchmark.py" in shell


@pytest.mark.parametrize(
    ("nodes", "mode"),
    ((1, "headline"), (4, "preflight")),
)
def test_request_rejects_wrong_node_shape(monkeypatch, nodes, mode):
    args = request(nodes=nodes)
    monkeypatch.setattr(
        "cloud.iris.grug_fixed_replay_launcher.resolve_launcher_source",
        lambda: Namespace(commit=args.runtime_commit),
    )

    with pytest.raises(ValueError, match="requires"):
        validate_request(args, benchmark_argv(**{"--mode": mode}))


def test_dry_run_does_not_contact_iris(monkeypatch, tmp_path):
    args = request(
        cluster_config=tmp_path / "cluster.yaml",
        job_name="dry-run",
        priority="production",
        cpu=48.0,
        memory="1500GB",
        disk="1000GB",
        dry_run=True,
    )
    monkeypatch.setattr("cloud.iris.grug_fixed_replay_launcher.parse_args", lambda: (args, benchmark_argv()))
    monkeypatch.setattr(
        "cloud.iris.grug_fixed_replay_launcher.resolve_launcher_source",
        lambda: Namespace(commit=args.runtime_commit),
    )
    monkeypatch.setattr("cloud.iris.grug_fixed_replay_launcher.build_runtime_bundle", lambda _commit: tmp_path)
    monkeypatch.setattr(
        "cloud.iris.grug_fixed_replay_launcher.iris_client",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not contact Iris"),
    )

    main()
