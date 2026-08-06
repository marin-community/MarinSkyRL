from __future__ import annotations

from argparse import Namespace

import pytest

from cloud.iris.grug_fixed_replay_launcher import controller_command, main, task_command, validate_request


def request(**overrides):
    values = {
        "runtime_commit": "a" * 40,
        "task_image": "registry/image@sha256:" + "b" * 64,
        "model": "org/model",
        "model_revision": "c" * 40,
        "flash_attn_wheel_sha256": "d" * 64,
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
        "--flash-attn-wheel-sha256": "d" * 64,
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

    command = controller_command(args, benchmark_argv())
    revision_flag = command.index("--prestage-model-revision")
    benchmark = command.index("--") + 1
    assert command[revision_flag + 1] == args.model_revision
    assert command[benchmark:] == [
        "python",
        "/app/marinskyrl/skyrl-train/scripts/grug_fixed_replay_benchmark.py",
        *benchmark_argv(),
    ]


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


def test_task_command_installs_and_verifies_the_pinned_flash_attn_wheel():
    args = request()

    command = task_command(args, benchmark_argv())

    assert command[:2] == ["bash", "-c"]
    shell = command[2]
    assert 'test "$wheel_count" = 1' in shell
    assert args.flash_attn_wheel_sha256 in shell
    assert "sha256sum -c -" in shell
    assert 'uv pip install --python "$(command -v python)" --no-deps "$flash_wheel"' in shell
    assert "import flash_attn, flash_attn_2_cuda" in shell


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
        "cloud.iris.grug_fixed_replay_launcher.open_iris_client",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not contact Iris"),
    )

    main()
