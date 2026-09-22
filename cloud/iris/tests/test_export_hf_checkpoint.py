"""Behavior tests for terminal checkpoint export command construction."""

from dataclasses import replace

import pytest

from cloud.iris import export_hf_checkpoint
from cloud.iris.export_hf_checkpoint import ExportJobSpec, build_command
from cloud.iris.terminal_policy import (
    TerminalPolicyExport,
    submit_terminal_policy_export,
)
from skyrl_train.hf_export_schema import HFExportRequest


def test_export_command_uses_config_path_for_lifecycle_values() -> None:
    request = HFExportRequest(
        step=1,
        checkpoint_base_path="s3://example/tmp/ttl=14d/run/checkpoints",
        checkpoint_path="s3://example/tmp/ttl=14d/run/checkpoints/global_step_1",
        export_path="s3://example/marin/users/alice/run/exports",
        model_path="Qwen/Qwen3-0.6B",
        num_nodes=1,
        gpus_per_node=8,
    )
    spec = ExportJobSpec(
        request=request,
        cluster="cw-rno2a",
        priority="batch",
        gpu_variant="H100",
        job_name="run-export-step-1",
        timeout=7200,
        no_wait=False,
        cluster_config="/tmp/cw-rno2a.yaml",
        cpu=96,
        memory="1600GB",
        disk="800GB",
        launch_config_path="/tmp/checkpoint-export.yaml",
    )

    command = build_command(spec)

    assert command[-2:] == ["--config", "/tmp/checkpoint-export.yaml"]


def test_terminal_policy_export_preserves_gpu_variant(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr("cloud.iris.terminal_policy.terminal_checkpoint_step", lambda _root: 7)
    monkeypatch.setattr("cloud.iris.terminal_policy.resource_exists", lambda _uri: True)
    monkeypatch.setattr(
        "cloud.iris.terminal_policy.subprocess.call",
        lambda command, **_kwargs: commands.append(command) or 0,
    )

    submit_terminal_policy_export(
        TerminalPolicyExport(
            checkpoint_root="s3://bucket/checkpoints",
            config_path="config.yaml",
            gpu_variant="GB200",
            cluster="cw-us-east-08a",
            priority="batch",
            job_name="iceball",
        )
    )

    command = commands[0]
    assert command[command.index("--gpu-variant") + 1] == "GB200"


def test_requested_four_rank_export_reserves_whole_eight_gpu_node(monkeypatch) -> None:
    request = HFExportRequest(
        step=2,
        checkpoint_base_path="s3://bucket/marin/users/alice/run/checkpoints",
        checkpoint_path="s3://bucket/marin/users/alice/run/checkpoints/global_step_2",
        export_path="s3://bucket/marin/users/alice/run/exports",
        model_path="BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT",
        num_nodes=1,
        gpus_per_node=4,
    )
    monkeypatch.setattr(export_hf_checkpoint, "_read_hf_export_request", lambda _path: request)
    monkeypatch.setattr(
        export_hf_checkpoint,
        "write_checkpoint_export_config",
        lambda _path, _request, _spec: "/tmp/checkpoint-export.yaml",
    )
    parser = export_hf_checkpoint.argument_parser()
    args = parser.parse_args(
        [
            "--request",
            request.checkpoint_path,
            "--rl_config",
            "config.yaml",
            "--allocation-gpus-per-node",
            "8",
            "--gpu-variant",
            "H100",
        ]
    )

    spec = export_hf_checkpoint.request_spec(args, parser)
    command = build_command(spec)

    assert spec.request.gpus_per_node == 4
    assert spec.allocated_gpus_per_node == 8
    assert command[-2:] == ["--config", "/tmp/checkpoint-export.yaml"]
    with pytest.raises(ValueError, match="fewer GPUs than the saved policy geometry"):
        build_command(replace(spec, allocation_gpus_per_node=2))
