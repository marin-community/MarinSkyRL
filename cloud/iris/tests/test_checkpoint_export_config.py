"""Behavior tests for config-native checkpoint export derivation."""

from dataclasses import replace

from omegaconf import OmegaConf

from cloud.iris.export_hf_checkpoint import (
    ExportJobSpec,
    build_command,
    checkpoint_export_launch_config,
)
from skyrl_train.hf_export_schema import HFExportRequest


def _training_config():
    return OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt", "mode": "train", "export_hf": True},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.main_base", "profile": "fsdp"},
            "iris": {
                "job_name": "run",
                "allocation": {"num_nodes": 2, "gpus_per_node": 8, "gpu_variant": "H100"},
            },
            "artifacts": {
                "checkpoint_root": "s3://run/checkpoints",
                "export_root": "s3://run/exports",
                "attempts_root": "s3://run/attempts",
                "resolved_config_uri": "s3://run/resolved.yaml",
                "terminal_manifest_uri": "s3://run/terminal.json",
            },
            "skyrl": {
                "trainer": {
                    "strategy": "fsdp2",
                    "placement": {"policy_num_nodes": 2, "policy_num_gpus_per_node": 8},
                    "policy": {"model": {"path": "Qwen/Qwen3-8B"}},
                },
                "generator": {"inference_engine_tensor_parallel_size": 1},
                "model_num_attention_heads": 8,
            },
        }
    )


def _request() -> HFExportRequest:
    return HFExportRequest(
        step=7,
        checkpoint_base_path="s3://run/checkpoints",
        checkpoint_path="s3://run/checkpoints/global_step_7",
        export_path="s3://run/exports",
        model_path="Qwen/Qwen3-8B",
        num_nodes=1,
        gpus_per_node=4,
    )


def _spec(request: HFExportRequest) -> ExportJobSpec:
    return ExportJobSpec(
        request=request,
        cluster="cw-rno2a",
        priority="batch",
        gpu_variant="H100",
        job_name="run-export",
        timeout=7200,
        no_wait=False,
        launch_config_path="/tmp/export.yaml",
    )


def test_checkpoint_export_config_carries_request_fields_as_data() -> None:
    request = _request()
    config = checkpoint_export_launch_config(_training_config(), request, _spec(request))

    assert config.run.mode == "checkpoint_export"
    assert config.runtime.entrypoint == "skyrl_train.entrypoints.checkpoint_export"
    assert config.runtime.profile == "fsdp-export"
    assert config.iris.timeout == 7200
    assert config.skyrl.trainer.placement.policy_num_nodes == 1
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4
    assert config.skyrl.checkpoint_export.checkpoint_path == request.checkpoint_path
    assert config.skyrl.checkpoint_export.export_root == request.export_path


def test_export_backend_uses_one_launch_document() -> None:
    request = _request()
    command = build_command(_spec(request))

    assert command[:4] == [command[0], "-m", "cloud.iris.launch", "iris"]
    assert command[-2:] == ["--config", "/tmp/export.yaml"]


def test_checkpoint_export_config_preserves_federated_routing() -> None:
    request = _request()
    spec = replace(
        _spec(request),
        target_cluster="cw-rno2a",
        parent_cluster_config="/tmp/marin.yaml",
    )

    config = checkpoint_export_launch_config(_training_config(), request, spec)

    assert config.iris.target_cluster == "cw-rno2a"
    assert config.iris.parent_cluster_config == "/tmp/marin.yaml"
