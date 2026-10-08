"""Behavior tests for config-native checkpoint export derivation."""

import sys
from dataclasses import replace

from omegaconf import OmegaConf

from cloud.iris.export_hf_checkpoint import (
    ExportJobSpec,
    _run_export,
    checkpoint_export_launch_config,
)
from marinskyrl.distillation import compile_distillation_plan
from skyrl_train.hf_export_schema import HFExportRequest


def _training_config():
    return OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt", "mode": "train", "export_hf": True},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.main_base", "profile": "megatron"},
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
                    "strategy": "megatron",
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
    assert config.runtime.profile == "megatron-export"
    assert config.iris.timeout == 7200
    assert config.iris.allocation.gpu_variant == "H100"
    assert config.skyrl.trainer.placement.policy_num_nodes == 1
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4
    assert config.skyrl.checkpoint_export.checkpoint_path == request.checkpoint_path
    assert config.skyrl.checkpoint_export.export_root == request.export_path


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


def test_checkpoint_export_config_preserves_saved_policy_geometry_on_whole_nodes() -> None:
    request = _request()
    spec = replace(_spec(request), allocation_gpus_per_node=8)

    config = checkpoint_export_launch_config(_training_config(), request, spec)

    assert config.iris.allocation.gpus_per_node == 8
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4


def test_nested_export_keeps_parent_response_stream_clean(capfd) -> None:
    request = _request()
    spec = replace(_spec(request), no_wait=True)

    _run_export(spec, [sys.executable, "-c", "print('nested export response')"])

    captured = capfd.readouterr()
    assert "nested export response" not in captured.out
    assert "nested export response" in captured.err


def _distillation_training_config():
    config = _training_config()
    config.skyrl.trainer.algorithm = {
        "advantage_estimator": "uniform",
        "policy_loss_type": "importance_sampling",
        "distillation": {
            "objective": "sampled_reverse_kl",
            "routing_plan": "opd",
            "coefficient": 1.0,
            "reward_mode": "replace",
        },
    }
    config.skyrl.teachers = {
        "expert": {
            "source": "local_inference",
            "placement": "pinned",
            "backend": "vllm",
            "evidence": "chosen_token",
            "model": {"path": "s3://models/expert/hf/step-272", "revision": "step-272"},
            "resources": {
                "num_nodes": 1,
                "gpus_per_node": 8,
                "tensor_parallel_size": 1,
                "data_parallel_size": 8,
                "expert_parallel_size": 8,
                "colocation_group": "teacher_expert",
            },
        }
    }
    config.skyrl.teacher_routing = {
        "opd": {"revision": "v1", "routes": {"default": {"teacher": "expert", "weight": 1.0}}}
    }
    return config


def test_checkpoint_export_config_drops_distillation_with_its_teachers() -> None:
    training = _distillation_training_config()
    assert compile_distillation_plan(OmegaConf.to_container(training.skyrl)) is not None
    request = _request()
    config = checkpoint_export_launch_config(training, request, _spec(request))
    skyrl = OmegaConf.to_container(config.skyrl)
    assert "teachers" not in skyrl
    assert "teacher_routing" not in skyrl
    assert "distillation" not in skyrl["trainer"]["algorithm"]
    # Launch validation compiles the distillation plan from the export document; a document that
    # kept the objective without its teachers was rejected before the export job could start.
    assert compile_distillation_plan(skyrl) is None
