"""Behavior tests for terminal checkpoint export command construction."""

from cloud.iris.export_hf_checkpoint import ExportJobSpec, build_command
from cloud.iris.terminal_policy import storage_user_from_resource_path
from skyrl_train.hf_export_schema import HFExportRequest


def test_export_command_encodes_lifecycle_storage_as_valid_hydra_values(parse_hydra_overrides) -> None:
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
        rl_config="config.yaml",
        cluster="cw-rno2a",
        priority="batch",
        job_name="run-export-step-1",
        timeout=7200,
        no_wait=False,
        cluster_config="/tmp/cw-rno2a.yaml",
        cpu=96,
        memory="1600GB",
        disk="800GB",
        storage_user="alice",
    )

    command = build_command(spec)
    encoded = [command[index + 1] for index, value in enumerate(command) if value == "--skyrl_override"]
    overrides = parse_hydra_overrides(encoded)

    assert command[command.index("--cluster-config") + 1] == spec.cluster_config
    assert command[command.index("--cpu") + 1] == str(spec.cpu)
    assert command[command.index("--memory") + 1] == spec.memory
    assert command[command.index("--disk") + 1] == spec.disk
    assert command[command.index("--storage-user") + 1] == spec.storage_user
    assert "--target-cluster" not in command
    assert overrides["checkpoint_export.checkpoint_path"] == request.checkpoint_path
    assert overrides["checkpoint_export.export_root"] == request.export_path


def test_export_command_preserves_federated_submission_configs() -> None:
    request = HFExportRequest(
        step=1,
        checkpoint_base_path="s3://example/run/checkpoints",
        checkpoint_path="s3://example/run/checkpoints/global_step_1",
        export_path="s3://example/run/exports",
        model_path="Qwen/Qwen3-0.6B",
        num_nodes=1,
        gpus_per_node=8,
    )
    spec = ExportJobSpec(
        request=request,
        rl_config="config.yaml",
        cluster="cw-rno2a",
        priority="batch",
        job_name="run-export-step-1",
        timeout=7200,
        no_wait=False,
        cluster_config="/tmp/cw-rno2a.yaml",
        target_cluster="cw-rno2a",
        parent_cluster_config="/tmp/marin.yaml",
    )

    command = build_command(spec)

    assert command[command.index("--cluster-config") + 1] == spec.cluster_config
    assert command[command.index("--target-cluster") + 1] == spec.target_cluster
    assert command[command.index("--parent-cluster-config") + 1] == spec.parent_cluster_config


def test_storage_user_is_derived_from_policy_paths() -> None:
    assert storage_user_from_resource_path("s3://bucket/tmp/ttl=14d/skyrl/users/alice/run/checkpoints") == "alice"
    assert storage_user_from_resource_path("s3://bucket/run/checkpoints") is None
