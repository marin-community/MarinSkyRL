"""Behavior tests for terminal checkpoint export command construction."""

import sys

from skyrl_train.hf_export_schema import HFExportRequest

from cloud.iris import terminal_policy
from cloud.iris.export_hf_checkpoint import ExportJobSpec, build_command
from cloud.iris.terminal_policy import TerminalPolicyExport, storage_user_from_resource_path


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
        receipt_uri="s3://example/run/receipts/attempt-1.json",
        request_fingerprint="request-1",
        attempt_id="attempt-1",
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
    assert overrides["checkpoint_export.completion_receipt_uri"] == spec.receipt_uri
    assert overrides["checkpoint_export.request_fingerprint"] == spec.request_fingerprint
    assert overrides["checkpoint_export.attempt_id"] == spec.attempt_id


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


def test_terminal_policy_export_uses_exact_step_and_receipt_metadata(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        terminal_policy.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0
    )
    spec = TerminalPolicyExport(
        checkpoint_root="s3://bucket/checkpoints",
        export_root="s3://bucket/exports",
        config_path="config.yaml",
        model_path="org/model",
        model_source_uri=None,
        model_source_identity=None,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        cluster="cw-us-east-02a",
        priority="batch",
        job_name="snowball",
        global_step=25,
        receipt_uri="s3://bucket/export/receipts/attempt-2.json",
        request_fingerprint="fingerprint-2",
        attempt_id="attempt-2",
        timeout_seconds=1800,
    )

    terminal_policy.submit_terminal_policy_export(spec)

    command, kwargs = calls[0]
    assert "--request" not in command
    assert command[command.index("--step") + 1] == "25"
    assert command[command.index("--export_path") + 1] == spec.export_root
    assert command[command.index("--export-receipt-uri") + 1] == spec.receipt_uri
    assert command[command.index("--export-request-fingerprint") + 1] == spec.request_fingerprint
    assert command[command.index("--export-attempt-id") + 1] == spec.attempt_id
    assert command[command.index("--timeout") + 1] == "1800"
    assert kwargs["stdout"] is sys.stderr
