"""Export a recorded training checkpoint as an independently retryable operation."""

from __future__ import annotations

import hashlib
import json
import posixpath
import subprocess
import tempfile
from dataclasses import asdict
from typing import Protocol

from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from cloud.iris.job import _path_exists, _write_json
from cloud.iris.protocol import (
    AttemptState,
    LaunchMode,
    SkyRLExportResponse,
    SkyRLExportSpec,
    SkyRLJobSpec,
    SkyRLModel,
    job_spec,
)
from cloud.iris.runtime_bundle import runtime_bundle_inputs
from cloud.iris.terminal_policy import TerminalPolicyExport, storage_user_from_resource_path, submit_terminal_policy_export
from cloud.iris.training_result import read_json, read_training_result, validate_native_checkpoint
from marinskyrl.checkpoint_paths import policy_export_path
from marinskyrl.export_completion import verify_export_receipt
from marinskyrl.training_completion import CompletionMode


class ExportBackend(Protocol):
    """External conversion job boundary."""

    def export(self, request: TerminalPolicyExport) -> None: ...


class IrisExportBackend:
    def export(self, request: TerminalPolicyExport) -> None:
        submit_terminal_policy_export(request)


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _export_config(source: SkyRLJobSpec, resolved: dict) -> str:
    """Restore the effective policy settings, including training-time overrides."""
    cfg = OmegaConf.create(source.request.config_yaml)
    for override in OverridesParser.create().parse_overrides(resolved["hydra_args"]):
        if override.key_or_group.startswith("trainer."):
            if override.is_delete():
                raise ValueError("Cannot export a training config with deleted trainer settings")
            OmegaConf.update(cfg, override.key_or_group, override.value(), merge=False, force_add=True)
    cfg.trainer.pop("completion", None)
    return OmegaConf.to_yaml(cfg, resolve=False)


def execute_export(
    spec: SkyRLExportSpec,
    *,
    mode: LaunchMode = LaunchMode.WAIT,
    backend: ExportBackend | None = None,
) -> SkyRLExportResponse:
    """Validate source provenance before GPU submission; never relaunch training."""
    if mode is LaunchMode.DETACH:
        raise ValueError("Detached export does not provide a completed model artifact")
    request = spec.request
    if _path_exists(request.output.terminal_manifest_uri):
        raise ValueError(f"Terminal manifest is immutable and already exists: {request.output.terminal_manifest_uri}")
    manifest = read_json(request.training_manifest_uri)
    source = job_spec(manifest)
    source_response = manifest["response"]
    if source_response["state"] != AttemptState.SUCCEEDED or source.request.completion_mode != CompletionMode.CHECKPOINT:
        raise ValueError("Export requires a successful checkpoint training manifest")
    if request.training_manifest_uri != source.request.output.terminal_manifest_uri:
        raise ValueError("Training manifest URI differs from the recorded training destination")
    runtime_bundle_inputs(source.request.runtime.commit)
    training = read_training_result(source.request, check_latest=False, check_files=False)
    if json.loads(json.dumps(asdict(training))) != source_response["training"]:
        raise ValueError("Training manifest and completion receipt disagree")
    checkpoint = training.checkpoint
    assert checkpoint is not None
    resolved = read_json(training.resolved_config_uri)
    export_config = _export_config(source, resolved)
    export_path = policy_export_path(request.output.export_root, checkpoint.global_step)
    fingerprint = _digest({
        "schema_version": spec.schema_version,
        "training_manifest": manifest,
        "resolved_config": resolved,
        "checkpoint": checkpoint.to_dict(),
        "export_path": export_path,
        "runtime": asdict(source.request.runtime),
    })
    receipt_uri = posixpath.join(request.output.export_root, "receipts", f"{fingerprint}.json")
    execution = spec.execution
    response_fields = dict(
        run_id=source.request.run_id,
        attempt_id=request.attempt_id,
        runtime=source.request.runtime,
        training_iris_job_id=source_response["iris_job_id"],
    )
    reused = False
    try:
        if _path_exists(receipt_uri):
            verify_export_receipt(receipt_uri, fingerprint, export_path, checkpoint.global_step)
            reused = True
        else:
            validate_native_checkpoint(checkpoint)
        if mode is LaunchMode.PREPARE:
            return SkyRLExportResponse(**response_fields, state=AttemptState.PREPARED, model=None, failure=None,
                                       reused_export=reused)
        if not reused:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as config_file:
                config_file.write(export_config)
                config_file.flush()
                (backend or IrisExportBackend()).export(TerminalPolicyExport(
                    checkpoint_root=source.request.output.checkpoint_root,
                    global_step=checkpoint.global_step,
                    export_root=request.output.export_root,
                    config_path=config_file.name,
                    model_path=source.request.model.local_path,
                    model_source_uri=source.request.model.uri,
                    model_source_identity=source.request.model.identity,
                    policy_num_nodes=source.request.topology.role_plan.policy_num_nodes,
                    policy_num_gpus_per_node=source.request.topology.role_plan.policy_num_gpus_per_node,
                    cluster=execution.cluster,
                    priority=execution.priority,
                    job_name=execution.job_name,
                    cluster_config=execution.cluster_config,
                    target_cluster=execution.target_cluster,
                    parent_cluster_config=execution.parent_cluster_config,
                    cpu=execution.cpu, memory=execution.memory, disk=execution.disk,
                    storage_user=storage_user_from_resource_path(source.request.output.checkpoint_root),
                    receipt_uri=receipt_uri, request_fingerprint=fingerprint, attempt_id=request.attempt_id,
                    timeout_seconds=execution.timeout_seconds or None,
                ))
            verify_export_receipt(receipt_uri, fingerprint, export_path, checkpoint.global_step)
        model = SkyRLModel(
            policy_export_uri=export_path, global_step=checkpoint.global_step,
            tokenizer_uri=source.request.model.tokenizer_uri,
            tokenizer_revision=source.request.model.tokenizer_revision,
            checkpoint_root=source.request.output.checkpoint_root,
            terminal_manifest_uri=request.output.terminal_manifest_uri,
        )
        response = SkyRLExportResponse(**response_fields, state=AttemptState.SUCCEEDED, model=model,
                                       failure=None, reused_export=reused)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        response = SkyRLExportResponse(**response_fields, state=AttemptState.FAILED, model=None, failure=str(error))
    payload = {"schema_version": spec.schema_version, "request": asdict(request),
               "execution": asdict(execution), "response": asdict(response),
               "training_manifest_sha256": _digest(manifest), "export_receipt_uri": receipt_uri}
    _write_json(posixpath.join(request.output.attempts_root, f"{request.attempt_id}.json"), payload)
    if response.state is AttemptState.SUCCEEDED:
        _write_json(request.output.terminal_manifest_uri, payload)
    return response
