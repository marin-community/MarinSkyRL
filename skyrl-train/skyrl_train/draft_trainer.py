"""Dedicated online draft training and bounded node-to-node artifact transport."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
from uuid import uuid4

import ray
from safetensors.torch import load_file, save_file
import torch

from marinskyrl.hf_model import sha256_file
from skyrl_train.distributed.tensor_transfer import (
    TensorTransferManifest,
    broadcast_tensor_payload,
    transfer_tensor_operations,
)
from skyrl_train.distributed.utils import get_free_port, init_custom_process_group
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    ONLINE_EAGLE_CAPTURE_TRANSFER_FORMAT,
    ONLINE_EAGLE_MANIFEST_FILENAME,
    ONLINE_EAGLE_TARGET_CONFIG_FILENAME,
    ONLINE_EAGLE_SCRATCH_ROOT,
    OnlineEagleTrainerRuntime,
    OnlineEagleTrainingJob,
    OnlineEagleUpdateResult,
    preserve_online_eagle_failure,
    publish_speculator_checkpoint,
    publish_online_eagle_failure_bundle,
    remove_online_eagle_scratch,
    restore_speculator_checkpoint,
    validate_online_eagle_serving_candidate,
)
from skyrl_train.utils import get_tcp_url


def _relative_transfer_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("DraftTrainer transfer paths must be nonempty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"DraftTrainer transfer path must be a normalized relative path: {value!r}")
    return Path(*path.parts)


class DraftTrainer:
    """Long-lived one-GPU owner for online draft update state and lineage."""

    def __init__(self, *, initial_draft_dir: str, initial_draft_revision: str, process_id: str):
        initial = Path(initial_draft_dir)
        if not initial.is_absolute():
            raise ValueError("DraftTrainer initial draft path must be absolute")
        if not initial_draft_revision:
            raise ValueError("DraftTrainer initial draft revision must be nonempty")
        self._initial_draft_dir = str(initial)
        self._accepted_draft_dir = str(initial)
        self._accepted_draft_revision = initial_draft_revision
        self._node_id = str(ray.get_runtime_context().get_node_id()) if ray.is_initialized() else None
        self._gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()] if ray.is_initialized() else []
        self._process_root = ONLINE_EAGLE_SCRATCH_ROOT / process_id
        self._accepted_transfer_manifest: dict[str, Any] | None = None
        self._accepted_candidate_tensors: dict[str, torch.Tensor] | None = None
        self._training_runtime: OnlineEagleTrainerRuntime | None = None
        self._draft_transfer_group = None
        self._owns_default_process_group = False

    def transfer_rendezvous(self) -> dict[str, Any]:
        """Return a trainer-node TCP endpoint for the persistent transfer group."""
        master_addr = ray._private.services.get_node_ip_address()
        return {"master_addr": master_addr, "master_port": get_free_port()}

    def init_transfer_group(
        self,
        *,
        master_addr: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Join the DraftTrainer and all serving ranks in one persistent group."""
        if self._draft_transfer_group is not None:
            raise RuntimeError("DraftTrainer transfer group is already initialized")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=get_tcp_url("127.0.0.1", get_free_port()),
                world_size=1,
                rank=0,
            )
            self._owns_default_process_group = True
        if backend == "nccl":
            torch.cuda.set_device(torch.cuda.current_device())
        self._draft_transfer_group = init_custom_process_group(
            backend=backend,
            init_method=get_tcp_url(master_addr, master_port),
            world_size=world_size,
            rank=0,
            group_name=group_name,
            timeout=timedelta(seconds=timeout_seconds),
        )
        return {"rank": 0, "world_size": world_size, "backend": backend}

    def receive_capture(self, transfer_plan: Mapping[str, Any], destination: str) -> dict[str, Any]:
        """Receive a selected multi-rank capture directly into trainer-local files."""
        if self._draft_transfer_group is None:
            raise RuntimeError("DraftTrainer transfer group is not initialized")
        if transfer_plan.get("format") != ONLINE_EAGLE_CAPTURE_TRANSFER_FORMAT:
            raise ValueError("Unsupported online EAGLE capture transfer plan")
        operations = transfer_plan.get("operations")
        files = transfer_plan.get("files")
        if not isinstance(operations, list) or not isinstance(files, list):
            raise ValueError("Online EAGLE capture transfer plan is incomplete")
        capture_dir = Path(destination)
        if capture_dir.exists():
            raise FileExistsError(f"DraftTrainer capture destination already exists: {capture_dir}")
        received = transfer_tensor_operations(
            operations,
            tensor_provider=None,
            group=self._draft_transfer_group,
        )
        assert received is not None
        staging = capture_dir.with_name(f".{capture_dir.name}.tmp-{uuid4().hex}")
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        try:
            for file_entry in files:
                relative_path = _relative_transfer_path(file_entry["destination_path"])
                tensors = {
                    tensor["name"]: received[f"{relative_path.as_posix()}::{tensor['name']}"]
                    for tensor in file_entry["tensors"]
                }
                save_file(tensors, str(staging / relative_path), metadata={"format": "pt"})
            config_path = staging / ONLINE_EAGLE_TARGET_CONFIG_FILENAME
            config_path.write_text(transfer_plan["target_config_json"])
            manifest = dict(transfer_plan["capture_manifest"])
            manifest["windows"] = [
                {**window, "sha256": sha256_file(staging / window["path"])} for window in manifest["windows"]
            ]
            manifest["target"] = {
                **manifest["target"],
                "weights_sha256": sha256_file(staging / manifest["target"]["weights_path"]),
                "config_sha256": sha256_file(config_path),
            }
            (staging / ONLINE_EAGLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
            os.replace(staging, capture_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {
            "capture_dir": str(capture_dir),
            "captured_rows": manifest["captured_rows"],
            "captured_windows": len(manifest["windows"]),
            "dropped_windows": manifest["dropped_windows"],
            "oversized_windows": manifest["oversized_windows"],
            "unselected_windows": manifest["unselected_windows"],
            "target_weights_sha256": manifest["target"]["weights_sha256"],
            "target_config_sha256": manifest["target"]["config_sha256"],
            "transfer_bytes": transfer_plan["total_bytes"],
        }

    def update(self, raw_job: Mapping[str, Any]) -> dict[str, Any]:
        """Train from one trainer-local capture and accept a complete finite candidate."""
        job = OnlineEagleTrainingJob.from_mapping(raw_job)
        if job.parent_draft_revision != self._accepted_draft_revision:
            raise RuntimeError(
                "DraftTrainer update parent does not match its accepted lineage: "
                f"expected {self._accepted_draft_revision}, got {job.parent_draft_revision}"
            )
        capture_dir = Path(job.capture_dir)
        if not capture_dir.is_dir():
            raise FileNotFoundError(f"DraftTrainer capture is not materialized: {capture_dir}")
        job = OnlineEagleTrainingJob.from_mapping(
            {
                **job.to_mapping(),
                "draft_model_dir": self._accepted_draft_dir,
            }
        )
        try:
            if self._training_runtime is None:
                self._training_runtime = OnlineEagleTrainerRuntime(job, capture_dir)
            result = self._training_runtime.update(job)
        except Exception as error:
            failure_dir = None
            preservation_error = None
            try:
                failure_dir = preserve_online_eagle_failure(job, error)
                published = publish_online_eagle_failure_bundle(failure_dir, job.failure_artifact_path)
                failure_artifact_path = published["path"]
            except Exception as failure_error:
                failure_artifact_path = job.failure_artifact_path
                preservation_error = f"{type(failure_error).__name__}: {failure_error}"
            result = OnlineEagleUpdateResult(
                active=True,
                accepted=False,
                step=job.step,
                error=f"{type(error).__name__}: {error}",
                failure_dir=failure_dir,
                failure_artifact_path=failure_artifact_path,
                failure_preservation_error=preservation_error,
            ).to_mapping()
            remove_online_eagle_scratch(capture_dir.parent)
            return {"result": result, "transfer_manifest": None}
        transfer_manifest = None
        if result.accepted:
            assert result.candidate_dir is not None
            assert result.draft_revision is not None
            assert result.weights_sha256 is not None
            transfer_manifest = self._accept_transfer_state(
                candidate_dir=result.candidate_dir,
                draft_revision=result.draft_revision,
                source_weights_sha256=result.weights_sha256,
                transfer_id=f"candidate-step-{result.step}",
            )
        remove_online_eagle_scratch(capture_dir.parent)
        return {"result": result.to_mapping(), "transfer_manifest": transfer_manifest}

    def broadcast_candidate(self, transfer_manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Broadcast the current accepted candidate directly to every vLLM rank."""
        if self._draft_transfer_group is None:
            raise RuntimeError("DraftTrainer transfer group is not initialized")
        manifest = TensorTransferManifest.from_mapping(transfer_manifest)
        if self._accepted_transfer_manifest != manifest.to_mapping() or self._accepted_candidate_tensors is None:
            raise RuntimeError(f"DraftTrainer has no matching accepted transfer: {manifest.transfer_id}")
        try:
            broadcast_tensor_payload(
                self._accepted_transfer_manifest,
                tensors=self._accepted_candidate_tensors,
                group=self._draft_transfer_group,
            )
        finally:
            self._accepted_candidate_tensors = None
        return {
            "transfer_id": manifest.transfer_id,
            "draft_revision": manifest.revision,
            "payload_sha256": manifest.payload_sha256,
            "total_bytes": manifest.total_bytes,
        }

    def restore(self, source: str, destination: str) -> dict[str, Any]:
        """Restore the exact paired draft and FP32 trainer state on the trainer node."""
        manifest = restore_speculator_checkpoint(source, destination)
        candidate_manifest = validate_online_eagle_serving_candidate(destination)
        transfer_manifest = self._accept_transfer_state(
            candidate_dir=destination,
            draft_revision=manifest["draft_revision"],
            source_weights_sha256=candidate_manifest["weights_sha256"],
            transfer_id=f"restore-{manifest['draft_revision']}",
        )
        self._training_runtime = None
        return {**manifest, "transfer_manifest": transfer_manifest}

    def publish(
        self,
        destination: str,
        draft_revision: str,
        served_target_revision: str,
        install_coverage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish the accepted draft and private trainer state from its owner."""
        if draft_revision != self._accepted_draft_revision:
            raise RuntimeError(
                f"DraftTrainer cannot publish {draft_revision!r}; accepted={self._accepted_draft_revision!r}"
            )
        return publish_speculator_checkpoint(
            self._accepted_draft_dir,
            destination,
            draft_revision=draft_revision,
            served_target_revision=served_target_revision,
            install_coverage=install_coverage,
        )

    def cleanup(self) -> dict[str, Any]:
        """Release this actor's process-scoped capture, candidate, and failure scratch."""
        remove_online_eagle_scratch(self._process_root)
        self._accepted_transfer_manifest = None
        self._accepted_candidate_tensors = None
        self._training_runtime = None
        if self._draft_transfer_group is not None:
            torch.distributed.destroy_process_group(self._draft_transfer_group)
            self._draft_transfer_group = None
        if self._owns_default_process_group:
            torch.distributed.destroy_process_group()
            self._owns_default_process_group = False
        return {"path": str(self._process_root)}

    def _accept_transfer_state(
        self,
        *,
        candidate_dir: str,
        draft_revision: str,
        source_weights_sha256: str,
        transfer_id: str,
    ) -> dict[str, Any]:
        candidate_manifest = validate_online_eagle_serving_candidate(candidate_dir)
        if candidate_manifest.get("draft_revision") != draft_revision:
            raise RuntimeError("Accepted online EAGLE candidate revision does not match its manifest")
        if candidate_manifest.get("weights_sha256") != source_weights_sha256:
            raise RuntimeError("Accepted online EAGLE candidate weight digest does not match its manifest")
        candidate_tensors = load_file(Path(candidate_dir) / candidate_manifest["weights_path"])
        manifest = TensorTransferManifest.from_tensors(
            transfer_id=transfer_id,
            revision=draft_revision,
            source_weights_sha256=source_weights_sha256,
            tensors=candidate_tensors,
        )
        previous = self._accepted_draft_dir
        self._accepted_draft_dir = candidate_dir
        self._accepted_draft_revision = draft_revision
        self._accepted_candidate_tensors = candidate_tensors
        self._accepted_transfer_manifest = manifest.to_mapping()
        previous_path = Path(previous).resolve()
        process_root = self._process_root.resolve()
        if previous_path != Path(self._initial_draft_dir).resolve() and previous_path.is_relative_to(process_root):
            remove_online_eagle_scratch(previous_path)
        return self._accepted_transfer_manifest

    def status(self) -> dict[str, Any]:
        return {
            "accepted_draft_dir": self._accepted_draft_dir,
            "accepted_draft_revision": self._accepted_draft_revision,
            "node_id": self._node_id,
            "gpu_ids": self._gpu_ids,
            "accepted_transfer_ready": self._accepted_candidate_tensors is not None,
            "transfer_group_initialized": self._draft_transfer_group is not None,
        }


def create_draft_trainer(*, initial_draft_dir: str, initial_draft_revision: str, process_id: str):
    """Create the dedicated GPU actor without embedding it in an inference placement group."""
    actor = ray.remote(num_gpus=1, max_restarts=0)(DraftTrainer)
    return actor.options(runtime_env={"env_vars": {"TORCH_COMPILE_DISABLE": "1"}}).remote(
        initial_draft_dir=initial_draft_dir,
        initial_draft_revision=initial_draft_revision,
        process_id=process_id,
    )
