"""Run the released Open-MOPD schedule with MarinSkyRL's routed OPD learner.

The prompt schedule is staged locally after a digest check. Full optimizer
checkpoints are written directly to durable object storage every two steps;
unlike the ordinary Iris resume policy, this run must retain every step for
independent target evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as pq
from skyrl_train.io import io

from cloud.iris.artifacts import fs_and_path
from cloud.iris.open_mopd_fidelity import load_config
from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from marinskyrl.resource_locator import join_resource_path

ROOT = Path(__file__).resolve().parents[3]
FIDELITY_CONFIG = ROOT / "cloud/iris/configs/open_mopd_fidelity.json"
SCHEDULE_SHA256 = "01d9c3904b13324f218f61e51db187e52872c641a466025b300295eef707dae5"
SCHEDULE_ROWS = 204_800
SCHEDULE_STEPS = 200
PARTITIONED_POLICY_GPUS = 4
COLOCATED_POLICY_GPUS = 5
COLOCATED_GPU_MEMORY_UTILIZATION = 0.75
CHECKPOINT_INTERVAL = load_config(FIDELITY_CONFIG).training.save_every


class Placement(StrEnum):
    PARTITIONED = "partitioned"
    COLOCATED = "colocated"


def placement_arguments(placement: Placement) -> tuple[str, ...]:
    if placement is Placement.PARTITIONED:
        return ()
    if placement is Placement.COLOCATED:
        return (
            "trainer.placement.colocate_all=true",
            f"trainer.placement.policy_num_gpus_per_node={COLOCATED_POLICY_GPUS}",
            f"generator.num_inference_engines={COLOCATED_POLICY_GPUS}",
            f"generator.gpu_memory_utilization={COLOCATED_GPU_MEMORY_UTILIZATION}",
        )
    raise ValueError(f"Unsupported Open-MOPD placement: {placement}")


def _replace_overrides(base: list[str], replacements: tuple[str, ...]) -> tuple[str, ...]:
    """Append ``replacements``, dropping any base override for the same key so no key repeats."""
    replaced_keys = {override.split("=", 1)[0].lstrip("+") for override in replacements}
    kept = [override for override in base if override.split("=", 1)[0].lstrip("+") not in replaced_keys]
    return (*kept, *replacements)


def hydra_arguments(
    data_path: Path,
    validation_path: Path,
    checkpoint_uri: str,
    export_uri: str,
    *,
    resume: bool = False,
    placement: Placement = Placement.PARTITIONED,
) -> tuple[str, ...]:
    config = load_config(FIDELITY_CONFIG)
    training = config.training
    args = [
        f"data.train_data=['{data_path}']",
        f"data.val_data=['{validation_path}']",
        "data.shuffle=false",
        "trainer.algorithm.advantage_estimator=uniform",
        "trainer.algorithm.use_kl_loss=false",
        "trainer.algorithm.use_entropy_loss=false",
        "trainer.algorithm.loss_reduction=token_mean",
        "++trainer.algorithm.distillation.objective=student_topk_policy_surrogate",
        "++trainer.algorithm.distillation.routing_plan=opd",
        "++trainer.algorithm.distillation.coefficient=1.0",
        "++trainer.algorithm.distillation.reward_mode=replace",
        f"++trainer.algorithm.distillation.domain_gradient_balance.gap_scale_alpha={training.gap_following_alpha}",
        f"trainer.policy.model.path={config.student.repository}",
        f"trainer.policy.model.revision={config.student.revision}",
        f"trainer.policy.optimizer_config.lr={training.learning_rate}",
        "trainer.policy.optimizer_config.weight_decay=0",
        "trainer.policy.optimizer_config.scheduler=constant",
        f"trainer.policy.optimizer_config.max_grad_norm={training.gradient_clip}",
        "trainer.strategy=fsdp2",
        "trainer.use_sample_packing=false",
        "trainer.flash_attn=true",
        "trainer.placement.colocate_all=false",
        f"trainer.placement.policy_num_gpus_per_node={PARTITIONED_POLICY_GPUS}",
        "trainer.epochs=1",
        f"trainer.max_steps={SCHEDULE_STEPS}",
        f"trainer.train_batch_size={training.train_batch_size}",
        f"trainer.policy_mini_batch_size={training.mini_batch_size}",
        f"trainer.micro_train_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"trainer.micro_forward_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"trainer.update_epochs_per_batch={training.ppo_epochs}",
        f"trainer.max_prompt_length={training.prompt_limit}",
        "trainer.eval_before_train=false",
        f"trainer.eval_interval={CHECKPOINT_INTERVAL}",
        "trainer.eval_batch_size=1",
        "trainer.dump_eval_results=true",
        f"trainer.ckpt_interval={CHECKPOINT_INTERVAL}",
        f"trainer.hf_save_interval={CHECKPOINT_INTERVAL}",
        "trainer.max_ckpts_to_keep=-1",
        f"trainer.ckpt_path={checkpoint_uri}",
        f"trainer.export_path={export_uri}",
        f"trainer.resume_mode={'latest' if resume else 'null'}",
        "trainer.logger=console",
        "trainer.project_name=open_mopd_native_repro",
        "trainer.run_name=released_200step_schedule",
        f"trainer.algorithm.eps_clip_low={training.clip_low}",
        f"trainer.algorithm.eps_clip_high={training.clip_high}",
        "generator.backend=vllm",
        "generator.num_inference_engines=1",
        "generator.inference_engine_tensor_parallel_size=1",
        "generator.n_samples_per_prompt=1",
        f"generator.sampling_params.max_generate_length={training.response_limit}",
        f"generator.sampling_params.temperature={training.temperature}",
        f"generator.sampling_params.top_p={training.nucleus_p}",
        f"generator.sampling_params.logprobs={training.top_k}",
        "++generator.engine_init_kwargs.max_model_len=32768",
        f"generator.eval_sampling_params.max_generate_length={training.response_limit}",
        "generator.eval_sampling_params.temperature=1.0",
        "generator.eval_sampling_params.top_p=1.0",
        "generator.eval_sampling_params.top_k=-1",
        "generator.eval_n_samples_per_prompt=1",
        "generator.gpu_memory_utilization=0.75",
        "generator.max_num_batched_tokens=4096",
        "generator.max_num_seqs=64",
        "generator.run_engines_locally=true",
        "generator.weight_sync_backend=nccl",
        "generator.async_engine=true",
        "generator.batched=true",
        "environment.env_class=prompt_only",
        f"environment.skyrl_gym.aime.evaluation_token_budget={training.response_limit}",
        "environment.skyrl_gym.aime.strict_box_verify=true",
        f"environment.skyrl_gym.aime.max_gen_length={training.response_limit}",
        "trajectory_runner.process_pool.num_coordinators=1",
        "trajectory_runner.process_pool.cpus_per_coordinator=4",
    ]
    for teacher, target_share in zip(config.teachers, training.target_gradient_shares, strict=True):
        domain = teacher.domain
        prefix = f"++teachers.{domain}"
        args.extend(
            (
                f"++trainer.algorithm.distillation.domain_gradient_balance.target_shares.{domain}={target_share}",
                f"{prefix}.source=local_inference",
                f"{prefix}.placement=pinned",
                f"{prefix}.model.path={teacher.repository}",
                f"{prefix}.model.revision={teacher.revision}",
                f"{prefix}.backend=vllm",
                f"{prefix}.evidence=student_selected_topk",
                f"{prefix}.top_k={training.top_k}",
                f"{prefix}.resources.num_nodes=1",
                f"{prefix}.resources.gpus_per_node=1",
                f"{prefix}.resources.tensor_parallel_size=1",
                f"{prefix}.resources.colocation_group=teacher_{domain}",
                f"{prefix}.resources.max_num_batched_tokens=4096",
                f"{prefix}.resources.gpu_memory_utilization=0.7",
                f"++teacher_routing.opd.routes.{domain}.teacher={domain}",
                f"++teacher_routing.opd.routes.{domain}.weight=1.0",
            )
        )
    args.append("++teacher_routing.opd.revision=open-mopd-native-200step-v1")
    return _replace_overrides(args, placement_arguments(placement))


def stage_schedule(source_uri: str, destination: Path) -> None:
    metadata = _verified_parquet(source_uri, destination, SCHEDULE_SHA256)
    if metadata.num_rows != SCHEDULE_ROWS or metadata.num_row_groups != SCHEDULE_STEPS:
        raise ValueError(
            f"Unexpected Open-MOPD schedule geometry: {metadata.num_rows} rows, {metadata.num_row_groups} groups"
        )


def stage_validation(source_uri: str, destination: Path, expected_sha256: str) -> None:
    metadata = _verified_parquet(source_uri, destination, expected_sha256)
    columns = set(metadata.schema.to_arrow_schema().names)
    if metadata.num_rows != 30 or not {"prompt", "env_class", "reward_model"} <= columns:
        raise ValueError(f"AIME validation dataset has {metadata.num_rows} rows and columns {sorted(columns)}")


def _verified_parquet(source_uri: str, destination: Path, expected_sha256: str) -> pq.FileMetaData:
    io.download_file(source_uri, str(destination))
    with destination.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Parquet dataset digest mismatch for {source_uri}: {digest}")
    return pq.ParquetFile(destination).metadata


def run(
    dataset_uri: str,
    validation_uri: str,
    validation_sha256: str,
    checkpoint_uri: str,
    export_uri: str,
    manifest_uri: str,
    source_commit: str,
    *,
    resume: bool = False,
    placement: Placement = Placement.PARTITIONED,
) -> int:
    if any(not uri.startswith("s3://") or "/users/" not in uri for uri in (checkpoint_uri, export_uri, manifest_uri)):
        raise ValueError("Native artifacts require durable user-owned s3:// paths")
    filesystem, manifest_path = fs_and_path(manifest_uri)
    identity = {
        "source_commit": source_commit,
        "dataset_uri": dataset_uri,
        "dataset_sha256": SCHEDULE_SHA256,
        "validation_uri": validation_uri,
        "validation_sha256": validation_sha256,
        "checkpoint_uri": checkpoint_uri,
        "export_uri": export_uri,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "target_eval_interval": CHECKPOINT_INTERVAL,
        "placement": placement.value,
    }
    existing = filesystem.exists(manifest_path)
    if existing != resume:
        raise FileExistsError(
            f"{'Resume requires' if resume else 'Refusing to overwrite'} an existing Open-MOPD run: {manifest_uri}"
        )
    if resume:
        previous = json.loads(filesystem.cat_file(manifest_path))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Open-MOPD resume identity differs from the existing run")
        if previous.get("status") == "complete":
            raise ValueError("Cannot resume a completed Open-MOPD run")
        marker_fs, marker_path = fs_and_path(join_resource_path(checkpoint_uri, LATEST_CHECKPOINT_FILE))
        if not marker_fs.exists(marker_path):
            raise FileNotFoundError(f"Open-MOPD resume has no durable checkpoint marker: {checkpoint_uri}")
    with tempfile.TemporaryDirectory(prefix="open-mopd-native-") as directory:
        schedule = Path(directory) / "weighted_200step_schedule.parquet"
        validation = Path(directory) / "aime24.parquet"
        stage_schedule(dataset_uri, schedule)
        stage_validation(validation_uri, validation, validation_sha256)
        command = [
            sys.executable,
            "-m",
            "skyrl_train.entrypoints.main_base",
            *hydra_arguments(schedule, validation, checkpoint_uri, export_uri, resume=resume, placement=placement),
        ]
        manifest = {
            "schema_version": 1,
            "status": "running",
            **identity,
            "resumed": resume,
            "command": command,
        }
        filesystem.pipe_file(manifest_path, json.dumps(manifest, sort_keys=True).encode())
        result = subprocess.run(command, check=False)
        manifest.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        filesystem.pipe_file(manifest_path, json.dumps(manifest, sort_keys=True).encode())
        return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--validation-uri", required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--export-uri", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--placement", choices=Placement, type=Placement, default=Placement.PARTITIONED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "hydra_arguments": hydra_arguments(
                        Path("/data/schedule.parquet"),
                        Path("/data/aime24.parquet"),
                        args.checkpoint_uri,
                        args.export_uri,
                        resume=args.resume,
                        placement=args.placement,
                    )
                },
                indent=2,
            )
        )
        return 0
    return run(
        args.dataset_uri,
        args.validation_uri,
        args.validation_sha256,
        args.checkpoint_uri,
        args.export_uri,
        args.manifest_uri,
        args.source_commit,
        resume=args.resume,
        placement=args.placement,
    )


if __name__ == "__main__":
    sys.exit(main())
