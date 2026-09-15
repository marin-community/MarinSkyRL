"""Run the pinned authors' Open-MOPD control inside one Iris GPU task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import posixpath
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from cloud.iris.artifacts import fs_and_path
from cloud.iris.open_mopd_fidelity import (
    DOMAINS,
    GATES,
    FidelityConfig,
    LfsFile,
    gpu_count,
    load_config,
    validate_output_uri,
)

CONTROL_MANIFEST_NAME = "control-manifest.json"


@dataclass(frozen=True)
class FileVerification:
    path: str
    expected_size: int
    observed_size: int
    expected_sha256: str
    observed_sha256: str


@dataclass(frozen=True)
class ArtifactVerification:
    repository: str
    revision: str
    files: tuple[FileVerification, ...]


@dataclass(frozen=True)
class StagedInputs:
    source: Path
    student: Path
    teachers: tuple[Path, ...]
    dataset: Path
    artifact_verifications: tuple[ArtifactVerification, ...] = ()


def _run(
    command: list[str], *, cwd: Path | None = None, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture_output)


def validate_runtime(config: FidelityConfig) -> None:
    expected_python = config.environment.python
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise ValueError(f"Python version mismatch: expected {expected_python}, found {actual_python}")
    for distribution, expected in config.environment.packages.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Required distribution is missing from the task image: {distribution}") from error
        if actual != expected:
            raise ValueError(f"{distribution} version mismatch: expected {expected}, found {actual}")


def checkout_source(config: FidelityConfig, destination: Path) -> Path:
    source = destination / "Open-MOPD"
    source.mkdir(parents=True)
    _run(["git", "init", "-q"], cwd=source)
    _run(["git", "remote", "add", "origin", config.source.repository], cwd=source)
    _run(["git", "fetch", "--depth=1", "origin", config.source.commit], cwd=source)
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=source)
    actual = _run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True).stdout.strip()
    if actual != config.source.commit:
        raise ValueError(f"Open-MOPD checkout mismatch: expected {config.source.commit}, found {actual}")
    return source


def verify_lfs_files(destination: Path, expected_files: tuple[LfsFile, ...]) -> tuple[FileVerification, ...]:
    """Verify downloaded model files against the pinned Hugging Face LFS manifest."""
    root = destination.resolve()
    failures = []
    verified = []
    for expected in expected_files:
        candidate = root / expected.path
        path = candidate.resolve()
        if not path.is_relative_to(root):
            failures.append(f"{expected.path}: resolves outside the model directory")
            continue
        if candidate.is_symlink() or not path.is_file():
            failures.append(f"{expected.path}: missing or not a regular file")
            continue
        observed_size = path.stat().st_size
        if observed_size != expected.size:
            failures.append(f"{expected.path}: expected {expected.size} bytes, found {observed_size}")
            continue
        with path.open("rb") as source:
            observed_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if observed_sha256 != expected.sha256:
            failures.append(f"{expected.path}: expected SHA-256 {expected.sha256}, found {observed_sha256}")
            continue
        verified.append(
            FileVerification(
                path=expected.path,
                expected_size=expected.size,
                observed_size=observed_size,
                expected_sha256=expected.sha256,
                observed_sha256=observed_sha256,
            )
        )
    if failures:
        raise ValueError("Model artifact integrity verification failed: " + "; ".join(failures))
    return tuple(verified)


def snapshot_model(
    repository: str, revision: str, expected_files: tuple[LfsFile, ...], destination: Path
) -> tuple[Path, ArtifactVerification]:
    code = (
        "from huggingface_hub import snapshot_download; import sys; "
        "print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))"
    )
    _run([sys.executable, "-c", code, repository, revision, str(destination)])
    files = verify_lfs_files(destination, expected_files)
    return destination, ArtifactVerification(repository=repository, revision=revision, files=files)


def _dataset(config: FidelityConfig, destination: Path) -> tuple[Path, ArtifactVerification]:
    artifact = config.dataset
    code = (
        "from huggingface_hub import hf_hub_download; import sys; "
        "print(hf_hub_download(sys.argv[1], sys.argv[3], repo_type='dataset', revision=sys.argv[2], "
        "local_dir=sys.argv[4]))"
    )
    _run(
        [
            sys.executable,
            "-c",
            code,
            artifact.repository,
            artifact.revision,
            artifact.path,
            str(destination),
        ]
    )
    path = destination / artifact.path
    observed_size = path.stat().st_size
    if observed_size != artifact.size:
        raise ValueError(f"Dataset size mismatch: expected {artifact.size} bytes, found {observed_size}")
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != artifact.sha256:
        raise ValueError(f"Dataset digest mismatch: expected {artifact.sha256}, found {actual}")
    verification = FileVerification(
        path=artifact.path,
        expected_size=artifact.size,
        observed_size=observed_size,
        expected_sha256=artifact.sha256,
        observed_sha256=actual,
    )
    return path, ArtifactVerification(
        repository=artifact.repository,
        revision=artifact.revision,
        files=(verification,),
    )


def stage_inputs(config: FidelityConfig, root: Path) -> StagedInputs:
    source = checkout_source(config, root / "source")
    student, student_verification = snapshot_model(
        config.student.repository,
        config.student.revision,
        config.student.lfs_files,
        root / "models" / "student",
    )
    teacher_snapshots = tuple(
        snapshot_model(
            teacher.repository,
            teacher.revision,
            teacher.lfs_files,
            root / "models" / teacher.domain,
        )
        for teacher in config.teachers
    )
    dataset, dataset_verification = _dataset(config, root / "data")
    return StagedInputs(
        source=source,
        student=student,
        teachers=tuple(snapshot for snapshot, _ in teacher_snapshots),
        dataset=dataset,
        artifact_verifications=(
            student_verification,
            *(verification for _, verification in teacher_snapshots),
            dataset_verification,
        ),
    )


def training_command(
    config: FidelityConfig,
    inputs: StagedInputs,
    gate: str,
    output: Path,
    *,
    world_size: int,
) -> list[str]:
    training = config.training
    steps = GATES[gate]
    overrides = [
        "algorithm.adv_estimator=token_reward_direct",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={training.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={training.ppo_epochs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.optim.lr={training.learning_rate}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        f"actor_rollout_ref.actor.optim.clip_grad={training.gradient_clip}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.entropy_coeff={training.entropy_coefficient}",
        f"actor_rollout_ref.actor.loss_agg_mode={training.loss_aggregation}",
        f"actor_rollout_ref.actor.clip_ratio_low={training.clip_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={training.clip_high}",
        "actor_rollout_ref.actor.opd_refresh_advantage=True",
        "actor_rollout_ref.actor.opd_reward_weight_mode=student_p",
        f"actor_rollout_ref.rollout.log_prob_top_k={training.top_k}",
        "actor_rollout_ref.rollout.top_k_strategy=only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode=student_p",
        f"actor_rollout_ref.rollout.temperature={training.temperature}",
        f"actor_rollout_ref.rollout.top_p={training.nucleus_p}",
        f"reward_model.micro_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"reward_model.teacher_temperature={training.teacher_temperature}",
        "+data.sampler.class_path=verl.utils.dataset.domain_weighted_sampler",
        "+data.sampler.class_name=DomainWeightedSampler",
        *(f"+data.domain_weights.{domain}={weight}" for domain, weight in zip(DOMAINS, training.domain_weights)),
        "data.dataloader_num_workers=0",
        "+mt_opd.domain_weighting=domain_routing",
        "+mt_opd.target_share_domains=[math,code,if]",
        f"+mt_opd.target_share_values=[{','.join(str(value) for value in training.target_gradient_shares)}]",
        f"+mt_opd.normalize_reward_scale={training.gap_following_alpha}",
        "+mt_opd.reward_scale_stat=mean",
        "+mt_opd.reward_scale_direction=multiply",
        "+mt_opd.reward_scale_anchored=True",
        "+mt_opd.conflict_policy=none",
        f"trainer.total_training_steps={steps}",
        f"trainer.save_freq={min(training.save_every, steps)}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.logger=['console']",
        "trainer.resume_mode=disable",
    ]
    command = [
        "bash",
        "scripts/local/mt_opd.sh",
        "--run",
        "--model",
        str(inputs.student),
        "--teacher",
        str(inputs.teachers[0]),
        "--teacher",
        str(inputs.teachers[1]),
        "--teacher",
        str(inputs.teachers[2]),
        "--domains",
        ",".join(DOMAINS),
        "--train",
        str(inputs.dataset),
        "--val",
        str(inputs.dataset),
        "--output",
        str(output),
        "--checkpoint",
        str(output / "checkpoints"),
        "--gpus",
        str(world_size),
    ]
    training_env = {
        "TRAIN_BATCH_SIZE": str(training.train_batch_size),
        "MAX_PROMPT_LENGTH": str(training.prompt_limit),
        "MAX_RESPONSE_LENGTH": str(training.response_limit),
        "N_RESPONSES": "1",
        "TOTAL_EPOCHS": "1",
    }
    for override in overrides:
        command.extend(["--extra", override])
    return ["env", *(f"{key}={value}" for key, value in training_env.items()), *command]


def sync_tree(local: Path, output_uri: str) -> None:
    filesystem, target = fs_and_path(output_uri)
    for source in local.rglob("*"):
        if not source.is_file():
            continue
        destination = posixpath.join(target, source.relative_to(local).as_posix())
        filesystem.makedirs(posixpath.dirname(destination), exist_ok=True)
        filesystem.put_file(str(source), destination)


def reject_existing_output(output_uri: str, manifest_name: str = CONTROL_MANIFEST_NAME) -> None:
    filesystem, target = fs_and_path(output_uri)
    if filesystem.exists(posixpath.join(target, manifest_name)):
        raise ValueError(f"Durable output already contains {manifest_name}: {output_uri}")


def periodic_sync(local: Path, output_uri: str, stop: threading.Event, interval: int) -> None:
    while not stop.wait(interval):
        sync_tree(local, output_uri)


def runtime_inventory(source: Path) -> dict[str, object]:
    return {
        "open_mopd_git_status": _run(["git", "status", "--porcelain"], cwd=source, capture_output=True).stdout,
        "python": sys.version,
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"], capture_output=True).stdout.splitlines(),
        "nvidia_smi": _run(["nvidia-smi", "-q"], capture_output=True).stdout,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", choices=tuple(GATES), required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--gpu-slice", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--launcher-commit", required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/open-mopd-fidelity"))
    parser.add_argument("--sync-interval", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    config = load_config(args.config)
    validate_output_uri(args.output_uri)
    validate_runtime(config)
    world_size = gpu_count(args.gpu_slice)
    reject_existing_output(args.output_uri)
    if args.work_root.exists():
        raise ValueError(f"Work root already exists: {args.work_root}")
    args.work_root.mkdir(parents=True)
    output = args.work_root / "output"
    output.mkdir()
    inputs = stage_inputs(config, args.work_root)
    command = training_command(config, inputs, args.gate, output, world_size=world_size)
    manifest = {
        "config": asdict(config),
        "gate": args.gate,
        "steps": GATES[args.gate],
        "command": command,
        "runtime": runtime_inventory(inputs.source),
        "artifact_verifications": [asdict(verification) for verification in inputs.artifact_verifications],
        "task_image": args.task_image,
        "launcher_commit": args.launcher_commit,
        "gpu_slice": args.gpu_slice,
    }
    manifest_path = output / CONTROL_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sync_tree(output, args.output_uri)
    stop = threading.Event()
    uploader = threading.Thread(
        target=periodic_sync,
        args=(output, args.output_uri, stop, args.sync_interval),
        daemon=True,
        name="open-mopd-output-sync",
    )
    uploader.start()
    try:
        result = subprocess.run(command, cwd=inputs.source, check=False)
        manifest["returncode"] = result.returncode
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_tree(output, args.output_uri)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
