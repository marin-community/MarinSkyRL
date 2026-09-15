"""Run the pinned Open-MOPD final-checkpoint evaluation inside one Iris task."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from cloud.iris.open_mopd_evaluation import (
    GATES,
    SMOKE_MAX_TOKENS,
    EvaluationBenchmark,
    EvaluationConfig,
    benchmark_coverage,
    evaluation_scale,
    load_evaluation_config,
)
from cloud.iris.open_mopd_fidelity import DatasetArtifact, FidelityConfig, gpu_count, load_config, validate_output_uri
from cloud.iris.open_mopd_fidelity_task import (
    ArtifactVerification,
    FileVerification,
    checkout_source,
    periodic_sync,
    reject_existing_output,
    runtime_inventory,
    snapshot_model,
    sync_tree,
    validate_runtime,
)

@dataclass(frozen=True)
class StagedBenchmark:
    benchmark: EvaluationBenchmark
    path: Path
    verification: ArtifactVerification


@dataclass(frozen=True)
class EvaluationInputs:
    source: Path
    model: Path
    benchmarks: tuple[StagedBenchmark, ...]
    model_verification: ArtifactVerification


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def verify_benchmark_file(benchmark: EvaluationBenchmark, path: Path) -> FileVerification:
    """Verify one downloaded benchmark against its pinned LFS object."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evaluation data for {benchmark.name} is missing or not a regular file")
    observed_size = path.stat().st_size
    if observed_size != benchmark.size:
        raise ValueError(
            f"Evaluation data integrity mismatch for {benchmark.name}: "
            f"expected {benchmark.size} bytes, found {observed_size}"
        )
    with path.open("rb") as source:
        observed_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if observed_sha256 != benchmark.sha256:
        raise ValueError(
            f"Evaluation data integrity mismatch for {benchmark.name}: "
            f"expected SHA-256 {benchmark.sha256}, found {observed_sha256}"
        )
    return FileVerification(
        path=benchmark.path,
        expected_size=benchmark.size,
        observed_size=observed_size,
        expected_sha256=benchmark.sha256,
        observed_sha256=observed_sha256,
    )


def _download_benchmark(
    config: EvaluationConfig, benchmark: EvaluationBenchmark, destination: Path
) -> StagedBenchmark:
    artifact = DatasetArtifact(
        repository=config.data.repository,
        revision=config.data.revision,
        path=benchmark.path,
        size=benchmark.size,
        sha256=benchmark.sha256,
    )
    code = (
        "from huggingface_hub import hf_hub_download; import sys; "
        "print(hf_hub_download(sys.argv[1], sys.argv[3], repo_type='dataset', revision=sys.argv[2], "
        "local_dir=sys.argv[4]))"
    )
    _run([sys.executable, "-c", code, artifact.repository, artifact.revision, artifact.path, str(destination)])
    path = destination / artifact.path
    file_verification = verify_benchmark_file(benchmark, path)
    verification = ArtifactVerification(
        repository=artifact.repository,
        revision=artifact.revision,
        files=(file_verification,),
    )
    return StagedBenchmark(benchmark=benchmark, path=path, verification=verification)


def stage_evaluation_inputs(
    evaluation: EvaluationConfig, fidelity: FidelityConfig, root: Path
) -> EvaluationInputs:
    source = checkout_source(fidelity, root / "source")
    model, model_verification = snapshot_model(
        fidelity.evaluation_reference.repository,
        fidelity.evaluation_reference.revision,
        fidelity.evaluation_reference.lfs_files,
        root / "model",
    )
    benchmarks = tuple(
        _download_benchmark(evaluation, benchmark, root / "data") for benchmark in evaluation.benchmarks
    )
    return EvaluationInputs(
        source=source,
        model=model,
        benchmarks=benchmarks,
        model_verification=model_verification,
    )


def rollout_commands(
    config: EvaluationConfig,
    inputs: EvaluationInputs,
    gate: str,
    output: Path,
    *,
    world_size: int,
) -> tuple[tuple[str, ...], ...]:
    if gate not in GATES:
        raise ValueError(f"Unknown evaluation gate: {gate}")
    commands = []
    for domain in ("math", "code", "if"):
        selected = [item for item in inputs.benchmarks if item.benchmark.domain == domain]
        protocol = selected[0].benchmark
        samples = 1 if gate == "smoke" else protocol.samples
        max_tokens = min(protocol.max_tokens, SMOKE_MAX_TOKENS) if gate == "smoke" else protocol.max_tokens
        command = [
            sys.executable,
            "-m",
            "evals.rollout_engine.vllm_rollout",
            "--model",
            str(inputs.model),
            "--input",
            *(str(item.path) for item in selected),
            "--output-dir",
            str(output / "rollouts"),
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            str(world_size),
            "--max-model-len",
            str(config.protocol.max_model_len),
            "--dtype",
            config.protocol.dtype,
            "--gpu-memory-utilization",
            str(protocol.gpu_memory_utilization),
            "--max-num-seqs",
            str(config.protocol.max_num_seqs),
            "--temperature",
            str(protocol.temperature),
            "--top-p",
            str(config.protocol.top_p),
            "--top-k",
            str(config.protocol.top_k),
            "--n",
            str(samples),
            "--max-tokens",
            str(max_tokens),
            "--stop-token-ids",
            ",".join(str(token_id) for token_id in config.protocol.stop_token_ids),
            "--trust-remote-code",
        ]
        if protocol.enable_thinking is not None:
            command.extend(["--enable-thinking", str(protocol.enable_thinking).lower()])
        if gate == "smoke":
            command.extend(["--base", "0", "--offset", "1"])
        commands.append(tuple(command))
    return tuple(commands)


def score_released_benchmarks(config: EvaluationConfig, inputs: EvaluationInputs, output: Path) -> None:
    rollout_dir = output / "rollouts"
    data_dir = inputs.benchmarks[0].path.parents[2]
    for benchmark in config.benchmarks:
        if benchmark.score_mode != "released":
            continue
        rollouts = sorted(rollout_dir.glob(f"{benchmark.name}_rollouts_*.parquet"))
        if not rollouts:
            raise ValueError(f"No rollout files found for released scorer {benchmark.name}")
        command = [
            sys.executable,
            "-m",
            "evals.verifier.score",
            "--rollout",
            *(str(path) for path in rollouts),
            "--output",
            str(output / "scores" / f"{benchmark.name}.json"),
            "--data-dir",
            str(data_dir),
        ]
        _run(command, cwd=inputs.source)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fidelity-config", type=Path, required=True)
    parser.add_argument("--gate", choices=GATES, required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--gpu-slice", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--launcher-commit", required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/open-mopd-final-eval"))
    parser.add_argument("--sync-interval", type=int, default=300)
    return parser


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    evaluation = load_evaluation_config(args.config)
    fidelity = load_config(args.fidelity_config)
    validate_output_uri(args.output_uri)
    validate_runtime(fidelity)
    world_size = gpu_count(args.gpu_slice)
    reject_existing_output(args.output_uri, "evaluation-manifest.json")
    if args.work_root.exists():
        raise ValueError(f"Work root already exists: {args.work_root}")
    output = args.work_root / "output"
    output.mkdir(parents=True)
    manifest_path = output / "evaluation-manifest.json"
    planned_completions, maximum_output_tokens = evaluation_scale(evaluation, args.gate)
    manifest: dict[str, object] = {
        "status": "staging",
        "gate": args.gate,
        "evaluation_config": asdict(evaluation),
        "fidelity_config": asdict(fidelity),
        "coverage": [asdict(item) for item in benchmark_coverage(evaluation, args.gate)],
        "planned_completions": planned_completions,
        "maximum_output_tokens": maximum_output_tokens,
        "task_image": args.task_image,
        "launcher_commit": args.launcher_commit,
        "gpu_slice": args.gpu_slice,
    }
    _write_manifest(manifest_path, manifest)
    sync_tree(output, args.output_uri)
    stop = threading.Event()
    uploader = threading.Thread(
        target=periodic_sync,
        args=(output, args.output_uri, stop, args.sync_interval),
        daemon=True,
        name="open-mopd-evaluation-output-sync",
    )
    uploader.start()
    try:
        inputs = stage_evaluation_inputs(evaluation, fidelity, args.work_root)
        commands = rollout_commands(evaluation, inputs, args.gate, output, world_size=world_size)
        manifest.update(
            {
                "status": "running",
                "runtime": runtime_inventory(inputs.source),
                "model_verification": asdict(inputs.model_verification),
                "data_verifications": [asdict(item.verification) for item in inputs.benchmarks],
                "rollout_commands": commands,
            }
        )
        _write_manifest(manifest_path, manifest)
        sync_tree(output, args.output_uri)
        for command in commands:
            _run(list(command), cwd=inputs.source)
        score_released_benchmarks(evaluation, inputs, output)
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_tree(output, args.output_uri)


if __name__ == "__main__":
    sys.exit(main())
