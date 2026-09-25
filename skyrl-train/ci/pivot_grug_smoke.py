"""Prepare pinned SWE inputs and run the existing split64 SkyRL launcher."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import tempfile

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rigging.filesystem.storage_path import StoragePath

from cloud.iris.hf_model_cache import ensure_hugging_face_model_cache
from cloud.iris.launch import LaunchState, execute_launch
from cloud.iris.launch_config import compose_launch_config, load_launch_config
from cloud.iris.runtime_bundle import resolve_launcher_source
from infra.rl_data.pivot_swe import prepare_smoke_sample, write_smoke_report

MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "b8c07f7df1df65525abbfdbcd1572318ba11c42f"
CLUSTER = "cw-us-east-02a"
RECIPE = Path("cloud/iris/configs/grug_pivot_swe_smoke.yaml")


def launch_config(
    run_id: str, output_root: str, temporary_root: str, model_uri: str, model_identity: str
) -> DictConfig:
    """Bind the smoke's immutable inputs to the standard launch schema."""
    data_root = f"{output_root}/data"
    source = {
        "kind": "directory",
        "uri": data_root,
        "identity": f"{run_id}/data",
        "local_path": "/tmp/pivot-grug/data",
    }
    return compose_launch_config(
        {
            "schema_version": 1,
            "run": {"id": run_id, "attempt_id": run_id, "seed": 17, "export_hf": True},
            "runtime": {
                "launcher_commit": resolve_launcher_source().commit,
                "profile": "megatron",
            },
            "iris": {
                "cluster": CLUSTER,
                "cluster_config": str(files("iris") / "config" / f"{CLUSTER}.yaml"),
                "job_name": run_id,
                "max_retries": 0,
                "timeout": 14400,
                "allocation": {
                    "num_nodes": 8,
                    "gpus_per_node": 8,
                    "gpu_variant": "H100",
                    "cpu": 32,
                    "memory": "1800GB",
                    "disk": "1TB",
                },
            },
            "ray": {
                "spill_dir": "/tmp/skyrl-ray-spill",
                "rendezvous_dir": f"{temporary_root}/rendezvous",
                "log_dir": f"{temporary_root}/ray-logs",
            },
            "artifacts": {
                "checkpoint_root": f"{temporary_root}/checkpoints",
                "export_root": f"{output_root}/exports",
                "attempts_root": f"{temporary_root}/attempts",
                "resolved_config_uri": f"{output_root}/resolved-launch.yaml",
                "terminal_manifest_uri": f"{output_root}/terminal.json",
                "resume_checkpoint_count": 1,
            },
            "inputs": {
                "model": {
                    "uri": model_uri,
                    "identity": model_identity,
                    "local_path": "/tmp/pivot-grug/model",
                    "tokenizer_uri": MODEL,
                    "tokenizer_revision": MODEL_REVISION,
                },
                "data_kind": "parquet",
                "train_data": [{**source, "relative_path": "train.parquet"}],
                "validation_data": [{**source, "relative_path": "probe.parquet"}],
            },
            "skyrl": OmegaConf.to_container(OmegaConf.load(RECIPE), resolve=True),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--temporary-root", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pivot-grug-") as directory:
        workdir = Path(directory)
        config_path = workdir / "launch.yaml"
        config = launch_config(
            args.run_id,
            args.output_root,
            args.temporary_root,
            "s3://marin-us-east-02a/preflight/grug",
            MODEL_REVISION,
        )
        OmegaConf.save(config, config_path)
        resolved = load_launch_config(config_path)
        logger.info(
            "Preflight: {} GPUs, {} prefixes x {} responses, {} steps, runtime {}",
            resolved.iris.allocation.num_nodes * resolved.iris.allocation.gpus_per_node,
            resolved.skyrl.trainer.train_batch_size,
            resolved.skyrl.generator.n_samples_per_prompt,
            resolved.skyrl.trainer.max_steps,
            resolved.runtime.launcher_commit,
        )
        if not args.run:
            print(OmegaConf.to_yaml(resolved))
            return

        for output in (args.output_root, args.temporary_root):
            if StoragePath(output).exists():
                raise FileExistsError(f"Smoke output already exists; choose a fresh run: {output}")

        sample_dir = workdir / "data"
        manifest = prepare_smoke_sample(
            sample_dir,
            tokenizer_name=MODEL,
            tokenizer_revision=MODEL_REVISION,
            chat_template_kwargs={"enable_thinking": False},
        )
        StoragePath(f"{args.output_root}/data").upload_from(f"{sample_dir}/", recursive=True)
        model_uri, model_manifest = ensure_hugging_face_model_cache(
            MODEL,
            MODEL_REVISION,
            ttl_days=14,
            source_prefix=args.output_root,
        )
        config.inputs.model.uri = model_uri
        config.inputs.model.identity = model_manifest.identity
        OmegaConf.save(config, config_path)
        logger.info("Launching {} with model {}", args.run_id, model_uri)
        result = execute_launch(config_path)
        logger.info("Launch result: {}", json.dumps(asdict(result), sort_keys=True))
        if result.state != LaunchState.SUCCEEDED:
            raise RuntimeError(f"Grug Pivot smoke failed: {result.failure}")
        summary = write_smoke_report(
            f"{args.output_root}/exports",
            f"{args.output_root}/diagnostics",
            f"{args.temporary_root}/attempts/trajectories",
            manifest,
            final_step=int(resolved.skyrl.trainer.max_steps),
            training_completed=True,
        )
        logger.info("Smoke report: {}", json.dumps(summary, sort_keys=True))
        final_step = int(resolved.skyrl.trainer.max_steps)
        responses_per_step = int(resolved.skyrl.trainer.train_batch_size) * int(
            resolved.skyrl.generator.n_samples_per_prompt
        )
        expected_counts = {step: responses_per_step for step in range(1, final_step + 1)}
        if summary["training_responses_per_step"] != expected_counts or summary["mixed_reward_groups"] == 0:
            raise RuntimeError("Smoke did not retain every training response with mixed-reward groups")
        expected_probes = manifest["probe_prefixes"]
        if (summary["before_probe_count"], summary["after_probe_count"]) != (expected_probes, expected_probes):
            raise RuntimeError("Smoke did not retain all held-out probes before and after training")


if __name__ == "__main__":
    main()
