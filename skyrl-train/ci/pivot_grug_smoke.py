"""Prepare pinned SWE inputs and run the existing split64 SkyRL launcher."""

from __future__ import annotations

import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import tempfile

from datasets import Dataset
from transformers import AutoTokenizer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rigging.filesystem.storage_path import StoragePath

from cloud.iris.hf_model_cache import ensure_hugging_face_model_cache
from cloud.iris.launch import LaunchState, execute_launch
from cloud.iris.launch_config import compose_launch_config, load_launch_config
from cloud.iris.runtime_bundle import resolve_launcher_source
from infra.rl_data.pivot_swe import prepare_smoke_sample, reference_action_tokens, write_smoke_report
from skyrl_train.pivot_token_budget import TOKEN_MATCH_TOLERANCE
from skyrl_train.utils.utils import validate_cfg

MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "b8c07f7df1df65525abbfdbcd1572318ba11c42f"
CLUSTER = "cw-us-east-02a"
RECIPE = Path("cloud/iris/configs/grug_pivot_swe_smoke.yaml")


def launch_config(
    run_id: str,
    output_root: str,
    temporary_root: str,
    model_uri: str,
    model_identity: str,
    *,
    steps: int = 10,
    arm: str = "rl",
    data_root: str | None = None,
) -> DictConfig:
    """Bind the smoke's immutable inputs to the standard launch schema."""
    data_root = data_root or f"{output_root}/data"
    recipe = OmegaConf.load(RECIPE)
    recipe.entrypoint = "pivot_rl"
    # Token counts stop each arm; twice the nominal updates bounds a stalled budget.
    recipe.trainer.max_steps = 2 * steps
    recipe.trainer.epochs = 2 * steps
    recipe.trainer.eval_interval = 2 * steps
    recipe.trainer.ckpt_interval = 2 * steps
    if arm == "sft":
        recipe.entrypoint = "pivot_sft"
        recipe.trainer.algorithm.policy_loss_type = "sft"
        recipe.trainer.algorithm.use_kl_loss = False
    elif arm != "rl":
        raise ValueError(f"Unknown experiment arm: {arm}")
    source = {
        "kind": "directory",
        "uri": data_root,
        "identity": data_root,
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
                "wandb_entity": "marin-community",
                "max_retries": 0,
                "timeout": 28800,
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
            "skyrl": OmegaConf.to_container(recipe, resolve=True),
        }
    )


def validate_smoke_config(config: DictConfig) -> None:
    """Check the pinned Grug recipe on CPU before preparing data or allocating GPUs."""
    skyrl = config.skyrl
    for role in ("policy", "ref"):
        if skyrl.trainer[role].megatron_config.context_parallel_size != 1:
            raise ValueError(
                f"Grug Pivot smoke requires trainer.{role}.megatron_config.context_parallel_size=1: "
                "Transformer Engine 2.11 p2p CP does not support Grug sliding-window attention; "
                "all_gather CP rejects packed sequences, and a2a CP2 cannot split Grug's five KV heads."
            )
    validate_cfg(skyrl)
    module_name = str(config.runtime.entrypoint)
    if not callable(getattr(importlib.import_module(module_name), "run", None)):
        raise ValueError(f"{module_name} must expose run(cfg) for the Iris training driver")


def compare_arms(output_root: str, manifest: dict, arm_summaries: dict) -> dict:
    """Pair RL and SFT predictions on the same held-out trajectories."""
    target_tokens = manifest["learner_token_budget"]
    for arm, summary in arm_summaries.items():
        remaining = target_tokens - summary["training_input_tokens"]
        if not 0 <= remaining <= TOKEN_MATCH_TOLERANCE * target_tokens:
            raise ValueError(f"{arm} did not match the common learner token budget")
    rows_by_arm = {}
    for arm in ("rl", "sft"):
        path = StoragePath(f"{output_root}/{arm}/diagnostics/comparison.jsonl")
        rows_by_arm[arm] = {
            row["trajectory_id"]: row for line in path.read_text().splitlines() if (row := json.loads(line))
        }
        if set(rows_by_arm[arm]) != set(manifest["probe_trajectory_ids"]):
            raise ValueError(f"{arm} evaluation did not cover the common probe set")
    paired = []
    for trajectory_id in manifest["probe_trajectory_ids"]:
        row = {"trajectory_id": trajectory_id}
        for arm in ("rl", "sft"):
            for phase in ("before", "after"):
                prediction = rows_by_arm[arm][trajectory_id][phase]
                if prediction is None or prediction["exception_type"] is not None:
                    raise ValueError(f"Incomplete {arm}/{phase} evaluation for trajectory {trajectory_id}")
                row[f"{arm}_{phase}"] = sum(prediction["score"])
        paired.append(row)
    count = len(paired)
    summary = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_revision": manifest["revision"],
        "train_prefixes": manifest["train_prefixes"],
        "probe_count": count,
        "updates_per_arm": {arm: len(summary["training_responses_per_step"]) for arm, summary in arm_summaries.items()},
        "learner_token_budget": target_tokens,
        "training_input_tokens": {arm: summary["training_input_tokens"] for arm, summary in arm_summaries.items()},
        "training_response_tokens": {
            arm: summary["training_response_tokens"] for arm, summary in arm_summaries.items()
        },
        "token_match_relative_error": abs(
            arm_summaries["rl"]["training_input_tokens"] - arm_summaries["sft"]["training_input_tokens"]
        )
        / target_tokens,
        "budget_matching": "nonpadding prompt + completion tokens, within 0.1%; generation and scoring compute are separate",
        **{
            key: sum(row[key] for row in paired) / count for key in ("rl_before", "rl_after", "sft_before", "sft_after")
        },
        "rl_wins": sum(row["rl_after"] > row["sft_after"] for row in paired),
        "sft_wins": sum(row["sft_after"] > row["rl_after"] for row in paired),
    }
    StoragePath(f"{output_root}/diagnostics/comparison.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in paired)
    )
    StoragePath(f"{output_root}/diagnostics/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--temporary-root", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-prefixes", type=int, default=512)
    parser.add_argument("--compare-sft", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    arms = ("rl", "sft") if args.compare_sft else ("rl",)
    with tempfile.TemporaryDirectory(prefix="pivot-grug-") as directory:
        workdir = Path(directory)
        configs = {}
        for arm in arms:
            output = f"{args.output_root}/{arm}" if args.compare_sft else args.output_root
            temporary = f"{args.temporary_root}/{arm}" if args.compare_sft else args.temporary_root
            config = launch_config(
                f"{args.run_id}-{arm}" if args.compare_sft else args.run_id,
                output,
                temporary,
                "s3://marin-us-east-02a/preflight/grug",
                MODEL_REVISION,
                steps=args.steps,
                arm=arm,
                data_root=f"{args.output_root}/data",
            )
            config_path = workdir / f"launch-{arm}.yaml"
            OmegaConf.save(config, config_path)
            resolved = load_launch_config(config_path)
            validate_smoke_config(resolved)
            if args.train_prefixes != int(resolved.skyrl.trainer.train_batch_size):
                raise ValueError("The paired smoke requires one full pass through the prefix set per update")
            logger.info(
                "Preflight {}: {} GPUs, {} prefixes x {} responses, {} safety-limit steps, runtime {}",
                arm,
                resolved.iris.allocation.num_nodes * resolved.iris.allocation.gpus_per_node,
                resolved.skyrl.trainer.train_batch_size,
                resolved.skyrl.generator.n_samples_per_prompt,
                resolved.skyrl.trainer.max_steps,
                resolved.runtime.launcher_commit,
            )
            configs[arm] = (config, config_path, output, temporary)
            if not args.run:
                print(OmegaConf.to_yaml(resolved))
        if not args.run:
            return
        for output in (args.output_root, args.temporary_root):
            if StoragePath(output).exists():
                raise FileExistsError(f"Output already exists; choose a fresh run: {output}")
        sample_dir = workdir / "data"
        manifest = prepare_smoke_sample(
            sample_dir,
            tokenizer_name=MODEL,
            tokenizer_revision=MODEL_REVISION,
            chat_template_kwargs={"enable_thinking": False},
            candidate_prefixes=args.train_prefixes,
            max_source_rows=8192,
            stop_when_ready=True,
            max_reference_tokens=512,
            max_prompt_tokens=int(resolved.skyrl.trainer.max_prompt_length),
        )
        StoragePath(f"{args.output_root}/data").upload_from(f"{sample_dir}/", recursive=True)
        model_uri, model_manifest = ensure_hugging_face_model_cache(
            MODEL,
            MODEL_REVISION,
            ttl_days=14,
            source_prefix=args.output_root,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        reference_tokens = 0
        for row in Dataset.from_parquet(str(sample_dir / "train.parquet")):
            prefix, completion = reference_action_tokens(row["prompt"], row["extra_info"], tokenizer)
            reference_tokens += len(prefix) + len(completion)
        samples = int(configs["rl"][0].skyrl.generator.n_samples_per_prompt)
        manifest["learner_token_budget"] = reference_tokens * samples * args.steps
        manifest["nominal_updates"] = args.steps
        manifest["sequences_per_full_update"] = manifest["train_prefixes"] * samples
        StoragePath(f"{args.output_root}/diagnostics/budget.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for config, config_path, _, _ in configs.values():
            config.inputs.model.uri = model_uri
            config.inputs.model.identity = model_manifest.identity
            config.skyrl.trainer.pivot_token_budget = manifest["learner_token_budget"]
            OmegaConf.save(config, config_path)
            validate_smoke_config(load_launch_config(config_path))
        logger.info(
            "Shared learner budget: {} tokens; launching {} GPU jobs", manifest["learner_token_budget"], len(configs)
        )
        arm_summaries = {}
        # Each launch supervises its own Iris child and installs signal handlers in
        # its process. Threads cannot run that launcher lifecycle.
        with ProcessPoolExecutor(max_workers=len(configs), mp_context=get_context("spawn")) as pool:
            launches = {
                pool.submit(execute_launch, config_path): arm for arm, (_, config_path, _, _) in configs.items()
            }
            for future in as_completed(launches):
                arm = launches[future]
                result = future.result()
                config, _, output, temporary = configs[arm]
                logger.info("{} launch result: {}", arm, json.dumps(asdict(result), sort_keys=True))
                if result.state != LaunchState.SUCCEEDED:
                    raise RuntimeError(f"Grug Pivot {arm} failed: {result.failure}")
                ledger = json.loads(StoragePath(f"{output}/exports/learner-token-budget.json").read_text())
                final_step = ledger["steps"][-1]["step"]
                consumed = {
                    (step["step"], str(instance), repetition)
                    for step in ledger["steps"]
                    for instance, repetition in step["trajectory_ids"]
                }
                summary = write_smoke_report(
                    f"{output}/exports",
                    f"{output}/diagnostics",
                    f"{temporary}/attempts/trajectories",
                    manifest,
                    final_step=final_step,
                    training_completed=True,
                    consumed_trajectories=consumed,
                )
                logger.info("{} report: {}", arm, json.dumps(summary, sort_keys=True))
                expected_counts = {step["step"]: len(step["trajectory_ids"]) for step in ledger["steps"]}
                if (
                    summary["training_responses_per_step"] != expected_counts
                    or summary["training_input_tokens"] != ledger["input_tokens"]
                ):
                    raise RuntimeError(f"{arm} retained records do not match the completed learner updates")
                remaining = ledger["target_tokens"] - ledger["input_tokens"]
                if (
                    ledger["target_tokens"] != manifest["learner_token_budget"]
                    or not 0 <= remaining <= TOKEN_MATCH_TOLERANCE * ledger["target_tokens"]
                ):
                    raise RuntimeError(f"{arm} did not reach the shared token budget")
                if arm == "rl" and summary["mixed_reward_groups"] == 0:
                    raise RuntimeError("RL run had no mixed-reward groups")
                if (summary["before_probe_count"], summary["after_probe_count"]) != (manifest["probe_prefixes"],) * 2:
                    raise RuntimeError(f"{arm} did not evaluate all held-out probes before and after training")
                arm_summaries[arm] = summary
        if args.compare_sft:
            logger.info(
                "RL/SFT comparison: {}",
                json.dumps(compare_arms(args.output_root, manifest, arm_summaries), sort_keys=True),
            )


if __name__ == "__main__":
    main()
