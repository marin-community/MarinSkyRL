"""Prepare a bounded, trajectory-disjoint smoke sample of NVIDIA SWE pivots."""

from __future__ import annotations

import gzip
import json
import random
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import fsspec
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from infra.rl_data.sources import prepare_pivot_swe_row
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action

DATASET_ID = "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1"
DATASET_REVISION = "4947a3c8ea803413a65f9eca14a96ef521b2ddf5"
TRAIN_PREFIXES = 64
PROBE_PREFIXES = 128
MAX_CANDIDATES = 1024
INITIAL_POLICY_CANDIDATES = 1536
INITIAL_POLICY_ROLLOUTS = 8
PIVOT_TRAIN_PREFIXES = 128
MAX_PROMPT_TOKENS = 15_872
TOKENIZER_REVISION = "c1899de"


def prepare_smoke_sample(
    output_dir: Path,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    chat_template_kwargs: dict[str, Any] | None = None,
    *,
    candidate_prefixes: int = TRAIN_PREFIXES,
    source_path: Path | None = None,
    max_source_rows: int = MAX_CANDIDATES,
    stop_when_ready: bool = False,
) -> dict[str, Any]:
    """Write candidate pivots and trajectory-ID-stratified held-out probes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=TOKENIZER_REVISION)
    template_kwargs = chat_template_kwargs or {}
    dataset = (
        (json.loads(line) for line in source_path.open())
        if source_path is not None
        else load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train", streaming=True)
    )
    eligible: list[tuple[int, dict[str, Any]]] = []
    trajectory_ids: set[int] = set()
    instance_ids: set[str] = set()

    for index, raw in enumerate(dataset):
        if index >= max_source_rows:
            break
        trajectory_id = raw["trajectory_id"]
        instance_id = raw["metadata"]["instance_id"]
        if trajectory_id in trajectory_ids or instance_id in instance_ids:
            continue
        if not 0.0 < raw["pass_rate"] < 1.0:
            continue
        request = {**raw["responses_create_params"], "chat_template_kwargs": template_kwargs}
        prepared = prepare_pivot_swe_row({**raw, "responses_create_params": request}, index)
        prompt_tokens = tokenizer.apply_chat_template(
            prepared["prompt"],
            tools=request.get("tools"),
            add_generation_prompt=True,
            tokenize=True,
            **template_kwargs,
        )
        if hasattr(prompt_tokens, "keys"):
            prompt_tokens = prompt_tokens["input_ids"]
        if len(prompt_tokens) > MAX_PROMPT_TOKENS:
            continue
        trajectory_ids.add(trajectory_id)
        instance_ids.add(instance_id)
        eligible.append((trajectory_id, prepared))
        if stop_when_ready and len(eligible) >= candidate_prefixes + PROBE_PREFIXES:
            break

    if len(eligible) < candidate_prefixes + PROBE_PREFIXES:
        raise RuntimeError(f"Only found {len(eligible)} eligible SWE pivots")
    sorted_ids = sorted(trajectory_id for trajectory_id, _ in eligible)
    probe_ids = {sorted_ids[index * len(sorted_ids) // PROBE_PREFIXES] for index in range(PROBE_PREFIXES)}
    probe = [prepared for trajectory_id, prepared in eligible if trajectory_id in probe_ids]
    train_candidates = sorted(
        ((trajectory_id, prepared) for trajectory_id, prepared in eligible if trajectory_id not in probe_ids),
        key=lambda pair: pair[0],
    )
    train_pairs = [
        train_candidates[index * len(train_candidates) // candidate_prefixes] for index in range(candidate_prefixes)
    ]
    random.Random(17).shuffle(train_pairs)
    train = [prepared for _, prepared in train_pairs]
    train_trajectory_ids = [trajectory_id for trajectory_id, _ in train_pairs]
    probe_trajectory_ids = sorted(probe_ids)
    Dataset.from_list(train).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(probe).to_parquet(str(output_dir / "probe.parquet"))
    manifest = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "train_trajectory_ids": train_trajectory_ids,
        "probe_trajectory_ids": probe_trajectory_ids,
        "train_prefixes": len(train),
        "probe_prefixes": len(probe),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_qwen_pivot_dataset(
    candidate_path: Path,
    evaluation_uri: str,
    output_root: str,
    *,
    rollouts_per_candidate: int = INITIAL_POLICY_ROLLOUTS,
    train_prefixes: int = PIVOT_TRAIN_PREFIXES,
) -> dict[str, Any]:
    """Retain Qwen mixed-reward prefixes and persist its initial predictions."""
    candidates = Dataset.from_parquet(str(candidate_path))
    by_id = {
        json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])["trajectory_id"]: row for row in candidates
    }
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rollouts_uri = f"{output_root}/initial_policy_rollouts.jsonl"
    with fsspec.open(evaluation_uri, "r") as source, fsspec.open(rollouts_uri, "w") as destination:
        for line in source:
            evaluation = json.loads(line)
            extra_info = evaluation["env_extras"]["extra_info"]
            record = json.loads(extra_info["nemotron_ultra"]["record_json"])
            trajectory_id = record["trajectory_id"]
            if trajectory_id not in by_id:
                raise ValueError(f"Evaluation includes unknown trajectory {trajectory_id}")
            reward = float(sum(evaluation["score"]))
            if reward not in (0.0, 1.0):
                raise ValueError(f"Nonbinary initial reward for trajectory {trajectory_id}: {reward}")
            rollout = {
                **_action_fields(extra_info, evaluation["output_response"] or "", evaluation["stop_reason"]),
                "rollout_index": len(groups[trajectory_id]),
                "reward": reward,
                "output_response": evaluation["output_response"],
                "expected_action": record["expected_action"],
                "exception_type": evaluation["exception_type"],
                "error_treatment": evaluation["error_treatment"],
            }
            groups[trajectory_id].append(rollout)
            destination.write(json.dumps(rollout, ensure_ascii=False) + "\n")

    if set(groups) != set(by_id):
        raise RuntimeError(f"Initial-policy evaluation covered {len(groups)} of {len(by_id)} candidate prefixes")

    pivot_stats = []
    for trajectory_id in sorted(groups):
        group = groups[trajectory_id]
        if len(group) != rollouts_per_candidate:
            raise RuntimeError(f"Trajectory {trajectory_id} has {len(group)} of {rollouts_per_candidate} rollouts")
        successes = sum(row["reward"] for row in group)
        qwen_mean = successes / rollouts_per_candidate
        record = json.loads(by_id[trajectory_id]["extra_info"]["nemotron_ultra"]["record_json"])
        nvidia_mean = record["profile_pass_rate"]
        pivot_stats.append(
            {
                "trajectory_id": trajectory_id,
                "source_index": by_id[trajectory_id]["extra_info"]["index"],
                "expected_tool": record["expected_action"].get("name"),
                "qwen_success_count": successes,
                "qwen_rollout_count": rollouts_per_candidate,
                "qwen_mean_reward": qwen_mean,
                "qwen_reward_variance": qwen_mean * (1 - qwen_mean),
                "nvidia_mean_reward": nvidia_mean,
                "nvidia_reward_variance": nvidia_mean * (1 - nvidia_mean),
                "nvidia_pass_rate_passed": record.get("profile_pass_rate_passed"),
                "nvidia_pass_rate_total": record.get("profile_pass_rate_total"),
                "qwen_mixed_reward": 0 < successes < rollouts_per_candidate,
            }
        )

    mixed = [row for row in pivot_stats if row["qwen_mixed_reward"]]
    selected_ids = {
        mixed[index * len(mixed) // train_prefixes]["trajectory_id"] for index in range(train_prefixes)
    } if len(mixed) >= train_prefixes else set()
    for row in pivot_stats:
        row["selected_for_training"] = row["trajectory_id"] in selected_ids
    stats_uri = f"{output_root}/initial_policy_pivots.jsonl"
    with fsspec.open(stats_uri, "w") as destination:
        for row in pivot_stats:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "candidate_prefixes": len(by_id),
        "qwen_mixed_prefixes": len(mixed),
        "selected_train_prefixes": len(selected_ids),
        "qwen_mean_reward": sum(row["qwen_mean_reward"] for row in pivot_stats) / len(pivot_stats),
        "nvidia_mean_reward": sum(row["nvidia_mean_reward"] for row in pivot_stats) / len(pivot_stats),
        "initial_policy_rollouts_uri": rollouts_uri,
        "initial_policy_pivots_uri": stats_uri,
        "raw_evaluation_uri": evaluation_uri,
    }
    with fsspec.open(f"{output_root}/summary.json", "w") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
    if len(mixed) < train_prefixes:
        raise RuntimeError(f"Only {len(mixed)} Qwen mixed-reward pivots; need {train_prefixes}")

    selected = [row for row in candidates if json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])["trajectory_id"] in selected_ids]
    random.Random(17).shuffle(selected)
    local_path = candidate_path.parent / "qwen_d_pivot.parquet"
    Dataset.from_list(selected).to_parquet(str(local_path))
    with local_path.open("rb") as source, fsspec.open(f"{output_root}/qwen_d_pivot.parquet", "wb") as destination:
        shutil.copyfileobj(source, destination)
    return summary


def _action_fields(extra_info: dict[str, Any], response: str, stop_reason: str | None) -> dict[str, Any]:
    pivot = extra_info["nemotron_ultra"]
    record = json.loads(pivot["record_json"])
    expected_tool = record["expected_action"].get("name")
    marker = "<tool_call>"
    start = response.find(marker)
    rendered_tool_name = None
    rendered_tool_json_valid = None
    rendered_action_match = None
    rendered_reward_category = None
    if start >= 0:
        end = response.find("</tool_call>", start + len(marker))
        rendered_tool_json_valid = False
        if end >= 0:
            try:
                rendered_call = json.loads(response[start + len(marker) : end])
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(rendered_call, dict):
                    rendered_tool_json_valid = True
                    rendered_tool_name = rendered_call.get("name")
                    arguments = rendered_call.get("arguments")
                    arguments_json = arguments if isinstance(arguments, str) else json.dumps(arguments)
                    rendered_reward, category = grade_expected_action(
                        record["expected_action"],
                        {"tool_calls": [{"function": {"name": rendered_tool_name, "arguments": arguments_json}}]},
                        word_count_similarity_threshold=0.0,
                    )
                    rendered_action_match = rendered_reward > 0
                    rendered_reward_category = category.name
    return {
        "trajectory_id": record["trajectory_id"],
        "source_index": extra_info["index"],
        "profile_pass_rate": record.get("profile_pass_rate"),
        "expected_tool": expected_tool,
        "available_tool_count": len(json.loads(pivot["request_json"]).get("tools") or []),
        "stop_reason": stop_reason,
        "tool_call_stop": stop_reason == "tool_calls",
        "truncated": stop_reason == "length",
        "rendered_tool_markup": start >= 0,
        "rendered_tool_json_valid": rendered_tool_json_valid,
        "rendered_tool_name": rendered_tool_name,
        "rendered_tool_matches_reference": None if rendered_tool_name is None else rendered_tool_name == expected_tool,
        "rendered_action_match": rendered_action_match,
        "rendered_reward_category": rendered_reward_category,
        "response_preview": response[:160],
    }


def _write_action_metrics(
    diagnostics_root: str,
    training_records: list[dict[str, Any]],
    evaluations: dict[int, dict[int, dict[str, Any]]],
) -> str:
    """Write one compact row per rollout or probe, retaining the raw files for inspection."""
    group_rewards: dict[tuple[int, str], list[float]] = defaultdict(list)
    for record in training_records:
        group_rewards[(record["global_step"], str(record["trajectory"]["instance_id"]))].append(
            float(record["reward"]["outcome"])
        )

    path = f"{diagnostics_root}/action_metrics.jsonl"
    with fsspec.open(path, "w") as file:
        for record in training_records:
            response = record["response"]
            trajectory = record["trajectory"]
            extra_info = trajectory["environment_extras"]["extra_info"]
            group_id = str(trajectory["instance_id"])
            rewards = group_rewards[(record["global_step"], group_id)]
            row = {
                "phase": "train",
                "step": record["global_step"],
                "record_id": record["record_id"],
                "group_id": group_id,
                "repetition_id": trajectory["repetition_id"],
                "group_size": len(rewards),
                "group_success_count": sum(rewards),
                "group_mixed": len(set(rewards)) > 1,
                "reward": float(record["reward"]["outcome"]),
                "response_tokens": len(response["token_ids"]),
                "prompt_tokens": len(record["prompt"]["token_ids"]),
                "prompt_message_count": len(record["prompt"]["messages"]),
                "exception_type": record["disposition"]["exception_type"],
                **_action_fields(extra_info, response["text"] or "", response["stop_reason"]),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

        baseline = evaluations.get(0, {})
        for step, probes in sorted(evaluations.items()):
            for trajectory_id, probe in probes.items():
                reward = sum(probe["score"])
                baseline_reward = sum(baseline[trajectory_id]["score"]) if trajectory_id in baseline else None
                row = {
                    "phase": "eval",
                    "step": step,
                    "reward": reward,
                    "baseline_reward": baseline_reward,
                    "delta_vs_baseline": None if baseline_reward is None else reward - baseline_reward,
                    "response_tokens": len(probe["score"]),
                    "exception_type": probe["exception_type"],
                    **_action_fields(
                        probe["env_extras"]["extra_info"], probe["output_response"] or "", probe["stop_reason"]
                    ),
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_smoke_report(
    export_root: str,
    diagnostics_root: str,
    retention_root: str,
    manifest: dict[str, Any],
    *,
    final_step: int,
    training_completed: bool,
) -> dict[str, Any]:
    """Persist paired probe predictions and all retained training actions."""
    data_source = "nemotron_swe_pivot.jsonl"
    eval_root = f"{export_root}/dumped_evals"

    def read_evaluation(step: int) -> dict[int, dict[str, Any]]:
        path = f"{eval_root}/global_step_{step}_evals/{data_source}"
        filesystem, key = fsspec.core.url_to_fs(path)
        if not filesystem.exists(key):
            return {}
        with filesystem.open(key, "r") as file:
            rows = [json.loads(line) for line in file]
        return {
            json.loads(row["env_extras"]["extra_info"]["nemotron_ultra"]["record_json"])["trajectory_id"]: row
            for row in rows
        }

    evaluations = {step: rows for step in range(final_step + 1) if (rows := read_evaluation(step))}
    before = evaluations.get(0, {})
    after = evaluations.get(final_step, {})
    paired = [
        {"trajectory_id": trajectory_id, "before": before.get(trajectory_id), "after": after.get(trajectory_id)}
        for trajectory_id in manifest["probe_trajectory_ids"]
    ]
    comparison_path = f"{diagnostics_root}/comparison.jsonl"
    with fsspec.open(comparison_path, "w") as file:
        for row in paired:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    filesystem, key = fsspec.core.url_to_fs(retention_root)
    training_records: list[dict[str, Any]] = []
    records_per_step: dict[int, int] = defaultdict(int)
    for archive_path in filesystem.glob(f"{key}/schema_v*/archives/phase=train/step=*/*.zip"):
        step = int(Path(archive_path).parent.name.removeprefix("step="))
        with filesystem.open(archive_path, "rb") as file:
            with ZipFile(BytesIO(file.read())) as archive:
                for name in archive.namelist():
                    if name.startswith("records/") and name.endswith(".json.gz"):
                        training_records.append(json.loads(gzip.decompress(archive.read(name))))
                        records_per_step[step] += 1
    with fsspec.open(f"{diagnostics_root}/training.jsonl", "w") as file:
        for row in training_records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    reward_groups: dict[tuple[int, str], set[float]] = defaultdict(set)
    for row in training_records:
        reward_groups[(row["global_step"], str(row["trajectory"]["instance_id"]))].add(
            float(row["reward"]["outcome"])
        )
    action_metrics_path = _write_action_metrics(diagnostics_root, training_records, evaluations)

    summary = {
        **manifest,
        "training_completed": training_completed,
        "before_probe_count": len(before),
        "after_probe_count": len(after),
        "training_response_count": len(training_records),
        "training_responses_per_step": dict(sorted(records_per_step.items())),
        "mixed_reward_groups": sum(len(rewards) > 1 for rewards in reward_groups.values()),
        "before_mean_reward": sum(sum(row["score"]) for row in before.values()) / len(before) if before else None,
        "after_mean_reward": sum(sum(row["score"]) for row in after.values()) / len(after) if after else None,
        "comparison_uri": comparison_path,
        "training_uri": f"{diagnostics_root}/training.jsonl",
        "action_metrics_uri": action_metrics_path,
        "raw_evaluation_root": eval_root,
    }
    with fsspec.open(f"{diagnostics_root}/summary.json", "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    return summary
