"""Source split and student-policy pivot selection behavior."""

import json
import hashlib
from pathlib import Path

import pytest
from datasets import Dataset

from infra.rl_data.pivot import adapt_row, filter_candidates, heldout_trajectories
from infra.rl_data.pivot_publish import publish_artifacts
from infra.rl_data.pivot_report import summarize, summarize_exposure, check_rollout_geometry


def swe_row(index):
    return {
        "trajectory_id": index // 2,
        "responses_create_params": {"input": [{"role": "user", "content": "fix it"}],
                                    "tools": [{"type": "function", "name": "edit", "parameters": {"type": "object"}}],
                                    "max_output_tokens": 512, "temperature": 0.7},
        "agent_ref": {"name": "single_step_tool_use_with_argument_comparison_swe"},
        "expected_action": {"type": "function_call", "name": "edit", "arguments": "{}"},
        # Deliberately contradict our policy's outcomes to catch accidental reuse.
        "pass_rate_passed": 8, "pass_rate_total": 8, "pass_rate": 1.0,
    }


def retained(row, repetition, reward):
    return {"phase": "eval", "global_step": 0,
            "trajectory": {"environment_extras": {"extra_info": row["extra_info"]}, "repetition_id": repetition},
            "provenance": {"model_path": "student@immutable", "model_version_step": 0, "model_source_identity": "revision"},
            "disposition": {"exception_type": None},
            "verification_result": {"status": "verified", "score": reward},
            "reward": {"outcome": reward, "shaped": reward}}


def test_adapter_preserves_request_and_source_and_splits_whole_trajectories():
    counts = {str(i): 2 for i in range(300)}
    holdout = heldout_trajectories(counts, 256, 42)
    assert holdout == heldout_trajectories(dict(reversed(list(counts.items()))), 256, 42)
    train, validation = set(), set()
    for i in range(600):
        source = swe_row(i)
        row = adapt_row(source, "swe", i, "validation" if str(i // 2) in holdout else "train")
        extra = row["extra_info"]
        (validation if extra["split"] == "validation" else train).add(extra["trajectory_id"])
        options = json.loads(extra["nemotron_ultra"]["request_json"])
        assert options == {k: v for k, v in source["responses_create_params"].items() if k != "input"}
        assert row["prompt"] == source["responses_create_params"]["input"]
        assert json.loads(extra["nemotron_ultra"]["record_json"])["expected_action"] == source["expected_action"]
    assert not train & validation
    assert sum(counts[key] for key in validation) >= 256


def test_filter_persists_all_student_statistics_and_only_mixed_difficult_rows(tmp_path):
    rows = [adapt_row(swe_row(i), "swe", i, "train") for i in range(4)]
    Dataset.from_list(rows).to_parquet(str(tmp_path / "candidates.parquet"))
    (tmp_path / "manifest.json").write_text('{"release_rows": 4}')
    outcomes = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    records = [retained(row, rep, outcome) for row, group in zip(rows, outcomes)
               for rep, outcome in enumerate(group)]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest = filter_candidates(tmp_path, tmp_path / "filtered", difficulty_threshold=.5,
                                 rollouts=path, profile_policy="student@immutable", profile_revision="revision", samples_per_prefix=4)
    selected = Dataset.from_parquet(str(tmp_path / "filtered/train.parquet"))
    profiles = Dataset.from_parquet(str(tmp_path / "filtered/profiled_candidates.parquet"))
    random_rows = Dataset.from_parquet(str(tmp_path / "filtered/random_train.parquet"))
    assert [row["extra_info"]["index"] for row in selected] == [1]
    assert [row["student_profile"]["mean"] for row in profiles] == [0, .25, .5, 1]
    assert [row["student_profile"]["variance"] for row in profiles] == [0, .1875, .25, 0]
    assert manifest["selected_rows"] + manifest["rejected_rows"] == 4
    # Seed 42 draws the all-fail candidate, which the pivot arm excluded.
    assert [row["extra_info"]["index"] for row in random_rows] == [0]
    assert len(random_rows) == len(selected)
    assert manifest["random_control"] == {
        "seed": 42, "sampling": "uniform_without_replacement", "eligible_rows": 4,
        "selected_rows": 1, "overlap_with_pivots": 0,
    }
    # Incomplete coverage must never silently produce a biased filtered dataset.
    path.write_text("".join(json.dumps(record) + "\n" for record in records[:-1]))
    with pytest.raises(ValueError, match="Incomplete profiling"):
        filter_candidates(tmp_path, tmp_path / "partial", difficulty_threshold=.5,
                          rollouts=path, profile_policy="student@immutable", profile_revision="revision", samples_per_prefix=4)


def test_report_clusters_correlated_prefixes_and_checks_rollout_geometry():
    rows = [adapt_row(swe_row(i), "swe", i, "validation") for i in range(4)]
    records = [retained(row, 0, int(i >= 2)) for i, row in enumerate(rows)]
    report = summarize(records, bootstrap_samples=1000)["swe"]
    assert report["accuracy"] == .5
    assert report["trajectories"] == 2
    assert report["ci95"] == [0, 1]
    training = [{**record, "phase": "train", "global_step": 1} for record in records]
    assert check_rollout_geometry(training, 4, 1)["trajectories"] == 4
    with pytest.raises(ValueError, match="geometry"):
        check_rollout_geometry(training, 64, 16)


def test_publish_is_content_addressed_and_rejects_changed_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.parquet").write_bytes(b"test artifact")
    digest = hashlib.sha256(b"test artifact").hexdigest()
    manifest = {"dataset": "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1", "revision": "pinned",
                "artifacts": {"train": {"sha256": digest}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    uri = publish_artifacts(source, str(tmp_path / "store"))
    assert (Path(uri) / "train.parquet").read_bytes() == b"test artifact"
    assert json.loads((Path(uri) / "manifest.json").read_text()) == manifest
    assert publish_artifacts(source, str(tmp_path / "store")) == uri
    # A profiled pilot has two ancestors: the candidate slice and full release.
    pilot_profile = {"parent": {"parent": manifest, "purpose": "pilot"},
                     "profile_policy": "student", "profile_revision": "student-sha",
                     "artifacts": manifest["artifacts"]}
    (source / "manifest.json").write_text(json.dumps(pilot_profile))
    pilot_uri = publish_artifacts(source, str(tmp_path / "store"))
    assert "/profiles/swe/student/student-sha/" in pilot_uri
    assert json.loads((Path(pilot_uri) / "manifest.json").read_text()) == pilot_profile
    (source / "train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        publish_artifacts(source, str(tmp_path / "store"))


def test_exposure_distinguishes_unique_rows_visits_and_response_multiplicity():
    row = adapt_row(swe_row(0), "swe", 0, "train")
    records = []
    for step in (1, 2):
        for repetition in range(16):
            record = retained(row, repetition, 1)
            record.update(phase="train", global_step=step, prompt={"token_ids": [1, 2, 3]},
                          response={"token_ids": [4, 5]})
            record["trajectory"]["instance_id"] = f"visit-{step}"
            records.append(record)
    result = summarize_exposure(records)
    assert result["unique_source_rows"] == 1
    assert result["prefix_visits"] == 2
    assert result["responses"] == 32
    assert result["prompt_tokens_per_response_total"] == 96
    assert result["response_tokens_total"] == 64
    assert result["responses_per_source_row"] == {row["extra_info"]["source_id"]: 32}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_exposure([*records, records[0]])
