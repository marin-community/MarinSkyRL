import gzip
import json
from zipfile import ZipFile

from datasets import load_dataset
from omegaconf import OmegaConf

from infra.rl_data import pivot_swe
from infra.rl_data.sources import prepare_pivot_swe_row
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
from skyrl_gym.verification import RolloutEvidence


def _raw_row(trajectory_id):
    return {
        "trajectory_id": trajectory_id,
        "responses_create_params": {
            "input": [{"role": "user", "content": "run a command"}],
            "tools": [{"type": "function", "name": "execute_bash", "parameters": {"type": "object"}}],
            "parallel_tool_calls": True,
        },
        "expected_action": {"type": "function_call", "name": "execute_bash", "arguments": '{"command":"pwd"}'},
        "agent_ref": {"name": "single_step_tool_use_with_argument_comparison_swe"},
        "pass_rate": 0.375,
    }


def test_prepared_swe_pivot_uses_local_action_grader():
    row = prepare_pivot_swe_row(_raw_row(7), 3)
    env = NemotronUltraEnv(OmegaConf.create({}), extras={"extra_info": row["extra_info"]})
    assert env.init(row["prompt"])[1]["chat_completion_params"]["tools"][0]["name"] == "execute_bash"

    env.set_rollout_evidence(
        RolloutEvidence(metadata={"assistant_message": {"tool_calls": [{"function": {"name": "execute_bash", "arguments": '{"command":"pwd"}'}}]}})
    )
    assert env.step("")["reward"] == 1.0

    env.set_rollout_evidence(
        RolloutEvidence(metadata={"assistant_message": {"tool_calls": [{"function": {"name": "other", "arguments": '{"command":"pwd"}'}}]}})
    )
    assert env.step("")["reward"] == 0.0


def test_smoke_sample_splits_trajectory_ids_and_writes_parquet(tmp_path, monkeypatch):
    rows = [_raw_row(1), _raw_row(1), *(_raw_row(index) for index in range(2, 258))]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [0, 1, 2]}

    monkeypatch.setattr(pivot_swe, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(pivot_swe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    manifest = pivot_swe.prepare_smoke_sample(tmp_path)

    probe_ids = manifest["probe_trajectory_ids"]
    assert len(set(manifest["train_trajectory_ids"] + probe_ids)) == 192
    assert probe_ids[0] == 1
    assert probe_ids[-1] >= 254
    assert max(right - left for left, right in zip(probe_ids, probe_ids[1:])) <= 3
    assert len(load_dataset("parquet", data_files=str(tmp_path / "train.parquet"), split="train")) == 64
    assert len(load_dataset("parquet", data_files=str(tmp_path / "probe.parquet"), split="train")) == 128


def test_report_pairs_probe_results_and_persists_training_errors(tmp_path):
    root = tmp_path / "run"
    export_root = root / "exports"
    diagnostics_root = root / "diagnostics"
    for step, score in ((0, 0.0), (8, 1.0)):
        eval_dir = export_root / "dumped_evals" / f"global_step_{step}_evals"
        eval_dir.mkdir(parents=True)
        (eval_dir / "nemotron_swe_pivot.jsonl").write_text(
            json.dumps(
                {
                    "output_response": f"action-{step}",
                    "score": score,
                    "exception_type": None,
                    "env_extras": {"extra_info": {"nemotron_ultra": {"record_json": '{"trajectory_id":7}'}}},
                }
            ) + "\n"
        )
    retention_root = root / "trajectories"
    for step in (1, 8):
        archive_dir = retention_root / "schema_v3" / "archives" / "phase=train" / f"step={step:08d}"
        archive_dir.mkdir(parents=True)
        with ZipFile(archive_dir / "batch.zip", "w") as archive:
            archive.writestr(
                "records/failed.json.gz",
                gzip.compress(
                    b'{"trajectory":{"instance_id":"pivot-7"},"reward":{"outcome":0},"disposition":{"exception_type":"TimeoutError"}}'
                ),
            )

    summary = pivot_swe.write_smoke_report(
        str(export_root),
        str(diagnostics_root),
        str(retention_root),
        {"probe_trajectory_ids": [7]},
        final_step=8,
        training_completed=True,
    )
    comparison = json.loads((diagnostics_root / "comparison.jsonl").read_text())
    training = [json.loads(line) for line in (diagnostics_root / "training.jsonl").read_text().splitlines()]

    assert summary["before_mean_reward"] == 0.0
    assert summary["after_mean_reward"] == 1.0
    assert summary["mixed_reward_groups"] == 0
    assert summary["training_responses_per_step"] == {1: 1, 8: 1}
    assert comparison["before"]["output_response"] == "action-0"
    assert comparison["after"]["output_response"] == "action-8"
    assert len(training) == 2
    assert all(row["disposition"]["exception_type"] == "TimeoutError" for row in training)
