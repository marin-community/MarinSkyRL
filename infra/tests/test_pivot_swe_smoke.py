import asyncio
import gzip
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from datasets import Dataset, load_dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from ci.pivot_grug_smoke import compare_arms, launch_config, validate_smoke_config
from cloud.iris.launch_config import load_launch_config
from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.pivot_token_budget import select_token_budget_groups
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.pivot_reference import PivotReferenceRunner
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryID

from infra.rl_data import pivot_swe
from infra.rl_data.sources import prepare_pivot_row
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
        "metadata": {"instance_id": f"task-{trajectory_id}"},
    }


def test_prepared_swe_pivot_uses_local_action_grader():
    row = prepare_pivot_row(_raw_row(7), 3, dataset="swe")
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
    same_task = _raw_row(999)
    same_task["metadata"]["instance_id"] = "task-2"
    rows.insert(3, same_task)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [0, 1, 2]}

    monkeypatch.setattr(pivot_swe, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(pivot_swe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    manifest = pivot_swe.prepare_smoke_sample(
        tmp_path, tokenizer_name="Qwen/Qwen3-0.6B", tokenizer_revision="c1899de",
        chat_template_kwargs={"enable_thinking": False},
    )

    probe_ids = manifest["probe_trajectory_ids"]
    assert len(set(manifest["train_trajectory_ids"] + probe_ids)) == 192
    assert 999 not in manifest["train_trajectory_ids"] + probe_ids
    assert probe_ids[0] == 1
    assert probe_ids[-1] >= 254
    assert max(right - left for left, right in zip(probe_ids, probe_ids[1:])) <= 3
    assert min(manifest["train_trajectory_ids"]) < 20
    assert max(manifest["train_trajectory_ids"]) > 240
    assert len(load_dataset("parquet", data_files=str(tmp_path / "train.parquet"), split="train")) == 64
    probe_rows = load_dataset("parquet", data_files=str(tmp_path / "probe.parquet"), split="train")
    assert len(probe_rows) == 128
    assert json.loads(probe_rows[0]["extra_info"]["nemotron_ultra"]["request_json"])["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_initial_qwen_rewards_select_only_mixed_pivots_and_keep_predictions(tmp_path):
    candidate_path = tmp_path / "train.parquet"
    Dataset.from_list([prepare_pivot_row(_raw_row(index), index, dataset="swe") for index in (1, 2, 3)]).to_parquet(
        str(candidate_path)
    )
    evaluation_path = tmp_path / "eval.jsonl"
    with evaluation_path.open("w") as destination:
        for index, rewards in ((1, (1, 0)), (2, (0, 0)), (3, (0, 1))):
            extra_info = prepare_pivot_row(_raw_row(index), index, dataset="swe")["extra_info"]
            for reward in rewards:
                destination.write(
                    json.dumps(
                        {
                            "env_extras": {"extra_info": extra_info},
                            "score": [0, reward],
                            "output_response": f"sample {index} reward {reward}",
                            "stop_reason": "stop",
                            "exception_type": None,
                            "error_treatment": None,
                        }
                    ) + "\n"
                )

    output_root = tmp_path / "initial_policy"
    summary = pivot_swe.build_qwen_pivot_dataset(
        candidate_path, str(evaluation_path), str(output_root), rollouts_per_candidate=2, train_prefixes=2
    )

    assert summary["qwen_mixed_prefixes"] == 2
    assert summary["qwen_mean_reward"] == 1 / 3
    assert summary["nvidia_mean_reward"] == 0.375
    rollouts = [json.loads(line) for line in (output_root / "initial_policy_rollouts.jsonl").read_text().splitlines()]
    assert len(rollouts) == 6
    assert rollouts[0]["output_response"] == "sample 1 reward 1"
    assert rollouts[0]["expected_action"]["name"] == "execute_bash"
    stats = [json.loads(line) for line in (output_root / "initial_policy_pivots.jsonl").read_text().splitlines()]
    assert [row["selected_for_training"] for row in stats] == [True, False, True]
    assert stats[0]["qwen_reward_variance"] == 0.25
    selected = Dataset.from_parquet(str(output_root / "qwen_d_pivot.parquet"))
    assert {
        json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])["trajectory_id"] for row in selected
    } == {1, 3}


def test_report_pairs_probe_results_and_persists_training_errors(tmp_path):
    root = tmp_path / "run"
    export_root = root / "exports"
    diagnostics_root = root / "diagnostics"
    extra_info = {
        "index": 7,
        "nemotron_ultra": {
            "record_json": json.dumps(
                {
                    "trajectory_id": 7,
                    "profile_pass_rate": 0.375,
                    "expected_action": {"type": "function_call", "name": "execute_bash", "arguments": '{"command":"pwd"}'},
                }
            ),
            "request_json": json.dumps({"tools": [{"name": "execute_bash"}]}),
        },
    }
    tool_call = '<tool_call>\n{"name":"execute_bash","arguments":{"command":"pwd"}}\n</tool_call>'
    responses = {0: "action-0", 4: "<tool_call>{oops</tool_call>", 8: tool_call}
    for step, score in ((0, 0.0), (4, 0.0), (8, 1.0)):
        eval_dir = export_root / "dumped_evals" / f"global_step_{step}_evals"
        eval_dir.mkdir(parents=True)
        (eval_dir / "nemotron_swe_pivot.jsonl").write_text(
            json.dumps(
                {
                    "output_response": responses[step],
                    "score": [0.0, score],
                    "stop_reason": "tool_calls" if step == 8 else "stop",
                    "exception_type": None,
                    "env_extras": {"extra_info": extra_info},
                }
            ) + "\n"
        )
    retention_root = root / "trajectories"
    for step in (1, 8):
        archive_dir = retention_root / "schema_v3" / "archives" / "phase=train" / f"step={step:08d}"
        archive_dir.mkdir(parents=True)
        with ZipFile(archive_dir / "batch.zip", "w") as archive:
            record = {
                "global_step": step,
                "record_id": f"record-{step}",
                "trajectory": {
                    "instance_id": "pivot-7",
                    "repetition_id": 0,
                    "environment_extras": {"extra_info": extra_info},
                },
                "prompt": {"messages": [{"role": "user", "content": "run a command"}], "token_ids": [1, 2]},
                "response": {
                    "text": tool_call if step == 8 else "answer",
                    "token_ids": [3, 4],
                    "stop_reason": "tool_calls" if step == 8 else "stop",
                },
                "reward": {"outcome": float(step == 8)},
                "disposition": {"exception_type": "TimeoutError" if step == 1 else None},
            }
            archive.writestr(
                "records/failed.json.gz",
                gzip.compress(json.dumps(record).encode()),
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
    action_rows = [json.loads(line) for line in (diagnostics_root / "action_metrics.jsonl").read_text().splitlines()]

    assert summary["before_mean_reward"] == 0.0
    assert summary["after_mean_reward"] == 1.0
    assert summary["mixed_reward_groups"] == 0
    assert summary["training_responses_per_step"] == {1: 1, 8: 1}
    assert summary["training_input_tokens"] == 8
    assert summary["training_response_tokens"] == 4
    assert comparison["before"]["output_response"] == "action-0"
    assert comparison["after"]["output_response"] == tool_call
    assert len(training) == 2
    assert next(row for row in training if row["global_step"] == 1)["disposition"]["exception_type"] == "TimeoutError"
    assert summary["action_metrics_uri"] == str(diagnostics_root / "action_metrics.jsonl")
    assert {(row["phase"], row["step"]) for row in action_rows} == {
        ("train", 1),
        ("train", 8),
        ("eval", 0),
        ("eval", 4),
        ("eval", 8),
    }
    assert all(not row["group_mixed"] for row in action_rows if row["phase"] == "train")
    consumed_report = pivot_swe.write_smoke_report(
        str(export_root), str(root / "consumed"), str(retention_root), {"probe_trajectory_ids": [7]},
        final_step=8, training_completed=True, consumed_trajectories={(8, "pivot-7", 0)},
    )
    assert consumed_report["training_input_tokens"] == 4
    assert consumed_report["training_responses_per_step"] == {8: 1}
    after = next(row for row in action_rows if row["phase"] == "eval" and row["step"] == 8)
    assert after["delta_vs_baseline"] == 1.0
    assert after["rendered_tool_matches_reference"] is True
    assert after["rendered_reward_category"] == "EXPECTED_TOOL_CALL"
    assert after["profile_pass_rate"] == 0.375
    middle = next(row for row in action_rows if row["phase"] == "eval" and row["step"] == 4)
    assert middle["rendered_tool_json_valid"] is False


def test_smoke_config_prepares_pivot_data_instead_of_inherited_gsm8k():
    config_dir = Path(__file__).resolve().parents[2] / "skyrl-train" / "skyrl_train" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="pivot_swe_smoke")
    assert list(cfg.data.train_data) == []
    assert list(cfg.data.val_data) == []


def test_reference_sft_supervises_action_tokens_and_generates_held_out_answers():
    backend = Tokenizer(
        WordLevel(
            {"UNK": 0, "USER": 1, "run": 2, "ASSISTANT": 3, "execute_bash": 4, "pwd": 5, "END": 6}, unk_token="UNK"
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="UNK", eos_token="END")
    tokenizer.chat_template = """{% for m in messages %}{% if m.role == 'user' %}USER run {% else %}ASSISTANT {{ m.tool_calls[0].function.name }} {{ m.tool_calls[0].function.arguments.command }} END{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT {% endif %}"""

    class EvaluationRunner(TrajectoryRunner):
        async def _run(self, input_batch, disable_tqdm=False):
            return {"prompt_token_ids": [[1, 2, 3]], "response_ids": [[0, 6]], "loss_masks": [[1, 1]], "rewards": [0.0]}

    row = prepare_pivot_row(_raw_row(7), 0, dataset="swe")
    config = OmegaConf.create({"max_input_length": 100, "sampling_params": {"max_generate_length": 16}})
    runner = PivotReferenceRunner(EvaluationRunner(), tokenizer, config)
    request = {
        "prompts": [row["prompt"]],
        "env_extras": [{"extra_info": row["extra_info"]}],
        "env_classes": ["nemotron_ultra"],
        "trajectory_ids": [TrajectoryID("7", 0)],
        "batch_metadata": BatchMetadata(1, "train"),
        "sampling_params": {},
    }
    training = asyncio.run(runner.run(request))
    assert training["prompt_token_ids"] == [[1, 2, 3]]
    assert training["response_ids"] == [[4, 5, 6]]
    assert training["loss_masks"] == [[1, 1, 1]]
    evaluation = asyncio.run(runner.run({**request, "batch_metadata": BatchMetadata(1, "eval")}))
    assert evaluation["response_ids"] == [[0, 6]]
    assert evaluation["rewards"] == [0.0]


def test_comparison_pairs_rl_and_sft_scores_on_identical_probes(tmp_path):
    for arm, scores in (("rl", [(0, 1), (1, 0)]), ("sft", [(0, 0), (1, 1)])):
        path = tmp_path / arm / "diagnostics" / "comparison.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "trajectory_id": index,
                "before": {"score": [before], "exception_type": None},
                "after": {"score": [after], "exception_type": None},
            }
            for index, (before, after) in enumerate(scores)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    manifest = {"probe_trajectory_ids": [0, 1], "revision": "test", "train_prefixes": 64, "learner_token_budget": 1000}
    arms = {
        "rl": {"training_input_tokens": 999, "training_response_tokens": 99, "training_responses_per_step": {1: 4, 2: 2}},
        "sft": {"training_input_tokens": 1000, "training_response_tokens": 80, "training_responses_per_step": {1: 4, 2: 4}},
    }
    summary = compare_arms(str(tmp_path), manifest, arms)
    assert summary["token_match_relative_error"] == 0.001
    assert summary["training_input_tokens"] == {"rl": 999, "sft": 1000}
    assert summary["rl_after"] == summary["sft_after"] == 0.5
    assert summary["rl_wins"] == summary["sft_wins"] == 1
    paired = [json.loads(line) for line in (tmp_path / "diagnostics/comparison.jsonl").read_text().splitlines()]
    assert [(row["trajectory_id"], row["rl_after"], row["sft_after"]) for row in paired] == [(0, 1, 0), (1, 0, 1)]


def test_token_budget_keeps_whole_groups_and_preserves_their_training_fields():
    # Interleaved groups: the long group does not fit, but two shorter groups do.
    uids = ["a", "b", "a", "b", "c", "c"]
    lengths = [5, 30, 6, 30, 3, 4]
    batch = {
        "prompt_token_ids": [[1, 2]] * 6,
        "response_ids": [[3] * (length - 2) for length in lengths],
        "rewards": [0, 1, 1, 0, 0, 1],
        "loss_masks": [[1] * (length - 2) for length in lengths],
        "trajectory_ids": [TrajectoryID(uid, repetition) for uid, repetition in zip(uids, [0, 0, 1, 1, 0, 1])],
    }
    selection = select_token_budget_groups(lengths, uids, remaining_tokens=20, group_size=2)
    selected = filter_trajectory_batch(batch, selection.indices)
    assert selection.indices == [0, 2, 4, 5]
    assert selection.input_tokens == 18
    assert selected["rewards"] == [0, 1, 0, 1]
    assert [len(tokens) for tokens in selected["response_ids"]] == [3, 4, 1, 2]
    assert [(item.instance_id, item.repetition_id) for item in selected["trajectory_ids"]] == [("a", 0), ("a", 1), ("c", 0), ("c", 1)]

    tensors = convert_prompts_responses_to_batch_tensors(
        SimpleNamespace(pad_token_id=0),
        selected["prompt_token_ids"], selected["response_ids"],
        [[0.0] * len(tokens) for tokens in selected["response_ids"]], selected["loss_masks"],
    )
    # Tensor padding adds positions but must not inflate the admitted learner tokens.
    assert tensors[0].numel() == 24
    assert int(tensors[1].sum()) == 18
    assert int(tensors[2].sum()) == 10


@pytest.mark.parametrize("arm", ["rl", "sft"])
def test_paired_smoke_preflight_rejects_entrypoint_without_iris_runner(monkeypatch, tmp_path, arm):
    config = launch_config(
        "paired-preflight", "/tmp/pivot-output", "/tmp/pivot-temporary",
        "s3://models/grug", "test-revision", steps=10, arm=arm,
    )
    path = tmp_path / f"launch-{arm}.yaml"
    OmegaConf.save(config, path)
    resolved = load_launch_config(path)
    module = importlib.import_module(str(resolved.runtime.entrypoint))
    monkeypatch.delattr(module, "run")

    with pytest.raises(ValueError, match=r"must expose run\(cfg\) for the Iris training driver"):
        validate_smoke_config(resolved)
