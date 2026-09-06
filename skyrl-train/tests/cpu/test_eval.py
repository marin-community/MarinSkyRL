"""
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_eval.py
"""

from unittest.mock import MagicMock
import json

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from omegaconf import OmegaConf

from skyrl_train.evaluate import evaluate
from skyrl_train.io import io
from skyrl_train.trajectory_runners.base import TrajectoryRunner, TrajectoryBatch
from tests.cpu.util import example_dummy_config


@pytest.fixture
def dummy_config():
    return example_dummy_config()


class DummyStatefulDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class DummyRunner(TrajectoryRunner):
    def __init__(self, output: TrajectoryBatch):
        self.output = output
        self.seen_inputs = []

    async def _run(self, input_batch, disable_tqdm: bool = False):
        self.seen_inputs.append(input_batch)
        return self.output


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["disabled", "local", "s3", "gs", "unwritable"])
@pytest.mark.parametrize("global_step", [None, 0, 5])
async def test_evaluate_computes_expected_metrics_and_persists_results(
    dummy_config, tmp_path, monkeypatch, storage, global_step
):
    cfg = dummy_config
    cfg.generator.backend = "vllm"
    cfg.generator.eval_sampling_params = OmegaConf.create(
        {
            "max_generate_length": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.generator.trajectory_retention.enabled = False
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.dump_eval_results = storage != "disabled"
    scheme = "s3" if storage == "unwritable" else storage
    cfg.trainer.export_path = (
        f"{scheme}://eval-bucket/{tmp_path.name}/exports" if scheme in ("s3", "gs") else str(tmp_path / "exports")
    )
    cloud = MemoryFileSystem()
    if scheme in ("s3", "gs"):
        monkeypatch.setattr(io, "_get_filesystem", lambda _path: cloud)
    if storage == "unwritable":

        def reject_write(*args, **kwargs):
            raise PermissionError("evaluation output denied")

        monkeypatch.setattr(cloud, "open", reject_write)
    monkeypatch.chdir(tmp_path)

    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "question-1"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        },
        {
            "prompt": [{"role": "user", "content": "question-2"}],
            "env_class": "custom_env",
            "env_extras": {"data_source": "dataset/b"},
            "uid": "uid-2",
        },
    ]
    eval_dataloader = DummyStatefulDataLoader([prompts_batch])

    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[101], [102]],
        "response_ids": [[201], [202]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    runner = DummyRunner(trajectory_batch)

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: f"decoded π {tokens[0]}"

    evaluation = evaluate(
        eval_dataloader=eval_dataloader,
        trajectory_runner=runner,
        cfg=cfg,
        global_step=global_step,
        tokenizer=tokenizer,
    )
    if storage == "unwritable":
        with pytest.raises(PermissionError, match="evaluation output denied"):
            await evaluation
        assert not cloud.exists(cfg.trainer.export_path)
        return
    metrics = await evaluation

    expected_metrics = {
        "eval/dataset_a/avg_score": 1.0,
        "eval/dataset_a/pass_at_1": 1.0,
        "eval/dataset_b/avg_score": 0.0,
        "eval/dataset_b/pass_at_1": 0.0,
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
    }

    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)

    assert len(runner.seen_inputs) == 1
    seen_batch = runner.seen_inputs[0]
    assert seen_batch["prompts"] == [prompt["prompt"] for prompt in prompts_batch]
    assert seen_batch["env_classes"] == ["gsm8k", "custom_env"]
    assert seen_batch["env_extras"] == [prompt["env_extras"] for prompt in prompts_batch]
    assert seen_batch["batch_metadata"].training_phase == "eval"

    if storage == "disabled":
        assert not (tmp_path / "exports").exists()
        return
    suffix = "eval_only" if global_step is None else f"global_step_{global_step}_evals"
    dump_root = f"{cfg.trainer.export_path}/dumped_evals/{suffix}"
    for i, dataset in enumerate(("a", "b")):
        with io.open_file(f"{dump_root}/dataset_{dataset}.jsonl", "r") as source:
            rows = [json.loads(line) for line in source]
        assert rows == [
            {
                "input_prompt": f"decoded π {101 + i}",
                "output_response": f"decoded π {201 + i}",
                "score": 1.0 - i,
                "stop_reason": "stop",
                "env_class": ["gsm8k", "custom_env"][i],
                "env_extras": {"data_source": f"dataset/{dataset}"},
                "data_source": f"dataset/{dataset}",
            }
        ]
    with io.open_file(f"{dump_root}/aggregated_results.jsonl", "r") as source:
        assert json.load(source) == expected_metrics
    if storage in ("s3", "gs"):
        assert len(cloud.find(cfg.trainer.export_path)) == 3
        assert not (tmp_path / f"{storage}:").exists()
