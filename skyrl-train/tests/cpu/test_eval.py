"""
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_eval.py
"""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from skyrl_gym.envs.registration import registry
from skyrl_gym.verification import VerificationResult

from skyrl_train.evaluate import _calculate_eval_metrics, evaluate
from skyrl_train.dataset.preference_pairs import PreferencePairFormat
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.preference_pairs import PreferencePairTrajectoryRunner
from skyrl_train.trajectory_runners.base import TrajectoryRunner, TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    execute_publication,
    parse_trajectory_retention_config,
)
from skyrl_train.trajectory_runners.trajectory_retention_publisher import InlineTrajectoryPublisher
from tests.cpu.util import example_dummy_config


@pytest.fixture
def dummy_config():
    return example_dummy_config()


def configure_eval(cfg, tmp_path):
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
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)
    return cfg


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
async def test_offline_preference_evaluation_generates_with_separate_validation_runner(dummy_config, tmp_path):
    cfg = configure_eval(dummy_config, tmp_path)
    cfg.environment.env_class = "preference_pair"
    cfg.generator.enable_http_endpoint = False
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "generated validation answer"
    training_runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=64
    )
    evaluation_runner = DummyRunner(
        {
            "prompt_token_ids": [[101]],
            "response_ids": [[201]],
            "rewards": [1.0],
            "loss_masks": [[1]],
            "stop_reasons": ["stop"],
            "rollout_logprobs": None,
            "verification_results": [VerificationResult.verified(1.0)],
        }
    )
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.tokenizer = tokenizer
    trainer.global_step = 4
    trainer.trajectory_runner = training_runner
    trainer.eval_trajectory_runner = evaluation_runner
    trainer.inference_engine_client = InferenceEngineClient([], tokenizer, cfg)
    trainer.trajectory_sink = TrajectorySink(
        parse_trajectory_retention_config(cfg.generator.trajectory_retention),
        tokenizer,
        publisher=InlineTrajectoryPublisher(execute_publication),
    )
    trainer.eval_dataloader = DummyStatefulDataLoader(
        [
            [
                {
                    "prompt": [{"role": "user", "content": "validation question"}],
                    "env_class": "gsm8k",
                    "env_extras": {"data_source": "holdout"},
                    "uid": "held-out-task",
                }
            ]
        ]
    )
    metrics = await trainer.eval()
    assert metrics["eval/all/avg_verifier_score"] == 1.0
    assert metrics["eval/all/verifier_score_coverage"] == 1.0
    assert metrics["eval/holdout/pass_at_1"] == 1.0


def test_eval_reports_normalized_verifier_score_alongside_raw_reward():
    batch: TrajectoryBatch = {
        "response_ids": [[1], [2]],
        "rewards": [5.0, 0.0],
        "verification_results": [
            VerificationResult.verified(5.0, score_min=1.0, score_max=5.0),
            VerificationResult.verified(0.0),
        ],
    }

    metrics = _calculate_eval_metrics(batch, ["a", "b"], ["genrm", "math"], 1)

    assert metrics["eval/all/avg_score"] == 2.5
    assert metrics["eval/all/avg_verifier_score"] == 0.5
    assert metrics["eval/all/verifier_score_coverage"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_computes_expected_metrics(dummy_config, tmp_path, monkeypatch):
    monkeypatch.setitem(registry, "custom_env", registry["gsm8k"])
    cfg = configure_eval(dummy_config, tmp_path)

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
        "env_classes": ["gsm8k", "custom_env"],
        "env_metrics": [{"truncated": 1}, {"truncated": 0}],
    }
    runner = DummyRunner(trajectory_batch)

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"
    # The run's shared sink is a Ray actor; an in-process sink with the same configuration keeps this test off Ray.
    sink = TrajectorySink(
        parse_trajectory_retention_config(cfg.generator.trajectory_retention),
        tokenizer,
        publisher=InlineTrajectoryPublisher(execute_publication),
    )

    metrics = await evaluate(
        eval_dataloader=eval_dataloader,
        trajectory_runner=runner,
        cfg=cfg,
        global_step=5,
        tokenizer=tokenizer,
        trajectory_sink=sink,
    )

    expected_metrics = {
        "eval/dataset_a/avg_score": 1.0,
        "eval/dataset_a/pass_at_1": 1.0,
        "eval/dataset_b/avg_score": 0.0,
        "eval/dataset_b/pass_at_1": 0.0,
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
        "eval/all/environment/gsm8k/truncated": 1.0,
        "eval/all/environment/custom_env/truncated": 0.0,
        "eval/dataset_a/environment/gsm8k/truncated": 1.0,
        "eval/dataset_b/environment/custom_env/truncated": 0.0,
    }

    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)

    assert len(runner.seen_inputs) == 1
    seen_batch = runner.seen_inputs[0]
    assert seen_batch["prompts"] == [prompt["prompt"] for prompt in prompts_batch]
    assert seen_batch["env_classes"] == ["gsm8k", "custom_env"]
    assert seen_batch["env_extras"] == [prompt["env_extras"] for prompt in prompts_batch]
    assert seen_batch["batch_metadata"].training_phase == "eval"
