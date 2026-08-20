"""CPU tests for how vLLM engine stats reach a consumer.

``VLLMStatsCallback`` queries the engines by RPC and adds the result to the trainer's per-step
metrics, which is what carries them to every configured tracker backend and to the ``WANDB_MIRROR``
stdout line the offline parsers and the nightly gate read. Writing them straight to wandb instead
kept them out of both, and dropped them entirely on a run launched with ``logger=console``.
"""

import asyncio

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import VLLMStatsCallback, engine_metrics

ONE_ENGINE = {
    "num_engines": 1,
    "total_peak_running_reqs": 12,
    "avg_peak_generation_throughput": 1450.5,
    "avg_median_generation_throughput": 1310.0,
    "avg_latency_e2e_mean": 4.25,
    "total_finished_requests": 64,
}

TWO_ENGINES = {
    **ONE_ENGINE,
    "num_engines": 2,
    "engines": [
        {"peak_generation_throughput": 700.0, "latency_e2e_mean": 4.0},
        {"peak_generation_throughput": 750.5, "latency_e2e_mean": 4.5},
    ],
}


class FakeEngineClient:
    def __init__(self, stats):
        self.stats = stats

    async def get_stats(self):
        return self.stats


class FakeTrainer:
    """Only the surface the callback touches: the per-step metric dict and the engine client."""

    def __init__(self, stats):
        self.all_metrics = {}
        self.inference_engine_client = FakeEngineClient(stats)


def _run_step(stats, *, global_step=3, **callback_kwargs):
    trainer = FakeTrainer(stats)
    callback = VLLMStatsCallback(**callback_kwargs)
    state = TrainerState(global_step=global_step, epoch=0, total_steps=10, num_steps_per_epoch=10)
    control = TrainerControl()
    callback.on_train_begin(state, control, trainer=trainer)
    asyncio.run(callback.on_step_end_async(state, control, trainer=trainer))
    return trainer


def test_engine_stats_land_in_the_trainers_step_metrics():
    trainer = _run_step(ONE_ENGINE)

    assert trainer.all_metrics["vllm/peak_generation_throughput"] == 1450.5
    assert trainer.all_metrics["vllm/median_generation_throughput"] == 1310.0
    assert trainer.all_metrics["vllm/latency_e2e_mean"] == 4.25
    assert trainer.all_metrics["vllm/total_finished_requests"] == 64
    assert all(key.startswith("vllm/") for key in trainer.all_metrics)


def test_per_engine_keys_appear_only_with_more_than_one_engine():
    assert not any("engine_0/" in key for key in engine_metrics(ONE_ENGINE))

    multi = engine_metrics(TWO_ENGINES)
    assert multi["vllm/engine_0/peak_generation_throughput"] == 700.0
    assert multi["vllm/engine_1/peak_generation_throughput"] == 750.5
    per_engine = {key.split("/", 2)[2] for key in multi if key.startswith("vllm/engine_0/")}
    assert per_engine == {key.split("/", 2)[2] for key in multi if key.startswith("vllm/engine_1/")}
    assert set(multi) == set(engine_metrics(ONE_ENGINE)) | {
        f"vllm/engine_{index}/{name}" for index in (0, 1) for name in per_engine
    }


def test_stats_are_collected_on_the_configured_interval_only():
    assert _run_step(ONE_ENGINE, global_step=3, log_every_steps=2).all_metrics == {}
    assert _run_step(ONE_ENGINE, global_step=4, log_every_steps=2).all_metrics != {}


def test_console_only_reporting_leaves_the_step_metrics_alone():
    assert _run_step(ONE_ENGINE, log_to_tracker=False).all_metrics == {}


def test_an_engineless_payload_records_nothing():
    assert _run_step({"num_engines": 0}).all_metrics == {}
