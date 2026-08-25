"""CPU tests for how vLLM engine stats reach a consumer.

``VLLMStatsCallback`` queries the engines by RPC and adds the result to the trainer's per-step
metrics, which is what carries them to every configured tracker backend and to the ``WANDB_MIRROR``
stdout line the offline parsers and the nightly gate read.
"""

import ast
import asyncio
import re
from pathlib import Path

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import VLLMStatsCallback

_SKYRL_TRAIN = Path(__file__).resolve().parents[3] / "skyrl_train"

# The keys logged for a step. Pinned by name: a rename here is a break in every existing panel.
EXPECTED_KEYS = {
    "vllm/num_engines",
    "vllm/peak_running_reqs",
    "vllm/peak_waiting_reqs",
    "vllm/peak_prompt_throughput",
    "vllm/peak_generation_throughput",
    "vllm/peak_gpu_cache_usage_perc",
    "vllm/peak_prefix_cache_hit_rate",
    "vllm/median_running_reqs",
    "vllm/median_waiting_reqs",
    "vllm/median_prompt_throughput",
    "vllm/median_generation_throughput",
    "vllm/median_gpu_cache_usage_perc",
    "vllm/median_prefix_cache_hit_rate",
    "vllm/latency_prefill_mean",
    "vllm/latency_prefill_p90",
    "vllm/latency_decode_mean",
    "vllm/latency_decode_p90",
    "vllm/latency_e2e_mean",
    "vllm/latency_e2e_p90",
    "vllm/latency_queued_mean",
    "vllm/latency_queued_p90",
    "vllm/latency_ttft_mean",
    "vllm/latency_ttft_p90",
    "vllm/total_finished_requests",
    "vllm/total_preempted_reqs",
    "vllm/total_samples",
    "vllm/total_active_samples",
}

ONE_ENGINE = {
    "num_engines": 1,
    "total_peak_running_reqs": 12,
    "total_peak_waiting_reqs": 3,
    "avg_peak_prompt_throughput": 900.0,
    "avg_peak_generation_throughput": 1450.5,
    "avg_peak_gpu_cache_usage_perc": 62.5,
    "avg_peak_prefix_cache_hit_rate": 0.41,
    "avg_median_running_reqs": 9.0,
    "avg_median_waiting_reqs": 1.0,
    "avg_median_prompt_throughput": 850.0,
    "avg_median_generation_throughput": 1310.0,
    "avg_median_gpu_cache_usage_perc": 58.0,
    "avg_median_prefix_cache_hit_rate": 0.38,
    "avg_latency_prefill_mean": 0.12,
    "max_latency_prefill_p90": 0.30,
    "avg_latency_decode_mean": 3.9,
    "max_latency_decode_p90": 8.0,
    "avg_latency_e2e_mean": 4.25,
    "max_latency_e2e_p90": 9.5,
    "avg_latency_queued_mean": 0.05,
    "max_latency_queued_p90": 0.20,
    "avg_latency_ttft_mean": 0.18,
    "max_latency_ttft_p90": 0.44,
    "total_finished_requests": 64,
    "total_preempted_reqs": 3,
    "total_samples": 128,
    "total_active_samples": 40,
}


class FakeEngineClient:
    def __init__(self, stats):
        self.stats = stats

    async def get_stats(self):
        return self.stats


class FakeTrainer:
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

    assert set(trainer.all_metrics) == EXPECTED_KEYS
    assert trainer.all_metrics["vllm/peak_generation_throughput"] == 1450.5
    assert trainer.all_metrics["vllm/median_generation_throughput"] == 1310.0
    assert trainer.all_metrics["vllm/latency_e2e_mean"] == 4.25
    assert trainer.all_metrics["vllm/total_finished_requests"] == 64


def test_every_key_read_from_a_payload_is_one_get_stats_produces():
    """The reads are unguarded, so a producer-side rename is a crash rather than a silent zero."""
    read = set(re.findall(r'stats\["([^"]+)"\]', (_SKYRL_TRAIN / "callbacks" / "builtin.py").read_text()))
    client = ast.parse((_SKYRL_TRAIN / "inference_engines" / "inference_engine_client.py").read_text())
    get_stats = next(
        node for node in ast.walk(client) if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_stats"
    )
    # The latency keys arrive through `**latency_agg`, built from this list one line above the loop.
    latency_keys = {
        f"{prefix}_latency_{name.value}_{suffix}"
        for node in ast.walk(get_stats)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.List)
        and any(isinstance(target, ast.Name) and target.id == "latency_keys" for target in node.targets)
        for name in node.value.elts
        if isinstance(name, ast.Constant)
        for prefix, suffix in (("avg", "mean"), ("max", "p90"))
    }
    produced = latency_keys | {
        key.value
        for node in ast.walk(get_stats)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant)
    }
    assert not read - produced, f"read from get_stats but never produced: {sorted(read - produced)}"


def test_stats_are_collected_on_the_configured_interval_only():
    assert _run_step(ONE_ENGINE, global_step=3, log_every_steps=2).all_metrics == {}
    assert set(_run_step(ONE_ENGINE, global_step=4, log_every_steps=2).all_metrics) == EXPECTED_KEYS


def test_console_only_reporting_leaves_the_step_metrics_alone():
    assert _run_step(ONE_ENGINE, log_to_tracker=False).all_metrics == {}


def test_an_engineless_payload_records_nothing():
    assert _run_step({"num_engines": 0}).all_metrics == {}
