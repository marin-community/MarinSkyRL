import asyncio
from dataclasses import fields

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import InferenceStatsCallback
from skyrl_train.inference_engines.vllm.stats import (
    DistributionSummary,
    HTTPBridgeStatsSnapshot,
    IntervalReadMode,
    VLLMCumulativeStats,
    VLLMCurrentStats,
    VLLMEngineStatsSnapshot,
    VLLMIntervalStats,
    InferenceStatsSnapshot,
)
from skyrl_train.inference_observability import trainer_metrics


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
    *(
        f"inference_bridge/{name}/{statistic}"
        for name in (
            "event_loop_lag_seconds",
            "response_bytes",
            "json_serialization_seconds",
        )
        for statistic in ("count", "mean", "p95", "maximum")
    ),
}


def _snapshot():
    values = {field.name: index + 1 for index, field in enumerate(fields(VLLMIntervalStats))}
    interval = VLLMIntervalStats(**values)
    return InferenceStatsSnapshot(
        (
            VLLMEngineStatsSnapshot(
                "engine-0",
                1.0,
                current=VLLMCurrentStats(),
                cumulative=VLLMCumulativeStats(),
                interval=interval,
            ),
        ),
        http_bridge=HTTPBridgeStatsSnapshot(
            event_loop_lag_seconds=DistributionSummary(2, 0.01, 0.02, 0.02),
            response_bytes=DistributionSummary(2, 100, 120, 120),
            json_serialization_seconds=DistributionSummary(2, 0.001, 0.002, 0.002),
        ),
    )


class _Sink:
    def __init__(self):
        self.calls = []

    def publish(self, snapshot, step):
        self.calls.append((snapshot, step))


class _EngineClient:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.read_modes = []

    async def get_stats(self, read_mode):
        self.read_modes.append(read_mode)
        return self.snapshot


class _Trainer:
    def __init__(self, snapshot):
        self.all_metrics = {}
        self.inference_engine_client = _EngineClient(snapshot)


def _run_step(snapshot, *, global_step=3, sinks=(), **callback_kwargs):
    trainer = _Trainer(snapshot)
    callback = InferenceStatsCallback(sinks=sinks, **callback_kwargs)
    state = TrainerState(global_step=global_step, epoch=0, total_steps=10, num_steps_per_epoch=10)
    control = TrainerControl()
    callback.on_train_begin(state, control, trainer=trainer)
    asyncio.run(callback.on_step_end_async(state, control, trainer=trainer))
    return trainer


def test_callback_is_the_single_fanout_boundary_for_the_exact_snapshot():
    snapshot = _snapshot()
    sink = _Sink()

    trainer = _run_step(snapshot, sinks=(sink,))

    assert trainer.inference_engine_client.read_modes == [IntervalReadMode.RESET]
    assert set(trainer.all_metrics) == EXPECTED_KEYS
    assert sink.calls == [(snapshot, 3)]


def test_every_interval_field_is_projected_or_deliberately_typed_only():
    projected = trainer_metrics(_snapshot())
    projected_names = {name.removeprefix("vllm/") for name in projected}
    aliases = {
        "finished_requests": "total_finished_requests",
        "preempted_reqs": "total_preempted_reqs",
        "samples": "total_samples",
        "active_samples": "total_active_samples",
    }
    typed_only = {
        "mean_prompt_throughput",
        "mean_generation_throughput",
        "latency_prefill_median",
        "latency_decode_median",
        "latency_e2e_median",
        "latency_queued_median",
        "latency_ttft_median",
    }

    missing = {
        field.name
        for field in fields(VLLMIntervalStats)
        if aliases.get(field.name, field.name) not in projected_names and field.name not in typed_only
    }
    assert missing == set()


def test_stats_are_collected_only_on_the_configured_interval():
    trainer = _run_step(_snapshot(), global_step=3, log_every_steps=2)
    assert trainer.all_metrics == {}
    assert trainer.inference_engine_client.read_modes == []
