"""Construction of Harbor trajectory runners inside rollout workers."""

from __future__ import annotations

from dataclasses import dataclass

import ray
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled

DEFAULT_CONCURRENT_TRIALS = 16
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", None)


def _detached(config: DictConfig) -> DictConfig:
    """Return a serializable copy without OmegaConf parent references."""
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def configured_concurrent_trials(terminal_bench_config: DictConfig) -> int:
    """The trial concurrency the config asks for across the whole pool."""
    harbor = terminal_bench_config.get("harbor", None)
    if harbor is not None:
        return int(harbor.get("n_concurrent_trials", DEFAULT_CONCURRENT_TRIALS))
    return int(terminal_bench_config.get("n_concurrent_trials", DEFAULT_CONCURRENT_TRIALS))


def per_worker_limits(terminal_bench_config: DictConfig, worker_count: int) -> DictConfig:
    """Divide Harbor's per-process limits among the pool's workers, so the pool's total load matches the config.

    ``harbor.n_concurrent_trials`` bounds one process's concurrent sandboxes and model calls, and
    ``environment.kwargs.connection_pool_maxsize`` sizes one process's Daytona HTTP pool. Neither drops below 1.
    """
    scaled = OmegaConf.create(OmegaConf.to_container(terminal_bench_config, resolve=False))
    harbor = scaled.get("harbor", None)
    if harbor is not None and "n_concurrent_trials" in harbor:
        harbor.n_concurrent_trials = max(1, int(harbor.n_concurrent_trials) // worker_count)
    environment_kwargs = OmegaConf.select(scaled, "environment.kwargs")
    if environment_kwargs is not None and "connection_pool_maxsize" in environment_kwargs:
        environment_kwargs.connection_pool_maxsize = max(
            1, int(environment_kwargs.connection_pool_maxsize) // worker_count
        )
    return scaled


@dataclass(frozen=True)
class HarborRunnerSpec:
    """Serializable inputs that build one rollout worker's Harbor runner."""

    config: DictConfig
    runner_config: DictConfig
    terminal_bench_config: DictConfig

    @classmethod
    def from_config(cls, config: DictConfig) -> HarborRunnerSpec:
        return cls(
            config=_detached(config),
            runner_config=_detached(config.generator),
            terminal_bench_config=_detached(config.terminal_bench_config),
        )

    def build(self, tokenizer: PreTrainedTokenizerBase, shard: WorkerShard) -> TrajectoryRunner:
        """Construct this worker's Harbor runner with its share of the per-process limits.

        Evaluation runs on one worker, so its session keeps the pool's full trial concurrency.
        """
        # Harbor is an optional agent-harness dependency and is absent from the CPU launcher environment.
        from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner  # noqa: PLC0415

        runner_config = self.runner_config.copy()
        if runner_config.get("http_endpoint_host", None) in LOOPBACK_HOSTS:
            # The trainer's endpoint listens on every interface. Agents outside this host's network namespace
            # cannot reach its loopback address, but they can reach the node address.
            runner_config.http_endpoint_host = ray.util.get_node_ip_address()
        algorithm = self.config.trainer.algorithm
        return HarborTrajectoryRunner(
            trajectory_runner_cfg=runner_config,
            terminal_bench_cfg=per_worker_limits(self.terminal_bench_config, shard.count),
            eval_concurrent_trials=configured_concurrent_trials(self.terminal_bench_config),
            tokenizer=tokenizer,
            moe_router_replay=bool(self.config.trainer.policy.megatron_config.get("moe_router_replay", False)),
            rollout_logprobs_required=rollout_logprobs_enabled(algorithm),
            tito_full=algorithm.get("tito_full", None),
            tis_splice=bool(algorithm.tis_splice),
            tis_lcs_alert_threshold=float(algorithm.tis_lcs_alert_threshold),
        )
