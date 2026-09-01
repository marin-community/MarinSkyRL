"""Workload-owned construction for Harbor trajectory runners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled
from skyrl_train.trajectory_runners.base import TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink


class HarborRunner(Protocol):
    """Lifecycle surface shared by in-process and process-isolated Harbor runners."""

    async def startup(self) -> None: ...

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch: ...

    async def shutdown(self) -> None: ...

    def set_trajectory_sink(self, sink: TrajectorySink) -> None: ...

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None: ...

    async def stop_eval_session(self) -> None: ...


def _detached(config: DictConfig) -> DictConfig:
    """Return a serializable copy without OmegaConf parent references."""
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


class ExecutionEnvironment(StrEnum):
    """The environment in which a trajectory workload runs."""

    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TEST = "test"


@dataclass(frozen=True)
class TrajectoryWorkload:
    """Workload facts that determine the execution placement."""

    environment: ExecutionEnvironment


@dataclass(frozen=True)
class ProcessPoolResources:
    """Resources for the process-isolated Harbor runner pool."""

    num_coordinators: int
    cpus_per_coordinator: int
    executor_workers: int
    rpc_timeout_seconds: float

    @classmethod
    def from_config(cls, config: DictConfig) -> ProcessPoolResources:
        process_pool = config.trajectory_runner.process_pool
        resources = cls(
            num_coordinators=int(process_pool.num_coordinators),
            cpus_per_coordinator=int(process_pool.cpus_per_coordinator),
            executor_workers=int(process_pool.executor_workers),
            rpc_timeout_seconds=float(process_pool.rpc_timeout_seconds),
        )
        if resources.num_coordinators <= 0 or resources.cpus_per_coordinator <= 0:
            raise ValueError("trajectory runner process-pool sizes must be positive")
        if resources.executor_workers <= 0 or resources.rpc_timeout_seconds <= 0:
            raise ValueError("trajectory runner executor size and RPC timeout must be positive")
        return resources


@dataclass(frozen=True)
class HarborRunnerSpec:
    """Serializable inputs required to construct one Harbor runner."""

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

    def with_runner_config(self, runner_config: DictConfig) -> HarborRunnerSpec:
        return replace(self, runner_config=_detached(runner_config))

    def with_terminal_bench_config(self, terminal_bench_config: DictConfig) -> HarborRunnerSpec:
        return replace(self, terminal_bench_config=_detached(terminal_bench_config))

    def build(self, tokenizer: PreTrainedTokenizerBase) -> HarborRunner:
        """Construct an in-process Harbor runner."""
        from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner  # noqa: PLC0415

        algorithm = self.config.trainer.algorithm
        return HarborTrajectoryRunner(
            trajectory_runner_cfg=self.runner_config,
            terminal_bench_cfg=self.terminal_bench_config,
            tokenizer=tokenizer,
            moe_router_replay=bool(self.config.trainer.policy.fsdp_config.get("moe_router_replay", False)),
            rollout_logprobs_required=rollout_logprobs_enabled(algorithm),
            tito_full=algorithm.get("tito_full", None),
            tis_splice=bool(algorithm.tis_splice),
            tis_lcs_alert_threshold=float(algorithm.tis_lcs_alert_threshold),
        )


def build_harbor_trajectory_runner(
    *,
    spec: HarborRunnerSpec,
    workload: TrajectoryWorkload,
    tokenizer: PreTrainedTokenizerBase,
    resources: ProcessPoolResources,
) -> HarborRunner:
    """Select execution placement from the runner workload, before trainer construction."""
    if workload.environment is ExecutionEnvironment.PRODUCTION:
        from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import RolloutDispatcher  # noqa: PLC0415

        return RolloutDispatcher(spec=spec, resources=resources)
    return spec.build(tokenizer)
