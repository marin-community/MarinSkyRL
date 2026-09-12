"""Route Nemotron Ultra's mixed RLVR batches to SkyRL Gym and Harbor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol

from skyrl_train.trajectory_runners.base import TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, retain_trajectories


class _Runner(Protocol):
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch: ...

    def set_trajectory_sink(self, sink: TrajectorySink) -> None: ...

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
        n_concurrent_trials: int | None = None,
    ) -> None: ...

    async def stop_eval_session(self) -> None: ...


def _select_rows(batch: TrajectoryRequestBatch, indices: list[int]) -> TrajectoryRequestBatch:
    size = len(batch["prompts"])
    selected: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, list) and len(value) == size:
            selected[key] = [value[index] for index in indices]
        else:
            selected[key] = value
    return selected  # type: ignore[return-value]


def _ultra_metadata(extras: dict[str, Any]) -> dict[str, Any]:
    extra_info = extras.get("extra_info")
    ultra = extra_info.get("nemotron_ultra") if isinstance(extra_info, dict) else None
    if not isinstance(ultra, dict):
        raise ValueError("Nemotron Ultra routing requires extra_info.nemotron_ultra for every row")
    return ultra


def _swe_instance_id(extras: dict[str, Any]) -> str | None:
    ultra = _ultra_metadata(extras)
    if not isinstance(ultra, dict) or ultra.get("route") != "terminal_bench":
        return None
    instance_id = ultra.get("terminal_bench_instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("a Nemotron Ultra terminal-bench row must provide terminal_bench_instance_id")
    return instance_id


def _coverage_key(extras: dict[str, Any]) -> str:
    ultra = _ultra_metadata(extras)
    blend = ultra.get("blend")
    agent = ultra.get("agent")
    if not isinstance(blend, str) or not blend or not isinstance(agent, str) or not agent:
        raise ValueError("Nemotron Ultra routing requires non-empty blend and agent fields for every row")
    return f"nemotron_ultra/coverage/{blend}/{agent}"


def _task_ids(task_path: Path) -> set[str]:
    """Return Harbor's directory ID and any source-dataset ID in task metadata."""
    task_ids = {task_path.name}

    swe_gym_config = task_path / "tests" / "config.json"
    if swe_gym_config.is_file():
        config = json.loads(swe_gym_config.read_text())
        instance_id = config.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            task_ids.add(instance_id)

    r2e_info = task_path / "tests" / "test_info.json"
    if r2e_info.is_file():
        info = json.loads(r2e_info.read_text())
        github_repo = info.get("github_repo")
        base_commit = info.get("base_commit")
        if isinstance(github_repo, str) and github_repo and isinstance(base_commit, str) and base_commit:
            task_ids.add(f"{github_repo.replace('/', '__')}-{base_commit}")

    return {task_id.casefold() for task_id in task_ids}


def _task_index(data_files: list[str]) -> dict[str, str]:
    dataset = TerminalBenchTaskDataset(data_files=data_files)
    result: dict[str, str] = {}
    for task_path in dataset.get_task_paths():
        # Harbor-safe directories can be lowercased or assigned generic names.
        # NVIDIA's blend keeps the upstream SWE-Gym/R2E-Gym instance ID, so
        # index both representations using the metadata shipped with the task.
        for task_id in _task_ids(task_path):
            if task_id in result and result[task_id] != str(task_path):
                raise ValueError(f"duplicate terminal-bench task ID {task_id!r}: {result[task_id]} and {task_path}")
            result[task_id] = str(task_path)
    return result


class NemotronUltraTrajectoryRouter:
    """Dispatch ordinary RLVR rows to Gym and SWE rows to Harbor.

    Both child runners retain their normal public lifecycle and finalization. The
    router only partitions and restores row order; it does not reinterpret either
    runner's token stream or reward.
    """

    def __init__(
        self,
        *,
        gym_runner: _Runner,
        harbor_runner: _Runner,
        terminal_bench_data: list[str],
        require_rollout_logprobs: bool,
        tis_lcs_alert_threshold: float,
    ) -> None:
        self.gym_runner = gym_runner
        self.harbor_runner = harbor_runner
        self.task_paths = _task_index(terminal_bench_data)
        self.require_rollout_logprobs = require_rollout_logprobs
        self.tis_lcs_alert_threshold = tis_lcs_alert_threshold
        self._global_step_fn = None
        self.trajectory_sink: TrajectorySink | None = None

    @property
    def global_step_fn(self):
        return self._global_step_fn

    @global_step_fn.setter
    def global_step_fn(self, callback) -> None:
        self._global_step_fn = callback
        self.gym_runner.global_step_fn = callback
        self.harbor_runner.global_step_fn = callback

    async def startup(self) -> None:
        await asyncio.gather(self.gym_runner.startup(), self.harbor_runner.startup())

    async def shutdown(self) -> None:
        await asyncio.gather(self.gym_runner.shutdown(), self.harbor_runner.shutdown())

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
        n_concurrent_trials: int | None = None,
    ) -> None:
        await asyncio.gather(
            self.gym_runner.start_eval_session(
                run_name=run_name,
                eval_step=eval_step,
                val_set_name=val_set_name,
                n_concurrent_trials=n_concurrent_trials,
            ),
            self.harbor_runner.start_eval_session(
                run_name=run_name,
                eval_step=eval_step,
                val_set_name=val_set_name,
                n_concurrent_trials=n_concurrent_trials,
            ),
        )

    async def stop_eval_session(self) -> None:
        await asyncio.gather(self.gym_runner.stop_eval_session(), self.harbor_runner.stop_eval_session())

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        env_extras = input_batch.get("env_extras")
        if env_extras is None or len(env_extras) != len(input_batch["prompts"]):
            raise ValueError("Nemotron Ultra routing requires one env_extras mapping per request row")

        gym_indices: list[int] = []
        harbor_indices: list[int] = []
        for index, extras in enumerate(env_extras):
            instance_id = _swe_instance_id(extras)
            if instance_id is None:
                gym_indices.append(index)
                continue
            if instance_id.casefold() not in self.task_paths:
                raise ValueError(f"terminal-bench task {instance_id!r} is absent from the configured task data")
            harbor_indices.append(index)

        jobs = []
        index_groups: list[list[int]] = []
        if gym_indices:
            gym_batch = _select_rows(input_batch, gym_indices)
            jobs.append(self.gym_runner.run(gym_batch, disable_tqdm=disable_tqdm))
            index_groups.append(gym_indices)
        if harbor_indices:
            harbor_batch = _select_rows(input_batch, harbor_indices)
            harbor_batch["prompts"] = [
                self.task_paths[_swe_instance_id(env_extras[index]).casefold()]  # type: ignore[union-attr]
                for index in harbor_indices
            ]
            jobs.append(self.harbor_runner.run(harbor_batch, disable_tqdm=disable_tqdm))
            index_groups.append(harbor_indices)

        outputs = await asyncio.gather(*jobs)
        if len(outputs) == 1:
            result = outputs[0]
        else:
            result = concatenate_trajectory_batches(
                outputs,
                require_rollout_logprobs=self.require_rollout_logprobs,
                tis_lcs_alert_threshold=self.tis_lcs_alert_threshold,
            )

        concatenated_indices = [index for group in index_groups for index in group]
        restore = sorted(range(len(concatenated_indices)), key=concatenated_indices.__getitem__)
        batch_size = len(concatenated_indices)
        for key, value in list(result.items()):
            if isinstance(value, list) and len(value) == batch_size:
                result[key] = [value[index] for index in restore]
        rollout_metrics = result.setdefault("rollout_metrics", {})
        if rollout_metrics is None:
            rollout_metrics = result["rollout_metrics"] = {}
        for extras in env_extras:
            key = _coverage_key(extras)
            rollout_metrics[key] = rollout_metrics.get(key, 0) + 1
        if self.trajectory_sink is not None:
            await retain_trajectories(self.trajectory_sink, input_batch, result)
        return result
