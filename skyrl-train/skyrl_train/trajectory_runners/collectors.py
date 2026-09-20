"""Harness-specific collection boundary used before trajectory projection."""

from typing import Awaitable, Callable, Generic, Protocol, TypeVar

from omegaconf import DictConfig

from skyrl_train.trajectory_runners.types import TrajectoryRequestBatch
from skyrl_train.retention_observability import GenerationRetentionObserver, estimate_agent_loop_payload
from skyrl_train.utils.progress import tqdm


InteractionT = TypeVar("InteractionT")


class RolloutCollector(Protocol, Generic[InteractionT]):
    """Collect structured interaction records from one harness."""

    def validate(self) -> None: ...

    async def collect(self, request: TrajectoryRequestBatch, *, disable_tqdm: bool = False) -> InteractionT: ...


class AgentLoopRunner(Protocol, Generic[InteractionT]):
    """Runner context required by the shared agent-loop fan-out helper."""

    trajectory_runner_cfg: DictConfig
    global_step_fn: Callable[[], int | None]


async def collect_agent_loops(
    runner: AgentLoopRunner[InteractionT],
    request: TrajectoryRequestBatch,
    agent_loop: Callable[..., Awaitable[InteractionT]],
    *,
    disable_tqdm: bool,
    on_error: Callable[[int, Exception], InteractionT] | None = None,
    retention_observer: GenerationRetentionObserver | None = None,
) -> list[InteractionT]:
    """Fan a request batch out over one harness-specific agent loop."""
    trajectory_ids = request.get("trajectory_ids")
    sampling_params = request.get("sampling_params")

    async def collect_one(index: int, prompt, env_class: str, env_extra: dict) -> InteractionT:
        producer = object()
        if retention_observer is not None:
            retention_observer.producer_started(producer, state="generating")
        try:
            try:
                result = await agent_loop(
                    prompt,
                    env_class,
                    env_extra,
                    runner.trajectory_runner_cfg.sampling_params.max_generate_length,
                    runner.trajectory_runner_cfg.max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[index] if trajectory_ids is not None else None,
                    global_step_fn=runner.global_step_fn,
                )
            except Exception as error:
                if on_error is None:
                    raise
                result = on_error(index, error)
            if retention_observer is not None:
                retention_observer.register_estimate(
                    result,
                    estimate=estimate_agent_loop_payload(result),
                    owner="fanout_completed",
                )
            return result
        finally:
            if retention_observer is not None:
                retention_observer.producer_finished(producer)

    tasks = [
        collect_one(index, prompt, env_class, env_extra)
        for index, (prompt, env_class, env_extra) in enumerate(
            zip(request["prompts"], request["env_classes"], request["env_extras"])
        )
    ]
    return await tqdm.gather(
        *tasks,
        desc="Generating Trajectories",
        miniters=max(1, len(tasks) / 10),
        mininterval=5,
        disable=disable_tqdm,
    )
