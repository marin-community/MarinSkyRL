"""Harness-specific collection boundary used before trajectory projection."""

from typing import Awaitable, Callable, Generic, Protocol, TypeVar

from skyrl_train.trajectory_runners.types import TrajectoryRequestBatch
from skyrl_train.utils.progress import tqdm


InteractionT = TypeVar("InteractionT")


class RolloutCollector(Protocol, Generic[InteractionT]):
    """Collect structured interaction records from one harness."""

    def validate(self) -> None: ...

    async def collect(self, request: TrajectoryRequestBatch, *, disable_tqdm: bool = False) -> InteractionT: ...


async def collect_agent_loops(
    runner,
    request: TrajectoryRequestBatch,
    agent_loop: Callable[..., Awaitable[InteractionT]],
    *,
    disable_tqdm: bool,
) -> list[InteractionT]:
    """Fan a request batch out over one harness-specific agent loop."""
    trajectory_ids = request.get("trajectory_ids")
    sampling_params = request.get("sampling_params")
    tasks = [
        agent_loop(
            prompt,
            env_class,
            env_extra,
            runner.trajectory_runner_cfg.sampling_params.max_generate_length,
            runner.trajectory_runner_cfg.max_input_length,
            sampling_params=sampling_params,
            trajectory_id=trajectory_ids[index] if trajectory_ids is not None else None,
            global_step_fn=runner.global_step_fn,
        )
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
