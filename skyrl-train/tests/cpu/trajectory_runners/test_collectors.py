import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from skyrl_train.trajectory_runners.collectors import collect_agent_loops


@pytest.mark.asyncio
async def test_collect_agent_loops_bounds_actual_agent_loop_concurrency():
    runner = SimpleNamespace(
        trajectory_runner_cfg=OmegaConf.create(
            {
                "max_concurrent_agent_loops": 1,
                "max_input_length": 128,
                "sampling_params": {"max_generate_length": 32},
            }
        ),
        global_step_fn=lambda: 0,
    )
    request = {
        "prompts": ["first", "second", "third"],
        "env_classes": ["test"] * 3,
        "env_extras": [{} for _ in range(3)],
    }
    running = 0
    peak_running = 0

    async def agent_loop(prompt, *_args, **_kwargs):
        nonlocal running, peak_running
        running += 1
        peak_running = max(peak_running, running)
        await asyncio.sleep(0)
        running -= 1
        return prompt

    outputs = await collect_agent_loops(runner, request, agent_loop, disable_tqdm=True)

    assert outputs == request["prompts"]
    assert peak_running == 1
