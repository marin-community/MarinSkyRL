import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skyrl_train.trajectory_runners.nemotron_ultra import NemotronUltraTrajectoryRouter
from skyrl_train.trajectory_runners.types import TrajectoryID


class StubRunner:
    def __init__(self, reward: float, *, token_level_rewards: bool = False):
        self.reward = reward
        self.token_level_rewards = token_level_rewards
        self.requests = []

    async def startup(self):
        pass

    async def shutdown(self):
        pass

    async def start_eval_session(self, **kwargs):
        pass

    async def stop_eval_session(self):
        pass

    def set_trajectory_sink(self, sink):
        self.sink = sink

    async def run(self, request, disable_tqdm=False):
        self.requests.append(request)
        size = len(request["prompts"])
        return {
            "prompt_token_ids": [[index] for index in range(size)],
            "response_ids": [[int(self.reward)] for _ in range(size)],
            "rewards": ([[self.reward]] * size if self.token_level_rewards else [self.reward] * size),
            "loss_masks": [[1]] * size,
            "stop_reasons": ["complete"] * size,
            "rollout_metrics": {},
            "rollout_logprobs": [[-0.1]] * size,
            "trajectory_ids": list(request["trajectory_ids"]),
        }


def _task(root: Path, name: str) -> None:
    directory = root / name
    directory.mkdir()
    (directory / "instruction.md").write_text("fix it")


@pytest.mark.asyncio
async def test_routes_only_swe_to_terminal_bench_and_restores_order(tmp_path):
    _task(tmp_path, "swe-1")
    gym = StubRunner(1.0, token_level_rewards=True)
    harbor = StubRunner(2.0)
    router = NemotronUltraTrajectoryRouter(
        gym_runner=gym,
        harbor_runner=harbor,
        terminal_bench_data=[str(tmp_path)],
        require_rollout_logprobs=True,
        tis_lcs_alert_threshold=0.005,
    )
    ids = [TrajectoryID(str(index), 0) for index in range(3)]
    batch = {
        "prompts": [[{"role": "user", "content": str(index)}] for index in range(3)],
        "env_classes": ["nemotron_ultra"] * 3,
        "env_extras": [
            {"extra_info": {"nemotron_ultra": {"blend": "rlvr1", "agent": "calendar_simple_agent", "route": "gym"}}},
            {
                "extra_info": {
                    "nemotron_ultra": {
                        "blend": "rlvr1",
                        "agent": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
                        "route": "terminal_bench",
                        "terminal_bench_instance_id": "swe-1",
                    }
                }
            },
            {"extra_info": {"nemotron_ultra": {"blend": "rlvr2", "agent": "calendar_simple_agent", "route": "gym"}}},
        ],
        "sampling_params": None,
        "trajectory_ids": ids,
        "batch_metadata": SimpleNamespace(training_phase="train"),
    }

    result = await router.run(batch)

    assert result["rewards"] == [[1.0], [2.0], [1.0]]
    assert result["trajectory_ids"] == ids
    assert result["rollout_metrics"]["nemotron_ultra/coverage/rlvr1/calendar_simple_agent"] == 1
    assert (
        result["rollout_metrics"][
            "nemotron_ultra/coverage/rlvr1/swe_pivot_single_step_tool_use_with_argument_comparison_agent"
        ]
        == 1
    )
    assert result["rollout_metrics"]["nemotron_ultra/coverage/rlvr2/calendar_simple_agent"] == 1
    assert harbor.requests[0]["prompts"] == [str(tmp_path / "swe-1")]
    assert gym.requests[0]["prompts"] == [batch["prompts"][0], batch["prompts"][2]]


@pytest.mark.asyncio
async def test_missing_terminal_bench_task_is_rejected(tmp_path):
    router = NemotronUltraTrajectoryRouter(
        gym_runner=StubRunner(1.0),
        harbor_runner=StubRunner(2.0),
        terminal_bench_data=[str(tmp_path)],
        require_rollout_logprobs=False,
        tis_lcs_alert_threshold=0.005,
    )
    batch = {
        "prompts": [[{"role": "user", "content": "x"}]],
        "env_classes": ["nemotron_ultra"],
        "env_extras": [
            {
                "extra_info": {
                    "nemotron_ultra": {
                        "blend": "rlvr1",
                        "agent": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
                        "route": "terminal_bench",
                        "terminal_bench_instance_id": "missing",
                    }
                }
            }
        ],
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("0", 0)],
        "batch_metadata": None,
    }

    with pytest.raises(ValueError, match="absent"):
        await router.run(batch)


@pytest.mark.asyncio
async def test_nvidia_swe_ids_match_lowercase_harbor_task_paths(tmp_path):
    _task(tmp_path, "project-monai__monai-123")
    harbor = StubRunner(2.0)
    router = NemotronUltraTrajectoryRouter(
        gym_runner=StubRunner(1.0),
        harbor_runner=harbor,
        terminal_bench_data=[str(tmp_path)],
        require_rollout_logprobs=False,
        tis_lcs_alert_threshold=0.005,
    )
    batch = {
        "prompts": [[{"role": "user", "content": "x"}]],
        "env_classes": ["nemotron_ultra"],
        "env_extras": [
            {
                "extra_info": {
                    "nemotron_ultra": {
                        "blend": "rlvr1",
                        "agent": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
                        "route": "terminal_bench",
                        "terminal_bench_instance_id": "Project-MONAI__MONAI-123",
                    }
                }
            }
        ],
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("0", 0)],
        "batch_metadata": None,
    }

    await router.run(batch)

    assert harbor.requests[0]["prompts"] == [str(tmp_path / "project-monai__monai-123")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_file", "metadata", "instance_id"),
    [
        ("config.json", {"instance_id": "Project-MONAI__MONAI-123"}, "Project-MONAI__MONAI-123"),
        (
            "test_info.json",
            {"github_repo": "python-pillow/Pillow", "base_commit": "3ba81234"},
            "python-pillow__Pillow-3ba81234",
        ),
    ],
)
async def test_source_instance_ids_resolve_generic_harbor_task_names(tmp_path, metadata_file, metadata, instance_id):
    _task(tmp_path, "generic-task-0001")
    tests = tmp_path / "generic-task-0001" / "tests"
    tests.mkdir()
    (tests / metadata_file).write_text(json.dumps(metadata))
    harbor = StubRunner(2.0)
    router = NemotronUltraTrajectoryRouter(
        gym_runner=StubRunner(1.0),
        harbor_runner=harbor,
        terminal_bench_data=[str(tmp_path)],
        require_rollout_logprobs=False,
        tis_lcs_alert_threshold=0.005,
    )
    batch = {
        "prompts": [[{"role": "user", "content": "x"}]],
        "env_classes": ["nemotron_ultra"],
        "env_extras": [
            {
                "extra_info": {
                    "nemotron_ultra": {
                        "blend": "rlvr2",
                        "agent": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
                        "route": "terminal_bench",
                        "terminal_bench_instance_id": instance_id,
                    }
                }
            }
        ],
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("0", 0)],
        "batch_metadata": None,
    }

    await router.run(batch)

    assert harbor.requests[0]["prompts"] == [str(tmp_path / "generic-task-0001")]
