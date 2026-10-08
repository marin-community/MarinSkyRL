import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from skyrl_gym.verification import RewardResult
from omegaconf import OmegaConf

try:
    import skyrl_train.trajectory_runners.harbor.runner as harbor_runner_module
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

from harbor.verifier.verifier import VerifierOutputParseError
from skyrl_train.metric_names import IDENTITY_AWARE_REWARD_METRIC_PREFIX
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryID
from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset
from skyrl_train.trajectory_runners.trajectory_retention import (
    build_trajectory_records,
    parse_trajectory_retention_config,
)
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
from skyrl_train.trajectory_runners.harbor.literal_log_store import LiteralLogStore


def _runner_output(
    trial: int,
    outcomes: dict[str, str],
    aggregate_reward: float,
    *,
    verifier_test_collection_factory,
    truncation_penalized: bool = False,
):
    return SimpleNamespace(
        trajectory_id=TrajectoryID(instance_id="task", repetition_id=trial),
        verifier_tests=verifier_test_collection_factory(trial, outcomes),
        disposition=SimpleNamespace(baseline_eligible=True),
        reward_result=RewardResult(unshaped_reward=aggregate_reward, optimization_reward=aggregate_reward),
        truncation_penalized=truncation_penalized,
    )


def _runner(shaper: str | None = None):
    runner = object.__new__(HarborTrajectoryRunner)
    runner._reward_shaping_enabled = True
    runner._reward_shaping_config = {"shaper_kwargs": {}}
    if shaper is not None:
        runner._reward_shaping_config["reward_shaper"] = shaper
    runner._truncation_penalty = 0.25
    return runner


class _Tokenizer:
    eos_token_id = 99

    def decode(self, _ids, **_kwargs):
        return "done"

    def apply_chat_template(self, *_args, add_generation_prompt=False, **_kwargs):
        return [1, 2] if add_generation_prompt else [1]


def _trial_runner() -> HarborTrajectoryRunner:
    runner = object.__new__(HarborTrajectoryRunner)
    runner._error_handling_config = ErrorHandlingConfig(
        enable_error_classification=True,
        passthrough_exceptions=frozenset({"TurnCapExhaustedError"}),
    )
    runner._rollout_logprobs_required = False
    runner._reward_shaping_enabled = False
    runner._collect_rollout_details = False
    runner._moe_router_replay = False
    runner._tito_full = None
    runner._tis_splice = True
    runner._truncation_penalty = 0.0
    runner._enable_token_reward_channel = False
    runner._chat_template_kwargs = {}
    runner.custom_chat_template_content = None
    runner.tokenizer = _Tokenizer()
    runner.trajectory_runner_cfg = OmegaConf.create(
        {"sampling_params": {"max_generate_length": 16}, "max_input_length": 16}
    )
    return runner


@pytest.mark.asyncio
async def test_task_harness_assignment_survives_shuffling_restarts_and_repeated_samples(tmp_path):
    for name in ("task-c", "task-a", "task-d", "task-b"):
        task = tmp_path / "data" / name
        task.mkdir(parents=True)
        (task / "instruction.md").write_text(name)
    panel = ("opencode", "claude-code", "codex", "mini-swe-agent")
    expected = dict(zip(("task-a", "task-b", "task-c", "task-d"), panel, strict=True))

    class CompletedOrchestrator:
        def __init__(self):
            self.assignments = {}

        async def submit_batch(self, trial_configs):
            futures = []
            for config in trial_configs:
                self.assignments.setdefault(config.task.path.name, set()).add(config.agent.name)
                future = asyncio.get_running_loop().create_future()
                future.set_result(SimpleNamespace(verifier_result=None, exception_info=None, agent_result=None))
                futures.append(future)
            return futures

    for order in ((3, 0, 2, 1), (1, 2, 0, 3)):
        dataset = TerminalBenchTaskDataset([str(tmp_path / "data")])
        items = [dataset[index] for index in order for _ in range(16)]
        runner = _trial_runner()
        orchestrator = CompletedOrchestrator()
        runner._orchestrator = orchestrator
        runner._orchestrator_started = True
        runner._eval_session_active = False
        runner._packed_task_materializer = harbor_runner_module.PackedTaskMaterializer(tmp_path / "packed")
        runner._harbor_config_builder = HarborConfigBuilder(
            OmegaConf.create({"harbor": {"agent_profiles": [{"name": name} for name in panel]}})
        )
        runner._agent_api_base = "http://localhost:8000/v1"
        runner._tracked_exceptions = []
        runner._literal_log_path = None
        runner.model_name = "model"
        runner.trials_dir = str(tmp_path / "trials")
        request = {
            "prompts": [item["prompt"] for item in items],
            "env_classes": ["bfcl"] * len(items),
            "env_extras": [item["env_extras"] for item in items],
            "trajectory_ids": [
                TrajectoryID(instance_id=item["uid"], repetition_id=index) for index, item in enumerate(items)
            ],
            "batch_metadata": BatchMetadata(global_step=0, training_phase="train"),
            "sampling_params": {"max_tokens": 16},
        }
        try:
            await runner.run(request)
        finally:
            runner._packed_task_materializer.close()
        assert orchestrator.assignments == {task: {agent} for task, agent in expected.items()}


@pytest.mark.parametrize(
    ("exception_info", "agent_stop_reason", "expected_exception", "expected_treatment"),
    [
        (
            SimpleNamespace(exception_type="TurnCapExhaustedError"),
            "turn_cap_exhausted",
            "TurnCapExhaustedError",
            "passthrough",
        ),
        (None, "complete", None, None),
    ],
)
def test_verified_harbor_result_preserves_terminal_disposition(
    monkeypatch, exception_info, agent_stop_reason, expected_exception, expected_treatment
):
    monkeypatch.setattr(
        harbor_runner_module,
        "get_response_ids_and_loss_mask_from_messages",
        lambda *_args, **_kwargs: ([10, 11], [1, 1], None),
    )
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}, stdout="passed"),
        exception_info=exception_info,
        agent_result=SimpleNamespace(
            metadata={
                "all_messages": [
                    {"role": "user", "content": "solve it"},
                    {"role": "assistant", "content": "done"},
                ],
                "summarization_count": 0,
                "stop_reason": agent_stop_reason,
            },
            rollout_details=None,
        ),
    )

    output = _trial_runner()._process_trial_result(
        result,
        TrajectoryID(instance_id="task", repetition_id=0),
    )

    assert output.verification.score == 1.0
    assert output.reward_result.unshaped_reward == 1.0
    assert output.reward_result.optimization_reward == 1.0
    assert output.evidence.stop_reason == agent_stop_reason
    assert output.disposition.loss_eligible
    assert output.disposition.baseline_eligible
    assert output.disposition.exception_type == expected_exception
    assert output.error_treatment == expected_treatment


def test_full_tito_scores_against_the_served_initial_prompt():
    runner = _trial_runner()
    runner._rollout_logprobs_required = True
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}, stdout="passed"),
        exception_info=None,
        agent_result=SimpleNamespace(
            metadata={
                "all_messages": [
                    {"role": "user", "content": "solve it"},
                    {"role": "assistant", "content": "done"},
                ],
                "summarization_count": 0,
                "stop_reason": "complete",
            },
            rollout_details=[
                {
                    "prompt_token_ids": [[7, 8, 2]],
                    "completion_token_ids": [[10]],
                    "logprobs": [[-0.25]],
                }
            ],
        ),
    )

    output = runner._process_trial_result(result, TrajectoryID(instance_id="task", repetition_id=0))

    assert output.evidence.prompt_token_ids == (7, 8)
    assert output.evidence.response_token_ids == (2, 10)
    assert output.loss_mask == [0, 1]
    np.testing.assert_allclose(output.evidence.behavior_logprobs, [0.0, -0.25])


@pytest.mark.asyncio
async def test_harbor_batch_retention_preserves_pass_fail_and_missing_verdicts(tmp_path):
    results = [
        SimpleNamespace(
            verifier_result=SimpleNamespace(rewards={"reward": score}, stdout="verifier output"),
            exception_info=None,
            agent_result=SimpleNamespace(
                metadata={
                    "all_messages": [
                        {"role": "user", "content": "solve it"},
                        {"role": "assistant", "content": "done"},
                    ],
                    "summarization_count": 0,
                    "stop_reason": "complete",
                },
                rollout_details=[
                    {"prompt_token_ids": [[7, 8, 2]], "completion_token_ids": [[10]], "logprobs": [[-0.25]]}
                ],
            ),
        )
        for score in (1.0, 0.0)
    ]
    results.append(SimpleNamespace(verifier_result=None, exception_info=None, agent_result=None))

    class CompletedOrchestrator:
        async def submit_batch(self, trial_configs):
            futures = []
            for result in results:
                future = asyncio.get_running_loop().create_future()
                future.set_result(result)
                futures.append(future)
            return futures

    runner = _trial_runner()
    runner._orchestrator = CompletedOrchestrator()
    runner._orchestrator_started = True
    runner._eval_session_active = False
    runner._packed_task_materializer = harbor_runner_module.PackedTaskMaterializer(tmp_path / "tasks")
    runner._harbor_config_builder = HarborConfigBuilder(OmegaConf.create({"harbor": {"name": "pi"}}))
    runner._agent_api_base = "http://localhost:8000/v1"
    runner._tracked_exceptions = []
    runner._literal_log_path = None
    runner._tito_full = True
    runner._collect_rollout_details = True
    runner._tis_lcs_alert_threshold = 0.1
    runner.model_name = "model"
    runner.trials_dir = str(tmp_path / "trials")
    request = {
        "prompts": [str(tmp_path / "task")] * 3,
        "env_classes": ["bfcl"] * 3,
        "env_extras": [{}] * 3,
        "trajectory_ids": [TrajectoryID(instance_id="task", repetition_id=index) for index in range(3)],
        "batch_metadata": BatchMetadata(global_step=0, training_phase="eval"),
        "sampling_params": {"max_tokens": 16},
    }
    batch = await runner.run(request)
    config = parse_trajectory_retention_config(
        {"enabled": True, "required": True, "output_path": str(tmp_path / "retained"), "run_id": "smoke"}
    )
    records = build_trajectory_records(request, batch, config, runner.tokenizer, runner_name="harbor")

    assert [record.verification_result.status.value for record in records] == ["verified", "verified", "unavailable"]
    assert [record.verification_result.passed for record in records] == [True, False, None]
    assert [record.verification_result.score for record in records] == [1.0, 0.0, None]
    assert [record.reward.outcome for record in records] == [1.0, 0.0, 0.0]
    assert records[0].prompt.token_ids == (7, 8)
    assert records[0].response.token_ids == (2, 10)
    assert records[0].response.loss_mask == (0, 1)


@pytest.mark.parametrize(
    ("shaper", "expected_rewards", "expected_groups"),
    [
        pytest.param(None, [1.0, 0.0], 1, id="identity-aware-default"),
        pytest.param("pass_ratio", [1.0, 0.5], None, id="explicit-pass-ratio-backup"),
    ],
)
def test_harbor_runner_reward_shaping_modes(
    shaper, expected_rewards, expected_groups, verifier_test_collection_factory
):
    outputs = [
        _runner_output(
            0,
            {"uniform": "passed", "mixed": "passed"},
            1.0,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
        _runner_output(
            1,
            {"uniform": "passed", "mixed": "failed"},
            0.5,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
    ]

    metrics = _runner(shaper)._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == expected_rewards
    assert metrics.get(f"{IDENTITY_AWARE_REWARD_METRIC_PREFIX}/groups") == expected_groups


def test_identity_aware_shaping_preserves_the_downstream_truncation_penalty(verifier_test_collection_factory):
    outputs = [
        _runner_output(
            0,
            {"mixed": "passed"},
            0.75,
            verifier_test_collection_factory=verifier_test_collection_factory,
            truncation_penalized=True,
        ),
        _runner_output(
            1,
            {"mixed": "failed"},
            0.0,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
    ]

    _runner()._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [0.75, 0.0]


def test_unrecognized_verifier_output_is_binned_as_a_verifier_failure():
    runner = object.__new__(HarborTrajectoryRunner)
    runner._error_handling_config = ErrorHandlingConfig(enable_error_classification=True)
    runner._reward_shaping_enabled = True
    runner._collect_rollout_details = False
    runner._reward_shaping_config = {
        "enable_reward_shaping": True,
        "reward_shaper": "threshold",
        "reward_shaping_fallback": False,
    }
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 0.0}, stdout="unrecognized verifier output"),
        exception_info=None,
        agent_result=SimpleNamespace(
            metadata={
                "all_messages": [
                    {"role": "user", "content": "solve it"},
                    {"role": "assistant", "content": "done"},
                ],
                "summarization_count": 0,
            }
        ),
    )

    with pytest.raises(VerifierOutputParseError):
        runner._process_trial_result(result, TrajectoryID(instance_id="task", repetition_id=0))

    treatment, exception_type = runner._classify_exception(VerifierOutputParseError("unrecognized output"))
    assert treatment == "mask"
    assert exception_type == "VerifierOutputParseError"


@pytest.mark.parametrize("recorded", [True, False])
@pytest.mark.parametrize("timed_out", [True, False])
def test_opencode_preserves_recorded_evidence_without_verifier_reward(tmp_path, recorded, timed_out):
    runner = _trial_runner()
    runner._preserve_logprobs_on_timeout = True
    runner._collect_rollout_details = True
    runner._rollout_logprobs_required = True
    runner._literal_log_store = LiteralLogStore()
    log = tmp_path / "literal.jsonl"
    entry = {
        "timestamp": 1.0,
        "status_code": 200,
        "trial_id": "timed-out-trial",
        "request": {
            "messages": [{"role": "user", "content": "solve it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        },
        "literal": {"prompt_token_ids": [7, 8, 2], "completion_token_ids": [10], "logprobs": [-0.25]},
    }
    log.write_text(json.dumps(entry) + "\n" if recorded else "")
    runner._literal_log_path = str(log)
    result = SimpleNamespace(
        agent_info=SimpleNamespace(name="opencode"),
        verifier_result=None,
        exception_info=SimpleNamespace(exception_type="AgentTimeoutError") if timed_out else None,
        agent_result=SimpleNamespace(
            metadata={"rollout_correlation_id": "timed-out-trial", "stop_reason": "timeout"},
            rollout_details=None,
        ),
    )
    output = runner._process_trial_result(result, TrajectoryID(instance_id="task", repetition_id=0))

    expected_eligible = recorded and timed_out
    assert output.disposition.loss_eligible is expected_eligible
    assert output.reward_result.optimization_reward == 0.0
    if expected_eligible:
        assert output.evidence.prompt_token_ids == (7, 8)
        assert output.evidence.response_token_ids == (2, 10)
        assert output.loss_mask == [0, 1]
        np.testing.assert_allclose(output.evidence.behavior_logprobs, [0.0, -0.25])
    else:
        assert output.loss_mask == [0]
