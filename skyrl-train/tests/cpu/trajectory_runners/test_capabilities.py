import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import (
    TrajectoryRunnerMode,
    validate_trajectory_runner_capabilities,
)


def _harbor_config(agent_name, **harbor_overrides):
    harbor = {"name": agent_name, "collect_rollout_details": True, **harbor_overrides}
    return OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "use_tis": True,
                    "policy_loss_type": "regular",
                    "tito_full": None,
                }
            },
            "terminal_bench_config": {"harbor": harbor},
        }
    )


def _skyrl_config(*, use_tis=True, policy_loss_type="regular"):
    return OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "use_tis": use_tis,
                    "policy_loss_type": policy_loss_type,
                    "tito_full": None,
                },
                "step_wise_training": False,
            },
            "generator": {
                "use_conversation_multi_turn": True,
                "chat_template": {"source": "name", "name_or_path": None},
            },
        }
    )


@pytest.mark.parametrize("agent_name", ["pi", "codex", "claude-code", "future-agent"])
def test_harbor_behavior_logprobs_reject_unsupported_agents(agent_name):
    with pytest.raises(ValueError, match="cannot supply exact sampled completion"):
        validate_trajectory_runner_capabilities(_harbor_config(agent_name), TrajectoryRunnerMode.HARBOR)


@pytest.mark.parametrize("agent_name", ["terminus-2", "terminus_2"])
def test_harbor_behavior_logprobs_accept_terminus_with_rollout_details(agent_name):
    validate_trajectory_runner_capabilities(_harbor_config(agent_name), TrajectoryRunnerMode.HARBOR)


def test_harbor_behavior_logprobs_require_rollout_details():
    with pytest.raises(ValueError, match="collect_rollout_details=true"):
        validate_trajectory_runner_capabilities(
            _harbor_config("terminus-2", collect_rollout_details=False), TrajectoryRunnerMode.HARBOR
        )


@pytest.mark.parametrize("version", [None, "latest", "1.18.1"])
def test_harbor_behavior_logprobs_require_tested_opencode_version(version):
    with pytest.raises(ValueError, match="terminal_bench.harbor.version=1.18.2"):
        validate_trajectory_runner_capabilities(
            _harbor_config("opencode", version=version), TrajectoryRunnerMode.HARBOR
        )


def test_harbor_behavior_logprobs_accept_tested_opencode_bridge():
    validate_trajectory_runner_capabilities(_harbor_config("opencode", version="1.18.2"), TrajectoryRunnerMode.HARBOR)


@pytest.mark.parametrize(
    ("mode", "expected_runner"),
    [
        (TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM, "fully-async SkyRL Gym"),
        (TrajectoryRunnerMode.MINI_SWE, "mini-swe"),
    ],
)
@pytest.mark.parametrize(
    ("use_tis", "policy_loss_type"),
    [(True, "regular"), (False, "behavior_clip")],
)
def test_behavior_logprobs_reject_runners_without_exact_evidence(mode, expected_runner, use_tis, policy_loss_type):
    cfg = _skyrl_config(use_tis=use_tis, policy_loss_type=policy_loss_type)

    with pytest.raises(ValueError, match=expected_runner):
        validate_trajectory_runner_capabilities(cfg, mode)


@pytest.mark.parametrize(
    ("step_wise_training", "use_tis", "policy_loss_type"),
    [(False, True, "regular"), (True, True, "regular"), (False, False, "behavior_clip")],
)
def test_behavior_logprobs_accept_exact_skyrl_gym_paths(step_wise_training, use_tis, policy_loss_type):
    cfg = _skyrl_config(use_tis=use_tis, policy_loss_type=policy_loss_type)
    cfg.trainer.step_wise_training = step_wise_training

    validate_trajectory_runner_capabilities(cfg, TrajectoryRunnerMode.SKYRL_GYM)


def test_behavior_logprobs_reject_multiturn_custom_template_retokenization():
    cfg = _skyrl_config()
    cfg.generator.chat_template.name_or_path = "qwen3"

    with pytest.raises(ValueError, match="resolved evidence fidelity is retokenized"):
        validate_trajectory_runner_capabilities(cfg, TrajectoryRunnerMode.SKYRL_GYM)


@pytest.mark.parametrize(
    ("mode", "agent_name", "version", "expected_runner"),
    [
        (TrajectoryRunnerMode.SKYRL_GYM, None, None, "SkyRL Gym"),
        (TrajectoryRunnerMode.HARBOR, "opencode", "1.18.2", "Harbor opencode"),
    ],
)
def test_explicit_full_tito_rejects_runners_without_exact_continuation(mode, agent_name, version, expected_runner):
    cfg = _skyrl_config() if agent_name is None else _harbor_config(agent_name, version=version)
    cfg.trainer.algorithm.use_tis = False
    cfg.trainer.algorithm.tito_full = True

    with pytest.raises(ValueError, match=expected_runner):
        validate_trajectory_runner_capabilities(cfg, mode)


def test_explicit_full_tito_accepts_terminus_exact_continuation():
    cfg = _harbor_config("terminus-2")
    cfg.trainer.algorithm.use_tis = False
    cfg.trainer.algorithm.tito_full = True

    validate_trajectory_runner_capabilities(cfg, TrajectoryRunnerMode.HARBOR)
