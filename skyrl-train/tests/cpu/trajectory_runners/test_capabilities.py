import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import (
    SUPPORTED_MINI_SWE_LITERAL_VERSION,
    SUPPORTED_OPENCODE_LITERAL_VERSION,
    EntrypointOperation,
    TrajectoryRunnerMode,
    harbor_exact_continuation_enabled,
    validate_trajectory_runner_capabilities,
)
from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder

GYM = TrajectoryRunnerMode.SKYRL_GYM
HARBOR = TrajectoryRunnerMode.HARBOR
MINI_SWE = TrajectoryRunnerMode.MINI_SWE
HARBOR_KEY = "terminal_bench_config.harbor"
NO_ROLLOUT_LOGPROBS = {"trainer.algorithm.off_policy_correction": "none"}
FULL_TITO = {"trainer.algorithm.off_policy_correction": "none", "trainer.algorithm.tito_full": True}
OPENCODE = {f"{HARBOR_KEY}.version": SUPPORTED_OPENCODE_LITERAL_VERSION}
CLAUDE_CODE = {f"{HARBOR_KEY}.version": "2.1.284"}
CODEX = {f"{HARBOR_KEY}.version": "0.118.0"}
HARBOR_MINI_SWE = {f"{HARBOR_KEY}.version": SUPPORTED_MINI_SWE_LITERAL_VERSION}
EXACT_CHAT = {
    "generator.chat_template.name_or_path": "qwen2_5_with_generation_tag_simplified",
    "generator.require_exact_chat_transport": True,
}


def _harbor_config(agent_name):
    harbor = {
        "name": agent_name,
        "collect_rollout_details": True,
        **({"thinking_format": "qwen-chat-template"} if agent_name == "pi" else {}),
    }
    return OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "off_policy_correction": "tis",
                    "off_policy_correction_rules": None,
                    "policy_loss_type": "regular",
                    "tito_full": None,
                },
                "placement": {"colocate_all": True},
            },
            "terminal_bench_config": {"harbor": harbor},
            "generator": {"backend": "vllm"},
        }
    )


def _skyrl_config():
    return OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "off_policy_correction": "tis",
                    "off_policy_correction_rules": None,
                    "policy_loss_type": "regular",
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


def test_harbor_panel_validates_each_harness_before_training():
    cfg = _harbor_config("pi")
    cfg.terminal_bench_config.harbor.collect_rollout_details = False
    cfg.terminal_bench_config.harbor.agent_profiles = [
        {"name": "pi", "collect_rollout_details": True},
        {"name": "opencode", "version": SUPPORTED_OPENCODE_LITERAL_VERSION, "collect_rollout_details": True},
    ]
    validate_trajectory_runner_capabilities(cfg, HARBOR)
    assert harbor_exact_continuation_enabled(cfg)
    assert HarborConfigBuilder(cfg.terminal_bench_config).get_collect_rollout_details()

    cfg.terminal_bench_config.harbor.agent_profiles.append({"name": "codex"})
    with pytest.raises(ValueError, match=r"agent_profiles\[2\].version"):
        validate_trajectory_runner_capabilities(cfg, HARBOR)


def test_harbor_panel_rejects_missing_evidence_in_one_profile():
    cfg = _harbor_config("pi")
    cfg.terminal_bench_config.harbor.agent_profiles = [
        {"name": "pi"},
        {"name": "opencode", "version": SUPPORTED_OPENCODE_LITERAL_VERSION, "collect_rollout_details": False},
    ]
    with pytest.raises(ValueError, match=r"agent_profiles\[1\].collect_rollout_details"):
        validate_trajectory_runner_capabilities(cfg, HARBOR)


def _config(agent_name, overrides, distillation, local_distillation_config):
    cfg = _skyrl_config() if agent_name is None else _harbor_config(agent_name)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return local_distillation_config(cfg) if distillation else cfg


# (mode, harbor agent or None for SkyRL Gym, dotted config overrides, distillation)
ACCEPTED = [
    pytest.param(HARBOR, "terminus-2", {}, False, id="harbor-terminus-2"),
    pytest.param(HARBOR, "terminus_kira", {}, False, id="harbor-terminus-kira-underscore-alias"),
    pytest.param(HARBOR, "opencode", OPENCODE, False, id="harbor-opencode-tested-version"),
    pytest.param(HARBOR, "mini-swe-agent", HARBOR_MINI_SWE, False, id="harbor-mini-swe-tested-version"),
    pytest.param(HARBOR, "pi", {}, False, id="harbor-pi"),
    pytest.param(GYM, None, {}, False, id="gym-tis"),
    pytest.param(GYM, None, {"trainer.step_wise_training": True}, False, id="gym-step-wise"),
    pytest.param(GYM, None, {"trainer.algorithm.policy_loss_type": "behavior_clip"}, False, id="gym-clip"),
    pytest.param(
        GYM,
        None,
        {**EXACT_CHAT, "trainer.algorithm.tito_full": True, "generator.sampling_params": {"logprobs": 0}},
        False,
        id="gym-full-tito-exact-chat",
    ),
    pytest.param(GYM, None, NO_ROLLOUT_LOGPROBS, True, id="distill-gym"),
    pytest.param(HARBOR, "terminus-2", NO_ROLLOUT_LOGPROBS, True, id="distill-harbor-terminus-2"),
    pytest.param(HARBOR, "opencode", {**NO_ROLLOUT_LOGPROBS, **OPENCODE}, True, id="distill-harbor-opencode"),
    pytest.param(HARBOR, "pi", NO_ROLLOUT_LOGPROBS, True, id="distill-harbor-pi"),
    pytest.param(
        HARBOR,
        "terminus-2",
        {**NO_ROLLOUT_LOGPROBS, "trainer.placement.colocate_all": False},
        True,
        id="distill-harbor-separate-placement",
    ),
    pytest.param(HARBOR, "terminus-2", FULL_TITO, False, id="full-tito-harbor-terminus-2"),
    pytest.param(HARBOR, "opencode", {**FULL_TITO, **OPENCODE}, False, id="full-tito-harbor-opencode"),
    pytest.param(HARBOR, "mini-swe-agent", {**FULL_TITO, **HARBOR_MINI_SWE}, False, id="full-tito-harbor-mini-swe"),
    pytest.param(HARBOR, "pi", FULL_TITO, False, id="full-tito-harbor-pi"),
    pytest.param(HARBOR, "claude-code", {**FULL_TITO, **CLAUDE_CODE}, False, id="full-tito-native-claude-code"),
    pytest.param(HARBOR, "codex", {**FULL_TITO, **CODEX}, False, id="full-tito-native-codex"),
]


@pytest.mark.parametrize(("mode", "agent_name", "overrides", "distillation"), ACCEPTED)
def test_runner_capabilities_accept(mode, agent_name, overrides, distillation, local_distillation_config):
    cfg = _config(agent_name, overrides, distillation, local_distillation_config)

    validate_trajectory_runner_capabilities(cfg, mode)


# (mode, harbor agent or None, overrides, distillation, operation, match naming the runner or config field)
REJECTED = [
    *(
        pytest.param(
            mode,
            None,
            {
                **NO_ROLLOUT_LOGPROBS,
                "trainer.callbacks": [
                    {
                        "type": "evaluation",
                        "additional_evaluations": {"sampled": {"sampling_params": {"temperature": 1.0}}},
                    }
                ],
            },
            False,
            "train",
            "does not support additional evaluation sampling profiles",
            id=f"{mode.value}-sampling-profiles",
        )
        for mode in (HARBOR, MINI_SWE)
    ),
    *(
        pytest.param(HARBOR, agent, overrides, False, "train", field, id=f"native-{agent}-{case}")
        for agent, pinned in (("claude-code", CLAUDE_CODE), ("codex", CODEX))
        for case, overrides, field in (
            ("unpinned", {}, "terminal_bench.harbor.version"),
            ("latest", {f"{HARBOR_KEY}.version": "latest"}, "terminal_bench.harbor.version"),
            ("sglang", {**pinned, "generator.backend": "sglang"}, "generator.backend"),
            ("no-details", {**pinned, f"{HARBOR_KEY}.collect_rollout_details": False}, "collect_rollout_details"),
        )
    ),
    pytest.param(HARBOR, "future-agent", {}, False, "train", "Harbor future-agent", id="harbor-unknown-agent"),
    *(
        pytest.param(
            HARBOR,
            agent,
            {f"{HARBOR_KEY}.collect_rollout_details": False},
            False,
            "train",
            "terminal_bench.harbor.collect_rollout_details",
            id=f"harbor-{agent}-no-rollout-details",
        )
        for agent in ("terminus-2", "pi")
    ),
    *(
        pytest.param(
            HARBOR,
            "opencode",
            {f"{HARBOR_KEY}.version": version},
            False,
            "train",
            "terminal_bench.harbor.version",
            id=f"harbor-opencode-version-{version}",
        )
        for version in (None, "latest", "1.18.1")
    ),
    pytest.param(
        HARBOR,
        "opencode",
        {**OPENCODE, "generator.backend": "sglang"},
        False,
        "train",
        "generator.backend",
        id="harbor-opencode-sglang",
    ),
    pytest.param(
        HARBOR,
        "mini-swe-agent",
        {**HARBOR_MINI_SWE, "generator.backend": "sglang"},
        False,
        "train",
        "generator.backend",
        id="harbor-mini-swe-sglang",
    ),
    pytest.param(
        HARBOR,
        "mini-swe-agent",
        {f"{HARBOR_KEY}.version": "1.0.0"},
        False,
        "train",
        "terminal_bench.harbor.version",
        id="harbor-mini-swe-legacy-transport",
    ),
    *(
        pytest.param(
            HARBOR,
            "pi",
            {f"{HARBOR_KEY}.thinking_format": fmt},
            False,
            "train",
            "terminal_bench.harbor.thinking_format",
            id=f"harbor-pi-thinking-format-{fmt}",
        )
        for fmt in (None, "unsupported")
    ),
    pytest.param(MINI_SWE, None, {}, False, "train", "mini-swe", id="mini-swe-tis"),
    pytest.param(
        MINI_SWE,
        None,
        {"trainer.algorithm.policy_loss_type": "behavior_clip"},
        False,
        "train",
        "mini-swe",
        id="mini-swe-clip",
    ),
    pytest.param(
        GYM,
        None,
        {"generator.chat_template.name_or_path": "qwen3"},
        False,
        "train",
        "SkyRL Gym custom-template multi-turn",
        id="gym-custom-template-retokenized",
    ),
    pytest.param(
        GYM,
        None,
        {**EXACT_CHAT, **NO_ROLLOUT_LOGPROBS, "generator.sampling_params": {"logprobs": None}},
        False,
        "train",
        "generator.sampling_params.logprobs",
        id="gym-exact-chat-without-logprobs",
    ),
    pytest.param(MINI_SWE, None, NO_ROLLOUT_LOGPROBS, True, "train", "mini-swe", id="distill-mini-swe"),
    pytest.param(
        HARBOR,
        "codex",
        NO_ROLLOUT_LOGPROBS,
        True,
        "train",
        "terminal_bench.harbor.version",
        id="distill-harbor-codex-unpinned",
    ),
    pytest.param(HARBOR, "terminus-2", NO_ROLLOUT_LOGPROBS, True, "generate", "training-only", id="distill-generate"),
    pytest.param(GYM, None, FULL_TITO, False, "train", "SkyRL Gym does not support", id="full-tito-gym"),
    pytest.param(
        HARBOR, "terminus-kira", FULL_TITO, False, "train", "Harbor terminus-kira", id="full-tito-harbor-kira"
    ),
]


@pytest.mark.parametrize(("mode", "agent_name", "overrides", "distillation", "operation", "match"), REJECTED)
def test_runner_capabilities_reject(
    mode, agent_name, overrides, distillation, operation, match, local_distillation_config
):
    cfg = _config(agent_name, overrides, distillation, local_distillation_config)

    with pytest.raises(ValueError, match=match):
        validate_trajectory_runner_capabilities(cfg, mode, EntrypointOperation(operation))
