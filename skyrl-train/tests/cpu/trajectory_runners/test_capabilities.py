import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import (
    SUPPORTED_OPENCODE_LITERAL_VERSION,
    EntrypointOperation,
    TrajectoryRunnerMode,
    validate_trajectory_runner_capabilities,
)

GYM = TrajectoryRunnerMode.SKYRL_GYM
HARBOR = TrajectoryRunnerMode.HARBOR
MINI_SWE = TrajectoryRunnerMode.MINI_SWE
HARBOR_KEY = "terminal_bench_config.harbor"
NO_ROLLOUT_LOGPROBS = {"trainer.algorithm.off_policy_correction": "none"}
FULL_TITO = {"trainer.algorithm.off_policy_correction": "none", "trainer.algorithm.tito_full": True}
OPENCODE = {f"{HARBOR_KEY}.version": SUPPORTED_OPENCODE_LITERAL_VERSION}
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
    pytest.param(HARBOR, "pi", FULL_TITO, False, id="full-tito-harbor-pi"),
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
                **NO_TIS,
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
    pytest.param(HARBOR, "codex", {}, False, "train", "Harbor codex cannot supply exact", id="harbor-codex"),
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
        HARBOR, "codex", NO_ROLLOUT_LOGPROBS, True, "train", "tokenized learner actions", id="distill-harbor-codex"
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
