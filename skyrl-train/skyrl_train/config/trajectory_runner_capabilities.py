"""Pre-launch behavior-evidence contracts for trajectory runners."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig

from marinskyrl.harbor_agent_names import (
    DEFAULT_HARBOR_AGENT_NAME,
    OPENCODE_HARBOR_AGENT_NAME,
    PI_HARBOR_AGENT_NAME,
    TERMINUS_KIRA_HARBOR_AGENT_NAME,
)

SUPPORTED_OPENCODE_LITERAL_VERSION = "1.18.2"
SUPPORTED_PI_THINKING_FORMATS = frozenset({"chat-template", "qwen-chat-template"})


class TrajectoryRunnerMode(StrEnum):
    SKYRL_GYM = "skyrl_gym"
    FULLY_ASYNC_SKYRL_GYM = "fully_async_skyrl_gym"
    MINI_SWE = "mini_swe"
    HARBOR = "harbor"


class EvidenceFidelity(StrEnum):
    EXACT = "exact"
    RETOKENIZED = "retokenized"
    UNAVAILABLE = "unavailable"


class ActionTokenHandling(StrEnum):
    EXACT = "exact"
    RUNTIME_VALIDATED = "runtime_validated"
    RETOKENIZED = "retokenized"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CapabilityRequirement:
    config_path: str
    expected_value: str
    satisfied: bool


@dataclass(frozen=True)
class TrajectoryRunnerCapabilities:
    runner: str
    sampled_completion: EvidenceFidelity
    full_context_continuation: EvidenceFidelity
    action_tokens: ActionTokenHandling
    requirements: tuple[CapabilityRequirement, ...] = ()


@dataclass(frozen=True)
class _HarborEvidenceProfile:
    sampled_completion: EvidenceFidelity
    full_context_continuation: EvidenceFidelity
    action_tokens: ActionTokenHandling


_EXACT_HARBOR_EVIDENCE = _HarborEvidenceProfile(
    sampled_completion=EvidenceFidelity.EXACT,
    full_context_continuation=EvidenceFidelity.EXACT,
    action_tokens=ActionTokenHandling.EXACT,
)
_EXACT_COMPLETION_ONLY_HARBOR_EVIDENCE = _HarborEvidenceProfile(
    sampled_completion=EvidenceFidelity.EXACT,
    full_context_continuation=EvidenceFidelity.UNAVAILABLE,
    action_tokens=ActionTokenHandling.EXACT,
)
_HARBOR_EVIDENCE_PROFILES = {
    DEFAULT_HARBOR_AGENT_NAME: _EXACT_HARBOR_EVIDENCE,
    # Native tool results change the structured chat history between model
    # turns. Harbor captures each sampled completion exactly, but currently
    # re-renders the next prompt after those tool messages instead of extending
    # the prior literal token prefix.
    TERMINUS_KIRA_HARBOR_AGENT_NAME: _EXACT_COMPLETION_ONLY_HARBOR_EVIDENCE,
    OPENCODE_HARBOR_AGENT_NAME: _EXACT_HARBOR_EVIDENCE,
    PI_HARBOR_AGENT_NAME: _EXACT_HARBOR_EVIDENCE,
}


def _terminal_bench_harbor_config(cfg: DictConfig) -> DictConfig | None:
    terminal_bench = cfg.get("terminal_bench_config")
    if terminal_bench is None and str(cfg.get("entrypoint", "")) == "terminal_bench":
        terminal_bench = cfg.get("terminal_bench")
    return terminal_bench.get("harbor") if terminal_bench is not None else None


def opencode_exact_continuation_enabled(cfg: DictConfig) -> bool:
    """Whether this launch needs the terminal-bench OpenCode continuation bridge."""
    harbor = _terminal_bench_harbor_config(cfg)
    if harbor is None:
        return False
    agent_name = str(harbor.get("name", DEFAULT_HARBOR_AGENT_NAME)).strip().lower().replace("_", "-")
    return bool(
        str(cfg.get("generator", {}).get("backend", "")) == "vllm"
        and agent_name == OPENCODE_HARBOR_AGENT_NAME
        and harbor.get("collect_rollout_details", False)
    )


def _harbor_capabilities(cfg: DictConfig) -> TrajectoryRunnerCapabilities:
    harbor = _terminal_bench_harbor_config(cfg)
    if harbor is None:
        return TrajectoryRunnerCapabilities(
            runner="harbor (unconfigured)",
            sampled_completion=EvidenceFidelity.UNAVAILABLE,
            full_context_continuation=EvidenceFidelity.UNAVAILABLE,
            action_tokens=ActionTokenHandling.UNAVAILABLE,
        )

    agent_name = str(harbor.get("name", DEFAULT_HARBOR_AGENT_NAME)).strip().lower().replace("_", "-")
    rollout_details = CapabilityRequirement(
        config_path="terminal_bench.harbor.collect_rollout_details",
        expected_value="true",
        satisfied=bool(harbor.get("collect_rollout_details", False)),
    )
    requirements = [rollout_details]
    if agent_name == OPENCODE_HARBOR_AGENT_NAME:
        requirements.extend(
            (
                CapabilityRequirement(
                    config_path="terminal_bench.harbor.version",
                    expected_value=SUPPORTED_OPENCODE_LITERAL_VERSION,
                    satisfied=str(harbor.get("version", "")).strip() == SUPPORTED_OPENCODE_LITERAL_VERSION,
                ),
                CapabilityRequirement(
                    config_path="generator.backend",
                    expected_value="vllm",
                    satisfied=str(cfg.get("generator", {}).get("backend", "")) == "vllm",
                ),
            )
        )
    elif agent_name == PI_HARBOR_AGENT_NAME:
        requirements.append(
            CapabilityRequirement(
                config_path="terminal_bench.harbor.thinking_format",
                expected_value=" or ".join(sorted(SUPPORTED_PI_THINKING_FORMATS)),
                satisfied=str(harbor.get("thinking_format", "")).strip() in SUPPORTED_PI_THINKING_FORMATS,
            )
        )

    profile = _HARBOR_EVIDENCE_PROFILES.get(agent_name)
    if profile is not None:
        return TrajectoryRunnerCapabilities(
            runner=f"Harbor {agent_name}",
            sampled_completion=profile.sampled_completion,
            full_context_continuation=profile.full_context_continuation,
            action_tokens=profile.action_tokens,
            requirements=tuple(requirements),
        )
    return TrajectoryRunnerCapabilities(
        runner=f"Harbor {agent_name}",
        sampled_completion=EvidenceFidelity.UNAVAILABLE,
        full_context_continuation=EvidenceFidelity.UNAVAILABLE,
        action_tokens=ActionTokenHandling.UNAVAILABLE,
    )


def trajectory_runner_capabilities(cfg: DictConfig, mode: TrajectoryRunnerMode) -> TrajectoryRunnerCapabilities:
    """Resolve the evidence contract for the selected runner and configuration."""
    if mode is TrajectoryRunnerMode.HARBOR:
        return _harbor_capabilities(cfg)
    if mode is TrajectoryRunnerMode.MINI_SWE:
        return TrajectoryRunnerCapabilities(
            runner="mini-swe",
            sampled_completion=EvidenceFidelity.UNAVAILABLE,
            full_context_continuation=EvidenceFidelity.UNAVAILABLE,
            action_tokens=ActionTokenHandling.RETOKENIZED,
        )
    if mode is TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM:
        return TrajectoryRunnerCapabilities(
            runner="fully-async SkyRL Gym",
            sampled_completion=EvidenceFidelity.RETOKENIZED,
            full_context_continuation=EvidenceFidelity.UNAVAILABLE,
            action_tokens=ActionTokenHandling.RETOKENIZED,
        )

    custom_template = bool(cfg.generator.chat_template.get("name_or_path"))
    if cfg.generator.use_conversation_multi_turn and custom_template:
        return TrajectoryRunnerCapabilities(
            runner="SkyRL Gym custom-template multi-turn",
            sampled_completion=EvidenceFidelity.RETOKENIZED,
            full_context_continuation=EvidenceFidelity.UNAVAILABLE,
            action_tokens=ActionTokenHandling.RETOKENIZED,
        )
    if cfg.trainer.step_wise_training:
        return TrajectoryRunnerCapabilities(
            runner="step-wise SkyRL Gym",
            sampled_completion=EvidenceFidelity.EXACT,
            full_context_continuation=EvidenceFidelity.UNAVAILABLE,
            action_tokens=ActionTokenHandling.EXACT,
        )
    return TrajectoryRunnerCapabilities(
        runner="SkyRL Gym",
        sampled_completion=EvidenceFidelity.EXACT,
        full_context_continuation=EvidenceFidelity.UNAVAILABLE,
        action_tokens=ActionTokenHandling.RUNTIME_VALIDATED,
    )


def validate_trajectory_runner_capabilities(cfg: DictConfig, mode: TrajectoryRunnerMode) -> None:
    """Reject objectives whose selected runner cannot support their evidence contract."""
    # Keep launcher imports Torch-free. Importing a skyrl_train.utils submodule
    # executes that package's eager registration imports, including Torch.
    from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled  # noqa: PLC0415

    algorithm = cfg.trainer.algorithm
    behavior_logprobs_required = rollout_logprobs_enabled(algorithm)
    full_tito_required = bool(algorithm.get("tito_full", False))
    if not behavior_logprobs_required and not full_tito_required:
        return

    capabilities = trajectory_runner_capabilities(cfg, mode)
    action_evidence_unusable = capabilities.action_tokens in {
        ActionTokenHandling.RETOKENIZED,
        ActionTokenHandling.UNAVAILABLE,
    }
    if capabilities.sampled_completion is not EvidenceFidelity.EXACT or action_evidence_unusable:
        raise ValueError(
            f"{capabilities.runner} cannot supply exact sampled completion token IDs and logprobs; "
            f"resolved evidence fidelity is {capabilities.sampled_completion.value} and action-token handling is "
            f"{capabilities.action_tokens.value}"
        )
    if full_tito_required and capabilities.full_context_continuation is not EvidenceFidelity.EXACT:
        raise ValueError(
            f"{capabilities.runner} does not support exact full-context continuation required by "
            "trainer.algorithm.tito_full=true"
        )

    unmet = [requirement for requirement in capabilities.requirements if not requirement.satisfied]
    if unmet:
        settings = ", ".join(f"{requirement.config_path}={requirement.expected_value}" for requirement in unmet)
        raise ValueError(f"Behavior-policy evidence with {capabilities.runner} requires {settings}")
