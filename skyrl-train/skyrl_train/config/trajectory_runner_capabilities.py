"""Pre-launch behavior-evidence contracts for trajectory runners."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig

from marinskyrl.distillation import compile_distillation_plan_from_config

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


class EntrypointOperation(StrEnum):
    TRAIN = "train"
    GENERATE = "generate"


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
    custom_template = bool(cfg.generator.chat_template.get("name_or_path"))
    sampling_params = cfg.generator.get("sampling_params") or {}
    exact_chat_requested = bool(cfg.generator.get("require_exact_chat_transport", False))
    exact_chat_requirements = (
        CapabilityRequirement(
            config_path="generator.chat_template.name_or_path",
            expected_value="a custom template",
            satisfied=custom_template,
        ),
        CapabilityRequirement(
            config_path="generator.sampling_params.logprobs",
            expected_value="an integer",
            satisfied=isinstance(sampling_params.get("logprobs"), int),
        ),
        CapabilityRequirement(
            config_path="generator.batched",
            expected_value="false",
            satisfied=not bool(cfg.generator.get("batched", False)),
        ),
    )

    def exact_chat_capabilities(runner: str) -> TrajectoryRunnerCapabilities:
        return TrajectoryRunnerCapabilities(
            runner=runner,
            sampled_completion=EvidenceFidelity.EXACT,
            full_context_continuation=EvidenceFidelity.EXACT,
            action_tokens=ActionTokenHandling.RUNTIME_VALIDATED,
            requirements=exact_chat_requirements,
        )

    if mode is TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM and exact_chat_requested:
        return exact_chat_capabilities("fully-async SkyRL Gym exact chat")
    if mode is TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM:
        return TrajectoryRunnerCapabilities(
            runner="fully-async SkyRL Gym",
            sampled_completion=EvidenceFidelity.RETOKENIZED,
            full_context_continuation=EvidenceFidelity.UNAVAILABLE,
            action_tokens=ActionTokenHandling.RETOKENIZED,
        )

    if exact_chat_requested:
        return exact_chat_capabilities("SkyRL Gym exact chat")
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


def _validate_exact_sampled_completion(capabilities: TrajectoryRunnerCapabilities, *, consumer: str) -> None:
    action_evidence_unusable = capabilities.action_tokens in {
        ActionTokenHandling.RETOKENIZED,
        ActionTokenHandling.UNAVAILABLE,
    }
    if capabilities.sampled_completion is not EvidenceFidelity.EXACT or action_evidence_unusable:
        raise ValueError(
            f"{capabilities.runner} cannot supply exact sampled completion token IDs required by {consumer}; "
            f"resolved evidence fidelity is {capabilities.sampled_completion.value} and action-token handling is "
            f"{capabilities.action_tokens.value}"
        )

    _validate_capability_requirements(capabilities, consumer=consumer)


def _validate_capability_requirements(capabilities: TrajectoryRunnerCapabilities, *, consumer: str) -> None:
    unmet = [requirement for requirement in capabilities.requirements if not requirement.satisfied]
    if unmet:
        settings = ", ".join(f"{requirement.config_path}={requirement.expected_value}" for requirement in unmet)
        raise ValueError(f"{consumer} with {capabilities.runner} requires {settings}")


def _validate_teacher_scoreable_tokens(capabilities: TrajectoryRunnerCapabilities) -> None:
    """Accept exact or reconstructed learner tokens and reject missing token sequences."""
    if capabilities.sampled_completion is EvidenceFidelity.UNAVAILABLE or (
        capabilities.action_tokens is ActionTokenHandling.UNAVAILABLE
    ):
        raise ValueError(
            f"{capabilities.runner} cannot supply tokenized learner actions required by teacher-scored distillation"
        )
    _validate_capability_requirements(capabilities, consumer="teacher-scored distillation")


def validate_trajectory_runner_capabilities(
    cfg: DictConfig,
    mode: TrajectoryRunnerMode,
    operation: EntrypointOperation = EntrypointOperation.TRAIN,
) -> None:
    """Reject operation and runner combinations that cannot supply required training evidence."""
    # Keep launcher imports Torch-free. Importing a skyrl_train.utils submodule
    # executes that package's eager registration imports, including Torch.
    from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled  # noqa: PLC0415

    distillation_plan = compile_distillation_plan_from_config(cfg)
    capabilities = trajectory_runner_capabilities(cfg, mode)
    if cfg.generator.get("require_exact_chat_transport", False):
        _validate_capability_requirements(capabilities, consumer="exact structured-chat transport")
    if distillation_plan is not None:
        if operation is not EntrypointOperation.TRAIN:
            raise ValueError("teacher-scored distillation is training-only and cannot be configured for generation")
        _validate_teacher_scoreable_tokens(capabilities)

    algorithm = cfg.trainer.algorithm
    behavior_logprobs_required = rollout_logprobs_enabled(algorithm)
    full_tito_required = bool(algorithm.get("tito_full", False))
    if not behavior_logprobs_required and not full_tito_required:
        return

    _validate_exact_sampled_completion(capabilities, consumer="behavior-policy evidence")
    if full_tito_required and capabilities.full_context_continuation is not EvidenceFidelity.EXACT:
        raise ValueError(
            f"{capabilities.runner} does not support exact full-context continuation required by "
            "trainer.algorithm.tito_full=true"
        )
