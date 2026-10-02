"""Row-routed environment for the released Nemotron 3 Ultra RLVR blends."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import reasoning_gym
import requests
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.lcb.livecodebench import DEFAULT_LIMITS, VerifierLimits
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer
from skyrl_gym.envs.nemotron_ultra.calendar import grade_calendar
from skyrl_gym.envs.nemotron_ultra.code_gen import DEFAULT_PER_TEST_TIMEOUT_SECONDS, grade_code
from skyrl_gym.envs.nemotron_ultra.format_verification import grade_format
from skyrl_gym.envs.nemotron_ultra.instruction_following import grade_instruction_following
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention, grade_multichallenge
from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.envs.nemotron_ultra.nvarc import grade_inductive_arc, grade_transductive_arc
from skyrl_gym.envs.nemotron_ultra.ns_tools import execute_python_calls
from skyrl_gym.envs.nemotron_ultra.rdkit_chemistry import grade_rdkit_chemistry
from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action
from skyrl_gym.envs.reasoning_gym.scoring import extract_answer
from skyrl_gym.verification import VERIFIER_RUNTIME_ERROR, RolloutEvidence, VerificationResult

_IPI_AGENT = "indirect_prompt_injection_simple_agent"
_NS_TOOLS_AGENT = "ns_tools_simple_agent"
_LEAN_AGENT = "math_formal_lean_refinement_agent"
_TOOL_COMPARISON_AGENTS = {
    "single_step_tool_use_with_argument_comparison_agent",
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
}
_FORMAT_AGENTS = {"citation_format_simple_agent", "freeform_formatting_simple_agent"}
_STRUCTURED_OUTPUT_AGENTS = {"structured_outputs_simple_agent", "structured_outputs_v3_simple_agent"}
_JAILBREAK_AGENTS = {
    "jailbreak_engagement_with_disclaimer",
    "jailbreak_hard_refusal_no_redirection",
    "jailbreak_hard_refusal_with_helplines",
    "jailbreak_refusal_with_explanation",
}


class NemotronUltraGrading(StrEnum):
    """Whether terminal steps run the row's verifier.

    ``skip`` keeps tool execution and turn control but returns no verdict, so rows whose
    verifier needs a judge endpoint can train under objectives that ignore the reward. Lean
    refinement rows are still verified, because the verdict decides their correction turn.
    """

    VERIFY = "verify"
    SKIP = "skip"


def _extract_reasoning_gym_answer(text: str) -> str:
    text = final_answer_text(text)
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return last_boxed_answer(text) or extract_answer(text)


class NemotronUltraEnv(BaseTextEnv):
    """Select the NVIDIA-compatible verifier declared by each dataset row."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__(env_config)
        self.tool_comparison_policy = env_config.get(
            "verifyit_tool_comparison_policy", "nemotron_strict_typed_arguments_v1"
        )
        self.math_verifier_timeout_seconds = env_config.get("verifyit_math_total_timeout_seconds", 60.0)
        self.judge_verifier_timeout_seconds = env_config.get("verifyit_judge_total_timeout_seconds", 120.0)
        self.judge_profile_policies = env_config.get("verifyit_judge_profile_policies", {})
        self.grading = NemotronUltraGrading(env_config.get("grading", NemotronUltraGrading.VERIFY))
        judges = env_config.get("judges", {})
        general_judge = judges.get("general") if isinstance(judges, Mapping) else None
        safety_judge = judges.get("safety") if isinstance(judges, Mapping) else None
        self.general_judge = OpenAIJudge(**dict(general_judge)) if isinstance(general_judge, Mapping) else None
        self.safety_judge = OpenAIJudge(**dict(safety_judge)) if isinstance(safety_judge, Mapping) else None
        extra_info = (extras or {}).get("extra_info")
        ultra = extra_info.get("nemotron_ultra") if isinstance(extra_info, Mapping) else None
        if not isinstance(ultra, Mapping):
            raise ValueError("nemotron_ultra environment requires extra_info.nemotron_ultra")
        if ultra.get("route") != "skyrl_gym":
            raise ValueError("terminal-bench Nemotron Ultra rows must not execute in the SkyRL Gym environment")
        self.agent = str(ultra["agent"])
        self.reasoning_record_json = ultra.get("record_json")
        self.record = self._decode_mapping(ultra.get("record_json"), "record_json")
        self.request_json = ultra.get("request_json")
        self.request = self._decode_mapping(self.request_json, "request_json")
        self.evidence: RolloutEvidence | None = None
        self.ipi_client = None
        self.ipi_calls = []
        self.ipi_url = env_config.get("ipi_resources_url")
        self.ipi_terminal = None
        sandbox_config = env_config.get("sandbox", {})
        self.sandbox = SandboxClient(
            host=str(sandbox_config.get("host", "127.0.0.1")),
            port=int(sandbox_config.get("port", 6000)),
        )
        self.sandbox_session_id = str(uuid.uuid4())
        self.genrm_config = dict(env_config.get("genrm", {}))
        code_verifier = env_config.get("code_verifier", {})
        if not isinstance(code_verifier, Mapping):
            code_verifier = {}
        # An explicit `null` in code_verifier config disables a bound; an unset key keeps it.
        self.code_verifier = VerifierLimits(
            max_memory_bytes=code_verifier.get("max_memory_bytes", DEFAULT_LIMITS.max_memory_bytes),
            total_timeout_seconds=code_verifier.get("total_timeout_seconds", DEFAULT_LIMITS.total_timeout_seconds),
        )
        self.code_verifier_timeout_seconds = int(
            code_verifier.get("per_test_timeout_seconds", DEFAULT_PER_TEST_TIMEOUT_SECONDS)
        )
        if self.agent in {"genrm_simple_agent", "genrm_simple_agent_reasoning_off"}:
            self.max_turns = 1
        elif self.agent == _IPI_AGENT and self.verifyit_enabled:
            self.max_turns = 5
        elif self.agent == _NS_TOOLS_AGENT:
            self.max_turns = 50
        elif self.agent == _LEAN_AGENT:
            self.max_turns = 3
        else:
            self.max_turns = 1

    @staticmethod
    def _decode_mapping(value: Any, field: str) -> dict[str, Any]:
        if not isinstance(value, str):
            raise TypeError(f"nemotron_ultra {field} must be a JSON string")
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError(f"nemotron_ultra {field} must decode to an object")
        return decoded

    def close(self) -> None:
        try:
            if self.ipi_client is not None:
                self.ipi_client.close(self.ipi_calls)
        finally:
            self.sandbox.close_session(self.sandbox_session_id)

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.evidence = evidence

    def init(self, prompt):
        return prompt, {"chat_completion_params": self.request}

    def _assistant_message(self, action: str) -> dict[str, Any]:
        if self.evidence is not None:
            message = self.evidence.metadata.get("assistant_message")
            if isinstance(message, Mapping):
                grading_message = dict(message)
                grading_message["content"] = final_answer_text(action)
                return grading_message
        return {"role": "assistant", "content": action, "tool_calls": []}

    def _grade_judge_profile(self, action: str, kind: str, judge):
        if self.verifyit_enabled:
            from skyrl_gym.envs.nemotron_ultra.judge_profiles_verifyit import grade_judge_profile_verifyit

            return grade_judge_profile_verifyit(
                action,
                self.record,
                judge,
                kind=kind,
                timeout_seconds=self.judge_verifier_timeout_seconds,
                policies=self.judge_profile_policies,
            )
        scorers = {"abstention": grade_abstention, "multichallenge": grade_multichallenge, "jailbreak": grade_jailbreak}
        return scorers[kind](action, self.record, judge)

    def _grade_math(self, action: str):
        if self.verifyit_enabled:
            from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import grade_math_verifyit

            return grade_math_verifyit(
                action,
                self.record,
                judge=self.general_judge,
                timeout_seconds=self.math_verifier_timeout_seconds,
            )
        return grade_math(action, self.record, judge=self.general_judge)

    def _require_general_judge(self) -> OpenAIJudge:
        if self.general_judge is None:
            raise RuntimeError(f"Nemotron Ultra verifier {self.agent!r} requires the general judge")
        return self.general_judge

    def _require_safety_judge(self) -> OpenAIJudge:
        if self.safety_judge is None:
            raise RuntimeError(f"Nemotron Ultra verifier {self.agent!r} requires the safety judge")
        return self.safety_judge

    def _ns_tools_turn(self, action: str, diagnostics: dict[str, Any]) -> BaseTextEnvStepOutput | None:
        """Execute the turn's Python calls; return the continuing step, or None when the rollout ends."""
        observations = execute_python_calls(
            self._assistant_message(action),
            sandbox=self.sandbox,
            session_id=self.sandbox_session_id,
        )
        if observations is not None and self.turns < self.max_turns:
            return BaseTextEnvStepOutput(
                observations=observations,
                reward=0.0,
                done=False,
                metadata={**diagnostics, "num_tool_calls": len(observations)},
                verification=VerificationResult.unavailable("tool execution is not a terminal verdict"),
            )
        diagnostics["max_steps_exhausted"] = observations is not None
        return None

    def _ipi_turn(self, action: str) -> BaseTextEnvStepOutput:
        from verifyit.grade import InvalidTask
        from verifyit.json_objects import unique_object
        from verifyit.modes.grade_json_schema import grade_json_schema_candidate

        from skyrl_gym.envs.nemotron_ultra.ipi import IPIClient, grade_trace, parse_calls

        if self.ipi_terminal is not None:
            return self.ipi_terminal
        if self.ipi_client is None:
            try:
                record = json.loads(self.reasoning_record_json, object_pairs_hook=unique_object)
                request = json.loads(self.request_json, object_pairs_hook=unique_object)
            except ValueError as error:
                raise InvalidTask("IPI trusted record/request is not unique-key JSON") from error
            if (
                grade_json_schema_candidate({"type": "array", "items": {"type": "object"}}, [record, request]).reward
                != 1
            ):
                raise InvalidTask("IPI trusted record and request must be finite objects")
            params = record.get("responses_create_params", {})
            if (
                grade_json_schema_candidate({"type": "array", "items": {"type": "object"}}, [request, params]).reward
                != 1
            ):
                raise InvalidTask("IPI trusted request metadata must be finite objects")
            shared = {key: {"const": value} for key, value in request.items() if key in params}
            if grade_json_schema_candidate({"type": "object", "properties": shared}, params).reward != 1:
                raise InvalidTask("IPI record and request metadata conflict")
            # Sky separates the request from record_json; NeMo requires its envelope, not its prompt for grading.
            record = {**record, "responses_create_params": {"input": [], **params, **request}}
            grade_trace(record, [], truncated=True)  # Validate trusted contract before tools/candidate.
            if not isinstance(self.ipi_url, str) or not self.ipi_url:
                raise RuntimeError("IPI requires ipi_resources_url pointing to the pinned NeMo service")
            self.ipi_client = IPIClient(self.ipi_url, record)
            self.ipi_client.seed()
        if self.evidence is None:
            raise RuntimeError("IPI requires structured rollout evidence")
        reason = self.evidence.stop_reason
        if reason not in {"stop", "tool_calls", "length", "max_output_tokens"}:
            raise RuntimeError("IPI completion protocol is unavailable")
        calls = parse_calls(self.evidence.metadata.get("assistant_message"))
        if len({call["call_id"] for call in [*self.ipi_calls, *calls]}) != len(self.ipi_calls) + len(calls):
            raise ValueError("IPI function identities repeat")
        self.ipi_calls.extend(calls)
        truncated = reason in {"length", "max_output_tokens"}
        if calls and not truncated and self.turns < self.max_turns:
            return BaseTextEnvStepOutput(
                observations=self.ipi_client.execute(calls),
                reward=0.0,
                done=False,
                metadata={"agent": self.agent},
                verification=VerificationResult.unavailable("tool episode is continuing"),
            )
        verdict = grade_trace(self.ipi_client.record, self.ipi_calls, truncated=truncated or bool(calls))
        self.ipi_client.close(self.ipi_calls)
        self.ipi_terminal = BaseTextEnvStepOutput(
            observations=[],
            reward=verdict.reward,
            done=True,
            metadata={"agent": self.agent},
            verification=VerificationResult.verified(verdict.reward, passed=verdict.reward == 1),
        )
        return self.ipi_terminal

    def step(self, action: str) -> BaseTextEnvStepOutput:
        raw_action = action
        if not (self.verifyit_enabled and self.agent in {"mcqa_simple_agent", "code_gen_simple_agent"}):
            action = final_answer_text(action)
        error_types = (requests.RequestException, RuntimeError, ValueError)
        invalid_task_types = ()
        preparation_error_types = ()
        if self.verifyit_enabled:
            error_types += (ImportError,)
            if self.agent in {"mcqa_simple_agent", "code_gen_simple_agent"}:
                error_types = (Exception,)
        try:
            if self.verifyit_enabled:
                from verifyit.grade import InvalidTask
                from verifyit.preparation.errors import PreparationError

                preparation_error_types = (PreparationError,)
                invalid_task_types = (InvalidTask,)
                error_types += invalid_task_types
            return self._step(action, raw_action=raw_action)
        except error_types as error:
            details = {
                "agent": self.agent,
                "error_type": VERIFIER_RUNTIME_ERROR,
                "error_category": "infrastructure",
                "cause_error_type": type(error).__name__,
                "error_message": str(error),
                "grading_action": action,
            }
            if self.verifyit_enabled and self.agent == "mcqa_simple_agent":
                logging.getLogger(__name__).exception("MCQA verification boundary failed")
                details.update(verifyit_status="infra_error", preparation_stage="mcqa_boundary")
            if self.verifyit_enabled and self.agent == "code_gen_simple_agent":
                logging.getLogger(__name__).exception("Code verification boundary failed")
                details.update(
                    verifyit_status="infra_error",
                    preparation_stage="code_boundary",
                    error_message="Code verification unavailable",
                    preparation={
                        "policy": "lcb_source_v1",
                        "response_policy": "nemotron_final_answer_text_v1",
                        "raw_response": raw_action,
                        "grading_response": action,
                    },
                )
            if isinstance(error, invalid_task_types):
                details.update(error_category="invalid_task", verifyit_status="invalid_task")
            if isinstance(error, preparation_error_types):
                details.update(error.verdict.detail)
                details.update(
                    verifyit_status=error.verdict.status.value,
                    error_category=error.failure.category.value,
                    cause_error_type=error.failure.error_type,
                    preparation_stage=error.failure.stage,
                )
            if self.verifyit_enabled and self.agent in _TOOL_COMPARISON_AGENTS:
                details["preparation"] = {
                    "policy": self.tool_comparison_policy,
                    "stage": "tool_comparison",
                }
                if isinstance(error, preparation_error_types):
                    details["preparation"].update(
                        stage=error.failure.stage, error_category=error.failure.category.value
                    )
            if self.agent == _IPI_AGENT and self.ipi_client is not None:
                try:
                    self.ipi_client.close(self.ipi_calls)
                except (requests.RequestException, RuntimeError) as cleanup_error:
                    details.update(cleanup_error=str(cleanup_error), error_category="infrastructure")
            result = BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=details,
                verification=VerificationResult.error("verifier failed", diagnostics=details),
            )

            if self.agent == _IPI_AGENT and self.verifyit_enabled:
                self.ipi_terminal = result
            return result

    def _step(self, action: str, *, raw_action: str | None = None) -> BaseTextEnvStepOutput:
        diagnostics: dict[str, Any] = {"agent": self.agent}
        self.turns += 1
        if self.agent == _IPI_AGENT and self.verifyit_enabled:
            return self._ipi_turn(action)
        if self.agent == _NS_TOOLS_AGENT:
            tool_turn = self._ns_tools_turn(action, diagnostics)
            if tool_turn is not None:
                return tool_turn
        # Lean verification decides whether a correction turn follows, so it runs in both modes.
        if self.grading is NemotronUltraGrading.SKIP and self.agent != _LEAN_AGENT:
            diagnostics["graded"] = 0.0
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=diagnostics,
                verification=VerificationResult.skipped("grading is skipped", diagnostics=diagnostics),
            )

        if self.agent in {"genrm_simple_agent", "genrm_simple_agent_reasoning_off"}:
            # Replaced cohort-wise by SkyRLGymTrajectoryRunner before projection.
            reward = float(self.genrm_config.get("default_score", 3.0))
            diagnostics["cohort_reward_pending"] = True
        elif self.agent == _NS_TOOLS_AGENT:
            reward, details = self._grade_math(action)
            diagnostics.update(details)
        elif self.agent == _LEAN_AGENT:
            reward, details, correction_prompt = verify_lean_attempt(
                action, self.record, sandbox=self.sandbox, verifyit_enabled=self.verifyit_enabled
            )
            diagnostics.update(details)
            if correction_prompt is not None and self.turns < self.max_turns:
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=False,
                    metadata=diagnostics,
                    verification=VerificationResult.unavailable("failed Lean attempt will be replaced by a correction"),
                    reset_conversation=[{"role": "user", "content": correction_prompt}],
                )
        elif self.agent in _TOOL_COMPARISON_AGENTS:
            if self.verifyit_enabled:
                from skyrl_gym.envs.nemotron_ultra.tool_comparison_verifyit import (
                    grade_prepared_tool_action,
                    prepare_tool_action,
                    structure_tool_action,
                )

                inputs = structure_tool_action(self.record["expected_action"], self._assistant_message(action))
                prepared = prepare_tool_action(
                    inputs,
                    self.tool_comparison_policy,
                )
                reward, category = grade_prepared_tool_action(prepared)
                diagnostics["preparation"] = {
                    "policy": prepared.policy.value,
                    "expected_action_sha256": hashlib.sha256(
                        json.dumps(prepared.raw.expected_action, sort_keys=True).encode()
                    ).hexdigest(),
                    "assistant_message_sha256": hashlib.sha256(
                        json.dumps(prepared.raw.assistant_message, sort_keys=True).encode()
                    ).hexdigest(),
                }
            else:
                reward, category = grade_expected_action(
                    self.record["expected_action"], self._assistant_message(action)
                )
            diagnostics["category"] = category.value
        elif self.agent == "calendar_simple_agent":
            calendar_scorer = grade_calendar
            if self.verifyit_enabled:
                from skyrl_gym.envs.nemotron_ultra.calendar_verifyit import grade_calendar_verifyit

                calendar_scorer = grade_calendar_verifyit
            reward, reason = calendar_scorer(action, self.record["exp_cal_state"])
            diagnostics["reason"] = reason
        elif self.agent in _FORMAT_AGENTS:
            format_scorer = grade_format
            if self.verifyit_enabled:
                from skyrl_gym.envs.nemotron_ultra.format_verifyit import grade_format_verifyit

                format_scorer = grade_format_verifyit
            reward, details = format_scorer(action, self.record["verifier"])
            diagnostics.update(details)
        elif self.agent == "mcqa_simple_agent":
            reward, details = grade_mcqa(action, self.record, verifyit_enabled=self.verifyit_enabled)
            diagnostics.update(details)
        elif self.agent in _STRUCTURED_OUTPUT_AGENTS:
            structured_scorer = grade_structured_output
            if self.verifyit_enabled:
                from skyrl_gym.envs.nemotron_ultra.structured_outputs_verifyit import grade_structured_output_verifyit

                structured_scorer = grade_structured_output_verifyit
            reward, details = structured_scorer(action, self.record, self._assistant_message(action))
            diagnostics.update(details)
        elif self.agent == "rdkit_chemistry_agent":
            reward, details = grade_rdkit_chemistry(action, self.record, verifyit_enabled=self.verifyit_enabled)
            diagnostics.update(details)
        elif self.agent == "nvarc_inductive_simple_agent":
            reward, details = grade_inductive_arc(
                action, self.record, sandbox=self.sandbox, verifyit_enabled=self.verifyit_enabled
            )
            diagnostics.update(details)
        elif self.agent == "nvarc_transductive_simple_agent":
            reward, details = grade_transductive_arc(action, self.record, verifyit_enabled=self.verifyit_enabled)
            diagnostics.update(details)
        elif self.agent == "code_gen_simple_agent":
            reward, details = grade_code(
                action,
                self.record,
                assistant_message=(
                    self.evidence.metadata.get("assistant_message") if self.evidence is not None else None
                )
                if self.verifyit_enabled
                else self._assistant_message(action),
                raw_response=raw_action,
                timeout_seconds=self.code_verifier_timeout_seconds,
                limits=self.code_verifier,
                verifyit_enabled=self.verifyit_enabled,
                sandbox=self.sandbox,
            )
            diagnostics.update(details)
        elif self.agent == "instruction_following_simple_agent":
            instruction_scorer = grade_instruction_following
            if self.verifyit_enabled:
                from skyrl_gym.envs.instruction_verifyit import grade_nemotron_instructions

                instruction_scorer = grade_nemotron_instructions
            reward, details = instruction_scorer(action, self.record)
            diagnostics.update(details)
        elif self.agent == "math_with_judge_simple_agent":
            reward, details = self._grade_math(action)
            diagnostics.update(details)
        elif self.agent == "abstention_simple_agent":
            reward, details = self._grade_judge_profile(
                raw_action if self.verifyit_enabled else action, "abstention", self._require_general_judge()
            )
            diagnostics.update(details)
        elif self.agent == "multichallenge_simple_agent":
            reward, details = self._grade_judge_profile(
                raw_action if self.verifyit_enabled else action, "multichallenge", self._require_general_judge()
            )
            diagnostics.update(details)
        elif self.agent in _JAILBREAK_AGENTS:
            reward, details = self._grade_judge_profile(
                raw_action if self.verifyit_enabled else action, "jailbreak", self._require_safety_judge()
            )
            diagnostics.update(details)
        elif self.agent == "reasoning_gym_simple_agent":
            record = self.record
            if self.verifyit_enabled:
                from verifyit.grade import InvalidTask
                from verifyit.json_objects import unique_object

                try:
                    record = json.loads(self.reasoning_record_json, object_pairs_hook=unique_object)
                    json.dumps(record, allow_nan=False)
                except ValueError as error:
                    raise InvalidTask("invalid Reasoning Gym record") from error
            task_name = record["metadata"]["source_dataset"]
            entry = {
                "question": record["question"],
                "answer": record.get("answer"),
                "metadata": record["metadata"],
            }
            answer = _extract_reasoning_gym_answer(action)
            if self.verifyit_enabled:
                from skyrl_gym.envs.verifyit_clients import grade_reasoning_entry

                reward = grade_reasoning_entry(task_name, entry, answer)
            else:
                reward = float(reasoning_gym.get_score_answer_fn(task_name)(answer=answer, entry=entry))
            diagnostics.update({"task_name": task_name, "extracted_answer": answer})
        else:
            raise NotImplementedError(f"Nemotron Ultra verifier {self.agent!r} has not been ported")

        if diagnostics.get("error_type") in {"schema_error", "verification_error"} or any(
            diagnostics.get("instruction_errors", [])
        ):
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=diagnostics,
                verification=VerificationResult.error("verifier configuration failed", diagnostics=diagnostics),
            )
        diagnostics["graded"] = 1.0
        verification = VerificationResult.verified(
            reward, passed=reward >= 1.0, diagnostics={**diagnostics, "grading_action": action}
        )
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata=diagnostics,
            verification=verification,
        )
