"""Row-routed environment for the released Nemotron 3 Ultra RLVR blends."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

import reasoning_gym
from omegaconf import DictConfig
from reasoning_gym.utils import extract_answer

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.nemotron_ultra.calendar import grade_calendar
from skyrl_gym.envs.nemotron_ultra.code_gen import grade_code
from skyrl_gym.envs.nemotron_ultra.format_verification import grade_format
from skyrl_gym.envs.nemotron_ultra.instruction_following import grade_instruction_following
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention, grade_multichallenge
from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.envs.nemotron_ultra.nvarc import grade_nvarc
from skyrl_gym.envs.nemotron_ultra.ns_tools import execute_python_calls
from skyrl_gym.envs.nemotron_ultra.rdkit_chemistry import grade_rdkit_chemistry
from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action
from skyrl_gym.verification import RolloutEvidence, VerificationResult

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


def _extract_reasoning_gym_answer(text: str) -> str:
    answer = extract_answer(text, tag_name="answer")
    if answer is not None:
        return answer
    if match := re.search(r"\\boxed\{([^}]+)\}", text):
        return match.group(1).strip()
    return text.strip()


class NemotronUltraEnv(BaseTextEnv):
    """Select the NVIDIA-compatible verifier declared by each dataset row."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
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
        self.record = self._decode_mapping(ultra.get("record_json"), "record_json")
        self.request = self._decode_mapping(ultra.get("request_json"), "request_json")
        self.evidence: RolloutEvidence | None = None
        sandbox_config = env_config.get("sandbox", {})
        self.sandbox = SandboxClient(
            host=str(sandbox_config.get("host", "127.0.0.1")),
            port=int(sandbox_config.get("port", 6000)),
        )
        self.sandbox_session_id = str(uuid.uuid4())
        self.genrm_config = dict(env_config.get("genrm", {}))
        if self.agent in {"genrm_simple_agent", "genrm_simple_agent_reasoning_off"}:
            self.max_turns = 1
        elif self.agent == "ns_tools_simple_agent":
            self.max_turns = 50
        elif self.agent == "math_formal_lean_refinement_agent":
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

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.evidence = evidence

    def init(self, prompt):
        return prompt, {"chat_completion_params": self.request}

    def _assistant_message(self, action: str) -> dict[str, Any]:
        if self.evidence is not None:
            message = self.evidence.metadata.get("assistant_message")
            if isinstance(message, Mapping):
                return dict(message)
        return {"role": "assistant", "content": action, "tool_calls": []}

    def _require_general_judge(self) -> OpenAIJudge:
        if self.general_judge is None:
            raise RuntimeError(f"Nemotron Ultra verifier {self.agent!r} requires the general judge")
        return self.general_judge

    def _require_safety_judge(self) -> OpenAIJudge:
        if self.safety_judge is None:
            raise RuntimeError(f"Nemotron Ultra verifier {self.agent!r} requires the safety judge")
        return self.safety_judge

    def step(self, action: str) -> BaseTextEnvStepOutput:
        diagnostics: dict[str, Any] = {"agent": self.agent}
        self.turns += 1
        if self.agent in {"genrm_simple_agent", "genrm_simple_agent_reasoning_off"}:
            # Replaced cohort-wise by SkyRLGymTrajectoryRunner before projection.
            reward = float(self.genrm_config.get("default_score", 3.0))
            diagnostics["cohort_reward_pending"] = True
        elif self.agent == "ns_tools_simple_agent":
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
            reward, details = grade_math(action, self.record, judge=self.general_judge)
            diagnostics.update(details)
            diagnostics["max_steps_exhausted"] = observations is not None
        elif self.agent == "math_formal_lean_refinement_agent":
            reward, details, correction_prompt = verify_lean_attempt(action, self.record, sandbox=self.sandbox)
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
            threshold = 0.0 if self.agent.startswith("swe_pivot_") else 0.1
            reward, category = grade_expected_action(
                self.record["expected_action"],
                self._assistant_message(action),
                word_count_similarity_threshold=threshold,
            )
            diagnostics["category"] = category.value
        elif self.agent == "calendar_simple_agent":
            reward, reason = grade_calendar(action, self.record["exp_cal_state"])
            diagnostics["reason"] = reason
        elif self.agent in _FORMAT_AGENTS:
            reward, details = grade_format(action, self.record["verifier"])
            diagnostics.update(details)
        elif self.agent == "mcqa_simple_agent":
            reward, details = grade_mcqa(action, self.record)
            diagnostics.update(details)
        elif self.agent in _STRUCTURED_OUTPUT_AGENTS:
            reward, details = grade_structured_output(action, self.record, self._assistant_message(action))
            diagnostics.update(details)
        elif self.agent == "rdkit_chemistry_agent":
            reward, details = grade_rdkit_chemistry(action, self.record)
            diagnostics.update(details)
        elif self.agent in {"nvarc_inductive_simple_agent", "nvarc_transductive_simple_agent"}:
            reward, details = grade_nvarc(
                action,
                self.record,
                inductive=self.agent.startswith("nvarc_inductive_"),
            )
            diagnostics.update(details)
        elif self.agent == "code_gen_simple_agent":
            reward, details = grade_code(action, self.record, assistant_message=self._assistant_message(action))
            diagnostics.update(details)
        elif self.agent == "instruction_following_simple_agent":
            reward, details = grade_instruction_following(action, self.record)
            diagnostics.update(details)
        elif self.agent == "math_with_judge_simple_agent":
            reward, details = grade_math(action, self.record, judge=self.general_judge)
            diagnostics.update(details)
        elif self.agent == "abstention_simple_agent":
            reward, details = grade_abstention(action, self.record, self._require_general_judge())
            diagnostics.update(details)
        elif self.agent == "multichallenge_simple_agent":
            reward, details = grade_multichallenge(action, self.record, self._require_general_judge())
            diagnostics.update(details)
        elif self.agent in _JAILBREAK_AGENTS:
            reward, details = grade_jailbreak(action, self.record, self._require_safety_judge())
            diagnostics.update(details)
        elif self.agent == "reasoning_gym_simple_agent":
            task_name = self.record["metadata"]["source_dataset"]
            entry = {
                "question": self.record["question"],
                "answer": self.record.get("answer"),
                "metadata": self.record["metadata"],
            }
            answer = _extract_reasoning_gym_answer(action)
            try:
                reward = float(reasoning_gym.get_score_answer_fn(task_name)(answer=answer, entry=entry))
            except Exception as error:
                reward = 0.0
                diagnostics["verifier_error"] = f"{type(error).__name__}: {error}"
            diagnostics.update({"task_name": task_name, "extracted_answer": answer})
        else:
            raise NotImplementedError(f"Nemotron Ultra verifier {self.agent!r} has not been ported")

        verification = VerificationResult.verified(reward, passed=reward > 0.0, diagnostics=diagnostics)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata=diagnostics,
            verification=verification,
        )
