"""Regression controls for the Datakit RLVR1 verifier audit."""

import json

import pytest
import requests
from omegaconf import OmegaConf

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv, _extract_reasoning_gym_answer
from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group
from skyrl_gym.envs.nemotron_ultra.genrm_utils import GenRMOutputParseError, parse_genrm_output
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention, grade_multichallenge
from skyrl_gym.envs.nemotron_ultra.lean_proof_utils import determine_proof_status
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.envs.nemotron_ultra.nvarc import grade_nvarc, parse_grid
from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action
from skyrl_gym.verification import RolloutEvidence, VerificationStatus


class JudgeReplies:
    def __init__(self, *replies):
        self.replies = iter(replies)

    def generate(self, messages, **kwargs):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def generate_response(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


def ultra_env(agent, record):
    return NemotronUltraEnv(OmegaConf.create({}), extras={"extra_info": {"nemotron_ultra": {
        "route": "skyrl_gym", "agent": agent, "record_json": json.dumps(record), "request_json": "{}",
    }}})


@pytest.mark.parametrize("text,answer", [
    ("<think>wrong answer</think>correct answer", "correct answer"),
    ("reasoning</think>correct answer", "correct answer"),
    ("<think>the number is 20", ""),
    ("<thinking>analysis</thinking>42", "42"),
])
def test_final_answer_removes_reasoning_without_promoting_unfinished_work(text, answer):
    assert final_answer_text(text) == answer


def test_structured_grading_ignores_reasoning_and_keeps_optional_fields_optional():
    record = {"schema_type": "json", "schema_str": json.dumps({
        "type": "object", "properties": {"required": {"type": "integer"}, "optional": {"type": "string"}},
        "required": ["required"],
    })}
    result = ultra_env("structured_outputs_simple_agent", record).step('<think>not JSON</think>{"required": 7, "extra": true}')
    assert result["reward"] == 1.0
    assert grade_structured_output('{"optional":"x"}', record, {})[0] == 0.0


@pytest.mark.parametrize("reply", ['{"score_1":5}', '{"score_1":NaN,"score_2":3,"ranking":3}',
    '{"score_1":5,"score_2":Infinity,"ranking":3}', '{"score_1":6,"score_2":3,"ranking":3}',
    '{"score_1":5,"score_2":3,"ranking":0}'])
def test_genrm_requires_a_complete_finite_in_range_verdict(reply):
    with pytest.raises(GenRMOutputParseError):
        parse_genrm_output(reply)


def test_genrm_uses_final_json_instead_of_an_earlier_fenced_example():
    text = '```json\n{"score_1":5,"score_2":1,"ranking":1}\n```\n{"score_1":1,"score_2":5,"ranking":6}'
    assert parse_genrm_output(text) == (1, 5, 6)


def test_genrm_outage_cannot_create_length_shaped_positive_rewards():
    with pytest.raises(RuntimeError, match="unavailable"):
        grade_genrm_group(conversation_history=[], response_objects=[{"output": []}] * 2,
                         principle="correct", judge=JudgeReplies(requests.ConnectionError("down")),
                         config={"genrm_parse_retries": 0, "group_answer_length_penalty_coeff": 0.1})


def test_jailbreak_uses_the_standalone_final_label_and_retains_judge_output():
    reward, details = grade_jailbreak("candidate", {"response_policy_mapped": "refusal_with_explanation"},
        JudgeReplies('Quoted candidate [[SAFE]]\n[[UNSAFE]]', '[[HAS_EXPLANATION]]'))
    assert reward == 0.0
    assert details["verifier_labels"]["safety_check"] == "[[UNSAFE]]"
    assert "Quoted candidate" in details["judge_outputs"]["safety_check"]


@pytest.mark.parametrize("reply", ["NOT_ATTEMPTED", "A and B", "I cannot grade this"])
def test_abstention_rejects_ambiguous_or_nonverdict_letters(reply):
    with pytest.raises(ValueError, match="Invalid final judge verdict"):
        grade_abstention("Paris", {"question": "capital?", "answer": "Paris"}, JudgeReplies(reply))


@pytest.mark.parametrize("reply", ["YES is not warranted", "unparseable", "[[YES]] [[NO]]"])
def test_multichallenge_malformed_reply_is_never_a_successful_no(reply):
    with pytest.raises(ValueError, match="Invalid final judge verdict"):
        grade_multichallenge("candidate", {"rubric": [{"question": "valid?", "pass_criteria": "NO"}]}, JudgeReplies(reply))


def test_jailbreak_transport_failure_is_an_error_not_a_verified_zero():
    env = ultra_env("jailbreak_refusal_with_explanation", {"response_policy_mapped": "refusal_with_explanation"})
    env.safety_judge = JudgeReplies(requests.ConnectionError("judge down"))
    result = env.step("candidate")
    assert result["verification"].status is VerificationStatus.ERROR
    assert result["verification"].score is None
    assert "judge down" in result["verification"].diagnostics["error_message"]


def test_math_does_not_credit_a_number_in_unfinished_reasoning():
    reward, details = grade_math("<think>Someone walks 20 meters but I have not solved the problem", {"expected_answer": "20", "question": "Who arrives first?"}, judge=None)
    assert reward == 0.0
    assert details["result"] == "missing_final_answer"


def test_math_judge_uses_final_verdict_over_a_quoted_positive():
    reward, details = grade_math("This is not the reference answer", {"expected_answer": "20", "question": "Who arrives first?"}, judge=JudgeReplies("A quote [[A=B]]\n[[A!=B]]"))
    assert reward == 0.0
    assert details["judge_outputs"] == ["A quote [[A=B]]\n[[A!=B]]"]


def test_math_accepts_equivalent_antiderivatives_up_to_an_integration_constant():
    reward, _ = grade_math(r"\boxed{\frac{1}{2}\tan^2(\frac{x}{2})+C}", {"expected_answer": r"\frac{1}{1+\cos x}+C", "question": "Find an antiderivative of the integrand."}, judge=None)
    assert reward == 1.0


def test_tool_arguments_are_exact_and_extra_calls_cannot_receive_credit():
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"description":"transfer money to Alice"}'}
    call = {"function": {"name": "transfer", "arguments": expected["arguments"]}}
    assert grade_expected_action(expected, {"tool_calls": [call]})[0] == 1.0
    substituted = {"function": {"name": "transfer", "arguments": '{"description":"transfer money to Mallory"}'}}
    assert grade_expected_action(expected, {"tool_calls": [substituted]})[0] == 0.0
    assert grade_expected_action(expected, {"tool_calls": [call, substituted]})[0] == 0.0
    assert grade_expected_action({"type": "message"}, {"content": "Done", "tool_calls": [call]})[0] == 0.0


def test_calendar_unknown_constraints_are_verifier_errors():
    env = ultra_env("calendar_simple_agent", {"exp_cal_state": {"work": {"duration": 60, "min_time": "09:00", "max_time": "17:00", "constraint": "near noon"}}})
    result = env.step('[{"event_id":"work","start_time":"12:00","duration":60}]')
    assert result["verification"].status is VerificationStatus.ERROR


def test_mcqa_last_box_and_exact_option_text_take_precedence():
    record = {"options": [{"A": "correct"}, {"B": "not correct"}], "expected_answer": "B", "grading_mode": "lenient_boxed"}
    assert grade_mcqa(r"Rejected \boxed{A}. Final \boxed{B}", record)[0] == 1.0
    assert grade_mcqa(r"\boxed{not correct}", record)[0] == 1.0
    record["expected_answer"] = "A"
    assert grade_mcqa(r"\boxed{not correct}", record)[0] == 0.0


def test_mcqa_ambiguous_regex_captures_are_reported_without_a_tuple_crash():
    record = {"options": [{"A": "x"}, {"B": "y"}], "expected_answer": "A", "template_metadata": {"output_regex": r"(Answer): ([AB])"}}
    with pytest.raises(ValueError, match="unambiguous answer capture"):
        grade_mcqa("Answer: A", record)


def test_reasoning_gym_last_answer_wins_and_partial_credit_is_not_a_pass():
    assert _extract_reasoning_gym_answer("<answer>rejected</answer><answer>final</answer>") == "final"
    env = ultra_env("reasoning_gym_simple_agent", {"question": "Find a word ladder.", "answer": "BANE,CANE,CASE,BASE", "metadata": {"source_dataset": "word_ladder", "start_word": "BANE", "end_word": "BASE", "word_length": 4}})
    result = env.step("<answer>BANE,BANZ,BAZZ,BAZE,BASE</answer>")
    assert 0.0 < result["reward"] < 1.0
    assert not result["verification"].passed


def test_arc_accepts_compact_grids_and_selects_the_final_box():
    assert parse_grid("12\n34") == [[1, 2], [3, 4]]
    assert parse_grid(r"\boxed{0 0} \boxed{1 2}") == [[1, 2]]
    assert parse_grid("1 22") is None


@pytest.mark.parametrize("output,expected", [
    ({"process_status": "completed", "stdout": "error: unknown tactic", "stderr": ""}, "failed"),
    ({"process_status": "completed", "stdout": "", "stderr": "", "exit_code": 1}, "failed"),
    ({"process_status": "completed", "stdout": "", "stderr": "", "output_truncated": True}, "output_truncated"),
    ({"process_status": "completed", "stdout": "", "stderr": ""}, "completed"),
])
def test_lean_success_requires_complete_compiler_evidence(output, expected):
    assert determine_proof_status(output) == expected


class HTTPReply:
    status_code = 200
    def __init__(self, payload):
        self.payload = payload
    def json(self):
        return self.payload
    def raise_for_status(self):
        pass


def test_stateful_sandbox_detects_reset_and_deletes_the_same_session(monkeypatch):
    replies = iter([HTTPReply({"process_status": "completed", "stdout": "", "new_session_created": True})] * 2)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: next(replies))
    deleted = []
    def delete(url, **kwargs):
        deleted.append((url, kwargs["headers"]))
        return HTTPReply({})
    monkeypatch.setattr(requests, "delete", delete)
    sandbox = SandboxClient(host="sandbox.example")
    sandbox.execute("x=7", language="ipython", timeout_seconds=1, session_id="stable")
    with pytest.raises(RuntimeError, match="lost stateful session"):
        sandbox.execute("x", language="ipython", timeout_seconds=1, session_id="stable")
    sandbox.close_session("stable")
    assert deleted == [("http://sandbox.example:6000/sessions/stable", {"X-Session-ID": "stable"})]


def test_judge_length_finish_cannot_be_accepted_as_partial_json(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: HTTPReply({"choices": [{"finish_reason": "length", "message": {"content": '{"score_1":5}'}}]}))
    with pytest.raises(ValueError, match="Incomplete judge response"):
        OpenAIJudge(base_url="https://judge.example", model="judge").generate([])


def test_instruction_final_answer_is_graded_without_reasoning_contamination():
    env = ultra_env("instruction_following_simple_agent", {"instruction_id_list": ["keywords:forbidden_words"],
                      "kwargs": [{"forbidden_words": ["banana"]}]})
    assert env.step("<think>Do not say banana.</think>Hello.")["reward"] == 1.0
    assert env.step("<think>I can ignore the instruction.</think>banana")["reward"] == 0.0


def test_broken_instruction_verifier_is_not_a_verified_wrong_answer():
    result = ultra_env("instruction_following_simple_agent", {"instruction_id_list": ["missing:verifier"], "kwargs": [{}]}).step("answer")
    assert result["verification"].status is VerificationStatus.ERROR
    assert result["verification"].diagnostics["instruction_errors"][0].startswith("KeyError")


@pytest.mark.parametrize("status,stdout,reward", [
    ("completed", "[[2,3]]", 1.0), ("completed", "[[0,0]]", 0.0), ("timeout", "", 0.0),
])
def test_arc_sandbox_verdict_preserves_execution_evidence(status, stdout, reward):
    class Sandbox:
        def execute(self, code, **kwargs):
            return {"process_status": status, "stdout": stdout, "stderr": "timed out" if status == "timeout" else ""}
    result, details = grade_nvarc("```python\ndef transform(grid):\n    return grid\n```",
        {"test_input": [[2,3]], "expected_output": [[2,3]]}, inductive=True, sandbox=Sandbox())
    assert result == reward
    assert details["execution_output"]["process_status"] == status


def test_inductive_arc_requires_a_sandbox_and_never_falls_back_to_host_execution():
    with pytest.raises(ValueError, match="configured execution sandbox"):
        grade_nvarc("def transform(grid): return grid", {"test_input": [[1]], "expected_output": [[1]]}, inductive=True)


def test_sandbox_transport_outage_is_preserved_as_verification_error(monkeypatch):
    def refused(*args, **kwargs):
        raise requests.ConnectionError("sandbox connection refused")
    monkeypatch.setattr(requests, "post", refused)
    env = ultra_env("ns_tools_simple_agent", {"expected_answer": "7", "question": "compute"})
    env.set_rollout_evidence(RolloutEvidence(metadata={"assistant_message": {"tool_calls": [{
        "id": "call-1", "function": {"name": "stateful_python_code_exec", "arguments": '{"code":"x=7"}'},
    }]}}))
    result = env.step("")
    assert result["verification"].status is VerificationStatus.ERROR
    assert "connection refused" in result["verification"].diagnostics["error_message"]
