"""Actual source judge compositions with a real local model protocol boundary."""

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import skyrl_gym
from omegaconf import OmegaConf

from verifyit.grade import InvalidTask

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_profiles_verifyit import (
    grade_judge_profile_verifyit,
    prepare_profile,
    structure_profile,
)
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import (
    _MULTICHALLENGE_PROMPT,
    grade_abstention,
    grade_multichallenge,
)
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak, _configuration


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            owner.requests.append(request)
            text = owner.by_prompt.get(request["messages"][-1]["content"], owner.reply)
            response = {
                "id": "source-judge",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    owner = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    owner.requests, owner.by_prompt, owner.reply = [], {}, "A"
    thread = threading.Thread(target=owner.serve_forever, daemon=True)
    thread.start()
    judge = OpenAIJudge(
        base_url=f"http://127.0.0.1:{owner.server_port}/v1",
        model="source-judge",
        api_key="test",
        timeout_seconds=10.0,
    )
    yield owner, judge
    owner.shutdown()
    owner.server_close()
    thread.join()


@pytest.mark.parametrize("reply,score", [("Assistant assessed Answer.\nA", 1.0), ("B", 0.0), ("C", 0.5)])
def test_abstention_native_and_existing_label_composition(server, reply, score):
    owner, judge = server
    owner.reply = reply
    record = {"question": "What is two plus two?", "answer": "4"}
    native = grade_abstention(r"\boxed{4}", record, judge)
    requests = list(owner.requests)
    owner.requests.clear()
    cutover = grade_judge_profile_verifyit(r"\boxed{4}", record, judge, kind="abstention")
    assert native[0] == cutover[0]
    assert cutover[0] == score
    assert owner.requests == requests


def test_idk_normalization_retains_half_without_a_model(server):
    owner, judge = server
    record = {"question": "What is two plus two?", "answer": "4"}
    native = grade_abstention(r"\boxed{[IDK]}", record, judge)
    cutover = grade_judge_profile_verifyit(r"\boxed{[IDK]}", record, judge, kind="abstention")
    assert native[0] == cutover[0] == 0.5
    assert owner.requests == []


@pytest.mark.parametrize(
    "expected,reply,score",
    [("YES", "[[YES]]", 1.0), ("NO", "[[NO]]", 1.0), ("YES", "[[NO]]", 0.0)],
)
def test_multichallenge_expected_direction_and_source_prompts(server, expected, reply, score):
    owner, judge = server
    owner.reply = reply
    record = {
        "context": "trusted {context} {response} {pass_criteria} {candidate} {reference}",
        "rubric": [
            {"question": "Criterion one?", "pass_criteria": expected},
            {
                "question": "Criterion two?",
                "pass_criteria": "NO" if expected == "YES" else "YES",
            },
        ],
    }
    native = grade_multichallenge("candidate {reference}", record, judge)
    requests = sorted(owner.requests, key=lambda item: item["messages"][-1]["content"])
    owner.requests.clear()
    cutover = grade_judge_profile_verifyit("candidate {reference}", record, judge, kind="multichallenge")
    assert native[0] == cutover[0]
    assert cutover[0] == 0.5
    assert cutover[1]["verifier_rewards"]["0"] == score
    assert sorted(owner.requests, key=lambda item: item["messages"][-1]["content"]) == requests


@pytest.mark.parametrize(
    "policy",
    [
        "engagement_with_disclaimer",
        "hard_refusal_no_redirection",
        "hard_refusal_with_helplines",
        "refusal_with_explanation",
    ],
)
@pytest.mark.parametrize("kind", ["positive", "negative"])
def test_jailbreak_all_source_policies_use_existing_label_grading(server, policy, kind):
    owner, judge = server
    config = _configuration()
    names = config["policy_verifier_map"][policy]["verifiers"]
    record = {
        "response_policy_mapped": policy,
        "adversarial_prompt": "trusted harmless fixture question",
    }
    for name in names:
        verifier = config[name]
        prompt = verifier["prompt_template"].format(
            adversarial_prompt=record["adversarial_prompt"],
            model_response="candidate response",
        )
        owner.by_prompt[prompt] = verifier["labels"][kind]
    native = grade_jailbreak("candidate response", record, judge)
    requests = sorted(owner.requests, key=lambda item: item["messages"][-1]["content"])
    owner.requests.clear()
    cutover = grade_judge_profile_verifyit("candidate response", record, judge, kind="jailbreak")
    assert native[0] == cutover[0]
    assert sorted(owner.requests, key=lambda item: item["messages"][-1]["content"]) == requests


@pytest.mark.parametrize(
    "kind,record",
    [
        ("abstention", {"question": "Question", "answer": ""}),
        ("multichallenge", {"rubric": []}),
        (
            "jailbreak",
            {"response_policy_mapped": "unknown", "adversarial_prompt": "Question"},
        ),
    ],
)
def test_malformed_trusted_judge_contract_cannot_gain_positive_credit(server, kind, record):
    owner, judge = server
    score, detail = grade_judge_profile_verifyit("candidate", record, judge, kind=kind)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"
    assert owner.requests == []


def test_later_multichallenge_failure_discards_prior_credit(server):
    owner, judge = server
    owner.reply = "[[YES]]"
    record = {"rubric": [{"question": "First?"}, {"question": "Second?"}]}
    owner.by_prompt[
        _MULTICHALLENGE_PROMPT.format(context="", response="candidate", question="Second?", pass_criteria="YES")
    ] = "ambiguous [[YES]] and [[NO]]"
    score, detail = grade_judge_profile_verifyit("candidate", record, judge, kind="multichallenge")
    assert score == 0.0
    assert detail["error_type"] == "verification_error"
    assert len(owner.requests) == 2


pytest_plugins = ["test_math_judge_verifyit"]


@pytest.mark.parametrize("response", ["[IDK]", "The [idk]!", r"answer elsewhere \boxed{[IDK]}"])
def test_abstention_gate_retains_source_policy_without_provider(response, judge_server):
    server, judge = judge_server
    reward, details = grade_judge_profile_verifyit(
        response, {"question": "Q?", "answer": "private reference"}, judge, kind="abstention"
    )
    assert reward == 0.5
    assert details["verifyit_status"] == "scored"
    assert not server.requests
    assert "private reference" not in json.dumps(details)
    assert details["preparation"]["policies"]["abstention"] == "nemotron_articles_punctuation_case_v1"


def test_multichallenge_preflights_later_reference_and_discards_later_provider_failure(judge_server):
    server, judge = judge_server
    record = {"rubric": [{"question": "First?"}, {"question": "Later?", "pass_criteria": ""}]}
    reward, details = grade_judge_profile_verifyit("candidate", record, judge, kind="multichallenge")
    assert reward == 0
    assert details["verifyit_status"] == "invalid_task"
    assert not server.requests
    record["rubric"][1]["pass_criteria"] = "YES"
    server.replies = ["YES", "malformed"]
    reward, details = grade_judge_profile_verifyit("candidate", record, judge, kind="multichallenge")
    assert reward == 0
    assert details["verifyit_status"] == "infra_error"
    assert details["preparation_stage"] == "judge_provider"


@pytest.mark.parametrize(
    "policy,replies",
    [
        ("engagement_with_disclaimer", ["[[SAFE]]", "[[ENGAGED]]", "[[MISSING_DISCLAIMER]]"]),
        ("hard_refusal_no_redirection", ["[[SAFE]]", "[[IMPROPER_RESPONSE]]"]),
        ("hard_refusal_with_helplines", ["[[SAFE]]", "[[MISSING_HELPLINES]]"]),
        ("refusal_with_explanation", ["[[SAFE]]", "[[NO_EXPLANATION]]"]),
    ],
)
def test_registered_policy_retains_fractional_product_and_raw_response_policy(policy, replies, judge_server):
    server, judge = judge_server
    server.replies = replies
    agent = "jailbreak_" + policy
    ultra = {
        "route": "skyrl_gym",
        "agent": agent,
        "record_json": json.dumps({"adversarial_prompt": "Task?", "response_policy_mapped": policy}),
        "request_json": "{}",
    }
    env = skyrl_gym.make(
        "nemotron_ultra",
        env_config={"verifyit_enabled": True, "judges": {"safety": dataclasses.asdict(judge)}},
        extras={"extra_info": {"nemotron_ultra": ultra}},
    )
    try:
        result = env.step("<think>private reasoning</think>Response")
    finally:
        env.close()
    assert result["reward"] == 0.3
    details = result["verification"].diagnostics
    assert details["verifyit_status"] == "scored"
    assert details["preparation"]["aggregation"] == "product"
    assert "private reasoning" not in server.requests[0]["messages"][-1]["content"]


def test_structural_snapshot_preserves_values_before_named_policy_and_rejects_unknown_controls(judge_server):
    _, judge = judge_server
    record = {"question": "Q?", "answer": "A", "unused": [None, {"value": "original"}]}
    inputs = structure_profile("<think>hidden</think>[IDK]", record, "abstention", {})
    record["unused"][1]["value"] = "changed"
    assert inputs.record["unused"] == [None, {"value": "original"}]
    assert inputs.response == "<think>hidden</think>[IDK]"
    prepared = prepare_profile(inputs, dataclasses.asdict(judge))
    assert prepared.candidate == "[IDK]"
    assert prepared.gate_candidate == "idk"
    malformed = dataclasses.replace(inputs, policies={"response": "unknown"})
    with pytest.raises(InvalidTask):
        prepare_profile(malformed, dataclasses.asdict(judge))


@pytest.mark.parametrize(
    "response_policy,status,reward", [("literal_v1", "scored", 1.0), ("unknown", "invalid_task", 0.0)]
)
def test_registered_profile_accepts_hydra_policy_controls_and_preserves_original_prompt(
    judge_server, response_policy, status, reward
):
    owner, judge = judge_server
    owner.replies = ["A"]
    ultra = {
        "route": "skyrl_gym",
        "agent": "abstention_simple_agent",
        "record_json": json.dumps({"question": "Q?", "answer": "private"}),
        "request_json": "{}",
    }
    config = OmegaConf.create(
        {
            "verifyit_enabled": True,
            "verifyit_judge_profile_policies": {"response": response_policy},
            "judges": {"general": dataclasses.asdict(judge)},
        }
    )
    env = skyrl_gym.make("nemotron_ultra", env_config=config, extras={"extra_info": {"nemotron_ultra": ultra}})
    try:
        result = env.step("<think>retained reasoning</think>answer")
    finally:
        env.close()
    assert result["reward"] == reward
    assert result["verification"].diagnostics["verifyit_status"] == status
    if status == "scored":
        assert "<think>retained reasoning</think>answer" in owner.requests[0]["messages"][-1]["content"]
        assert result["verification"].diagnostics["preparation"]["policies"]["response"] == "literal_v1"
    else:
        assert not owner.requests
