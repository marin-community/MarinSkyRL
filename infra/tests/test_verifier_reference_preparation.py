"""Prepared infrastructure task references preserve source and unified scoring."""

import copy
import dataclasses
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from omegaconf import OmegaConf
from infra.rl_data.sources import nemotron_ultra_mopd_source, nemotron_ultra_rlvr1_source, nemotron_ultra_rlvr2_source
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge

@pytest.mark.parametrize(
    "factory",
    [
        nemotron_ultra_mopd_source,
        nemotron_ultra_rlvr1_source,
        nemotron_ultra_rlvr2_source,
    ],
)
def test_preparation_serializes_reference_before_framework_roundtrip(factory):
    raw = {
        "uuid": "frozen",
        "agent_ref": {"name": "instruction_following_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "Write a response"}]},
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"instruction": "red blue green"}],
    }
    original = copy.deepcopy(raw)
    source = factory(instruction_reference_seed=13)
    state = random.getstate()
    row = source.prepare_row(raw, 0, None)
    assert row == source.prepare_row(raw, 0, None)
    assert random.getstate() == state
    assert raw == original
    record = json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])
    assert record["kwargs"][0]["keyword"] == "blue"
    for candidate, expected in [("certain permitted sentence", 1.0), ("a blue b", 0.0)]:
        scores = []
        for enabled in [False, True]:
            env = NemotronUltraEnv(OmegaConf.create({"verifyit_enabled": enabled}), extras=row)
            env.init(row["prompt"])
            try:
                scores.append(env.step(candidate)["reward"])
            finally:
                env.close()
        assert scores == [expected, expected]


@pytest.mark.parametrize(
    "identity",
    ["length_constraints:nth_paragraph_first_word", "count:count_increment_word"],
)
def test_default_source_defects_are_resolved_before_serialization(identity):
    raw = {
        "uuid": "resolved-default",
        "agent_ref": {"name": "instruction_following_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "Write a response"}]},
        "instruction_id_list": [identity],
        "kwargs": [{}],
    }
    row = nemotron_ultra_mopd_source(instruction_reference_seed=13).prepare_row(raw, 0, None)
    record = json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])
    args = record["kwargs"][0]
    if identity == "length_constraints:nth_paragraph_first_word":
        assert 1 <= args["nth_paragraph"] <= args["num_paragraphs"]
        positive = "\n\n".join([args["first_word"] + " body"] * args["num_paragraphs"])
    else:
        assert isinstance(args["keyword1"], str) and isinstance(args["keyword2"], str)
        positive = f"{args['keyword1']} {args['keyword2']} {args['keyword2']}"
    for candidate, expected in [(positive, 1.0), ("unrelated content", 0.0)]:
        scores = []
        for enabled in [False, True]:
            env = NemotronUltraEnv(OmegaConf.create({"verifyit_enabled": enabled}), extras=row)
            env.init(row["prompt"])
            try:
                scores.append(env.step(candidate)["reward"])
            finally:
                env.close()
        assert scores == [expected, expected]



@pytest.fixture
def judge_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.requests.append(body)
            index = min(len(server.requests) - 1, len(server.replies) - 1)
            response = {
                "id": "source-judge",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": server.finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": server.replies[index],
                        },
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    server.replies = ["[[A=B]]"]
    server.finish_reason = "stop"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    judge = OpenAIJudge(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="recorded-test-judge",
        timeout_seconds=2,
    )
    try:
        yield server, judge
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("agent", ["math_with_judge_simple_agent", "ns_tools_simple_agent"])
@pytest.mark.parametrize("label,score", [("[[A=B]]", 1.0), ("[[A!=B]]", 0.0)])
def test_prepared_semantic_reference_roundtrips_framework(agent, label, score, judge_server):

    server, judge = judge_server
    server.replies = [label]
    raw = {
        "agent_ref": {"name": agent},
        "responses_create_params": {"input": [{"role": "user", "content": "Describe the strategy."}]},
        "question": "Describe the strategy.",
        "expected_answer": "Choose the next box (unless it is empty).\n• Stop at Box 8 — then return.",
    }
    # This selection is made before candidate creation; the source record remains unchanged.
    row = nemotron_ultra_mopd_source(math_reference_kind="semantic").prepare_row(raw, 0, None)
    serialized = json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])
    assert serialized["math_reference_kind"] == "semantic"
    assert "math_reference_kind" not in raw
    candidate = "Move to each nonempty next box and stop when reaching the eighth box."
    results = []
    requests = []
    for enabled in (False, True):
        server.requests.clear()
        env = NemotronUltraEnv(
            OmegaConf.create(
                {
                    "verifyit_enabled": enabled,
                    "judges": {"general": dataclasses.asdict(judge)},
                }
            ),
            extras=row,
        )
        env.init(row["prompt"])
        try:
            results.append(env.step(candidate)["reward"])
            requests.append(list(server.requests))
        finally:
            env.close()
    assert results == [score, score]
    assert requests[0] == requests[1]
    assert len(requests[1]) == (2 if score else 1)

