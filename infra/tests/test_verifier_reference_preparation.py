"""Prepared infrastructure task references preserve source and unified scoring."""

import copy
import json
import random

import pytest
from omegaconf import OmegaConf
from infra.rl_data.sources import nemotron_ultra_mopd_source, nemotron_ultra_rlvr1_source, nemotron_ultra_rlvr2_source
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv

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

