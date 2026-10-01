"""Freeze stochastic task references before a candidate reaches the environment."""

import copy
import json
import random

import pytest
from omegaconf import OmegaConf

from infra.rl_data.sources import (
    nemotron_ultra_mopd_source,
    nemotron_ultra_rlvr1_source,
    nemotron_ultra_rlvr2_source,
)
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
from skyrl_gym.envs.instruction_verifyit import grade_nemotron_instructions
from skyrl_gym.envs.nemotron_ultra.instruction_following import (
    grade_instruction_following,
)


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
        "responses_create_params": {
            "input": [{"role": "user", "content": "Write a response"}]
        },
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
            env = NemotronUltraEnv(
                OmegaConf.create({"verifyit_enabled": enabled}), extras=row
            )
            env.init(row["prompt"])
            try:
                scores.append(env.step(candidate)["reward"])
            finally:
                env.close()
        assert scores == [expected, expected]


def test_explicit_keyword_needs_no_unused_random_instruction():
    record = {
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"keyword": "blue"}],
    }
    for candidate in ["a blue b", "permitted"]:
        assert grade_nemotron_instructions(
            candidate, record
        ) == grade_instruction_following(candidate, record)


def test_explicit_zero_span_index_is_deterministic():
    record = {
        "instruction_id_list": ["new:copy_span_idx"],
        "kwargs": [
            {"prompt_to_repeat": "long example string", "n_start": 0, "n_end": 3}
        ],
    }
    state = random.getstate()
    for candidate, score in [("lon", 1.0), ("ong", 0.0)]:
        native = grade_instruction_following(candidate, record)
        cutover = grade_nemotron_instructions(candidate, record)
        assert native == cutover
        assert cutover[0] == score
    assert random.getstate() == state


@pytest.mark.parametrize("candidate", ["permitted", "red blue green"])
def test_legacy_reference_runtime_preserves_source_score_and_rng(candidate):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"instruction": "red blue green"}],
    }
    original_state = random.getstate()
    try:
        random.seed(7194)
        before = random.getstate()
        expected = grade_instruction_following(candidate, record)
        after = random.getstate()
        random.setstate(before)
        actual = grade_nemotron_instructions(candidate, record)
        assert actual == expected
        assert random.getstate() == after
    finally:
        random.setstate(original_state)


@pytest.mark.parametrize(
    "identity",
    ["length_constraints:nth_paragraph_first_word", "count:count_increment_word"],
)
def test_default_source_defects_are_resolved_before_serialization(identity):
    raw = {
        "uuid": "resolved-default",
        "agent_ref": {"name": "instruction_following_simple_agent"},
        "responses_create_params": {
            "input": [{"role": "user", "content": "Write a response"}]
        },
        "instruction_id_list": [identity],
        "kwargs": [{}],
    }
    row = nemotron_ultra_mopd_source(instruction_reference_seed=13).prepare_row(
        raw, 0, None
    )
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
            env = NemotronUltraEnv(
                OmegaConf.create({"verifyit_enabled": enabled}), extras=row
            )
            env.init(row["prompt"])
            try:
                scores.append(env.step(candidate)["reward"])
            finally:
                env.close()
        assert scores == [expected, expected]


def test_frozen_reference_revision_mismatch_is_invalid():
    record = {
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"keyword": "blue"}],
        "instruction_reference_seed": 13,
        "instruction_reference_revision": "different-registry",
    }
    score, detail = grade_nemotron_instructions("permitted", record)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"


@pytest.mark.parametrize("arguments", [{}, {"keywords": None}, {"keywords": []}])
def test_keyword_default_sentinels_consume_identical_source_rng(arguments):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {"instruction_id_list": ["keywords:existence"], "kwargs": [arguments]}
    original_state = random.getstate()
    try:
        random.seed(790)
        before = random.getstate()
        expected = grade_instruction_following("ordinary unrelated answer", record)
        after = random.getstate()
        random.setstate(before)
        assert (
            grade_nemotron_instructions("ordinary unrelated answer", record) == expected
        )
        assert random.getstate() == after
    finally:
        random.setstate(original_state)


@pytest.mark.parametrize("candidate", ["strin", "incorrect"])
def test_nonempty_negative_copy_span_preserves_source_contract(candidate):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {
        "instruction_id_list": ["new:copy_span_idx"],
        "kwargs": [
            {"prompt_to_repeat": "long example string", "n_start": -6, "n_end": -1}
        ],
    }
    assert grade_nemotron_instructions(
        candidate, record
    ) == grade_instruction_following(candidate, record)


@pytest.mark.parametrize(
    "candidate",
    ["This is an English sentence about a garden.", "这是一个中文句子。", "a"],
)
def test_source_detector_policy_is_reproducible(candidate):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {
        "instruction_id_list": ["language:response_language"],
        "kwargs": [{"language": "en"}],
    }
    expected = grade_instruction_following(candidate, record)
    assert grade_nemotron_instructions(candidate, record) == expected
    assert grade_instruction_following(candidate, record) == expected


def test_failed_runtime_does_not_commit_partially_consumed_rng():
    record = {
        "instruction_id_list": ["keywords:existence", "unknown:instruction"],
        "kwargs": [{}, {}],
    }
    before = random.getstate()
    score, feedback = grade_nemotron_instructions("candidate", record)
    assert score == 0.0
    assert feedback["error_type"] == "schema_error"
    assert random.getstate() == before


def test_malformed_scored_child_result_does_not_commit_rng(monkeypatch):
    import dataclasses
    import verifyit.grade

    real_run = verifyit.grade.run

    def missing_feedback(*args, **kwargs):
        verdict = real_run(*args, **kwargs)
        detail = dict(verdict.detail)
        assert "random_state" in detail
        detail.pop("source_feedback")
        return dataclasses.replace(verdict, detail=detail)

    monkeypatch.setattr(verifyit.grade, "run", missing_feedback)
    before = random.getstate()
    score, feedback = grade_nemotron_instructions(
        "unrelated candidate",
        {"instruction_id_list": ["keywords:existence"], "kwargs": [{}]},
    )
    assert score == 0.0
    assert feedback["error_type"] == "verification_error"
    assert random.getstate() == before
