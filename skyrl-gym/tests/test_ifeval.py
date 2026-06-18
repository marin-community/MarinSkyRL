import json
import skyrl_gym
import pytest
from omegaconf import DictConfig


def _gt(**overrides) -> str:
    """Build a RLVR-IFeval-shaped ground_truth JSON: every spec key present, null by default."""
    spec = {
        "func_name": None,
        "N": None,
        "quantifier": None,
        "end_phrase": None,
        "keyword_list": None,
        "word": None,
        "forbidden_words": None,
        "letter": None,
        "i": None,
        "first_word": None,
        "postscript_marker": None,
        "options": None,
        "section_splitter": None,
        "original_prompt": None,
    }
    spec.update(overrides)
    return json.dumps(spec)


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        # validate_lowercase
        ("this is all lowercase.", _gt(func_name="validate_lowercase"), 1.0),
        ("This Has Capitals.", _gt(func_name="validate_lowercase"), 0.0),
        # verify_keywords
        (
            "the quick brown fox jumps",
            _gt(func_name="verify_keywords", keyword_list=["quick", "fox"]),
            1.0,
        ),
        (
            "the slow brown bear sleeps",
            _gt(func_name="verify_keywords", keyword_list=["quick", "fox"]),
            0.0,
        ),
        # validate_word_constraint (at least)
        (
            "one two three four five",
            _gt(func_name="validate_word_constraint", N=3, quantifier="at least"),
            1.0,
        ),
        (
            "one two",
            _gt(func_name="validate_word_constraint", N=3, quantifier="at least"),
            0.0,
        ),
        # verify_postscript
        (
            "Here is my answer.\nP.S. thanks for reading",
            _gt(func_name="verify_postscript", postscript_marker="P.S."),
            1.0,
        ),
        (
            "Here is my answer with no postscript.",
            _gt(func_name="verify_postscript", postscript_marker="P.S."),
            0.0,
        ),
        # validate_end
        (
            "the conversation is over. that's all folks",
            _gt(func_name="validate_end", end_phrase="that's all folks"),
            1.0,
        ),
        (
            "that's all folks but then I kept going",
            _gt(func_name="validate_end", end_phrase="that's all folks"),
            0.0,
        ),
        # validate_json_format
        ('{"a": 1, "b": [2, 3]}', _gt(func_name="validate_json_format"), 1.0),
        ("this is not json", _gt(func_name="validate_json_format"), 0.0),
        # verify_keyword_frequency (exact count, word-boundary)
        (
            "go go go and stop",
            _gt(func_name="verify_keyword_frequency", word="go", N=3),
            1.0,
        ),
        (
            "go go and stop",
            _gt(func_name="verify_keyword_frequency", word="go", N=3),
            0.0,
        ),
        # validate_uppercase
        ("THIS IS LOUD.", _gt(func_name="validate_uppercase"), 1.0),
        ("This is quiet.", _gt(func_name="validate_uppercase"), 0.0),
        # verify_bullet_points (exactly N)
        (
            "* one\n* two\n* three",
            _gt(func_name="verify_bullet_points", N=3),
            1.0,
        ),
        (
            "* one\n* two",
            _gt(func_name="verify_bullet_points", N=3),
            0.0,
        ),
        # validate_no_commas
        ("no commas here at all", _gt(func_name="validate_no_commas"), 1.0),
        ("this, has, commas", _gt(func_name="validate_no_commas"), 0.0),
        # verify_letter_frequency with a bad (multi-char) letter -> checker raises -> score 0
        (
            "anything",
            _gt(func_name="verify_letter_frequency", letter="ab", N=1),
            0.0,
        ),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "ifeval",
        env_config=DictConfig({"env_class": "ifeval"}),
        extras={"reward_model": {"method": "rule", "ground_truth": ground_truth}},
    )
    step_output = env.step(output)
    assert step_output["reward"] == expected


def test_unknown_func_name_scores_zero():
    # An unknown/unimplemented func_name must NOT crash the rollout (env.step is not
    # wrapped in try/except in the SkyRL generator) — it logs + scores 0.
    env = skyrl_gym.make(
        "ifeval",
        env_config=DictConfig({"env_class": "ifeval"}),
        extras={
            "reward_model": {
                "method": "rule",
                "ground_truth": _gt(func_name="not_a_real_check"),
            }
        },
    )
    step_output = env.step("anything")
    assert step_output["reward"] == 0.0


def test_missing_func_name_scores_zero():
    env = skyrl_gym.make(
        "ifeval",
        env_config=DictConfig({"env_class": "ifeval"}),
        extras={
            "reward_model": {"method": "rule", "ground_truth": _gt()}
        },  # func_name=None
    )
    step_output = env.step("anything")
    assert step_output["reward"] == 0.0
