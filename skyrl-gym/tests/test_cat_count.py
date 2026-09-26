import pytest
import skyrl_gym
from omegaconf import DictConfig

from skyrl_gym.envs.cat_count.reward import cat_count_score
from skyrl_gym.verification import RolloutEvidence

NS = [1, 2, 4, 7, 10, 20]


def reward(completion, n, *, stop_reason=None):
    return cat_count_score(completion, n, stop_reason=stop_reason).reward


def cats(k: int) -> str:
    return " ".join(["cat"] * k)


@pytest.mark.parametrize("n", NS)
def test_exact_scores_one_without_clipping(n):
    s = cat_count_score(cats(n), n)
    assert s.exact and s.reward == pytest.approx(1.0)
    assert reward(cats(n) + "<|im_end|>", n) == pytest.approx(1.0)
    assert reward(cats(n) + "<|im_end|> <|eot_id|>", n) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [n for n in NS if n > 1])
def test_reward_rises_monotonically_toward_n(n):
    scores = [reward(cats(k), n) for k in range(0, n + 1)]
    assert all(a < b for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("n", NS)
def test_exact_count_with_length_stop_keeps_verification_and_pays_penalty(n):
    score = cat_count_score(cats(n), n, stop_reason="length")
    assert score.exact
    assert score.truncated
    assert score.reward == pytest.approx(0.9)


@pytest.mark.parametrize("n", NS)
def test_junk_is_always_worse_than_the_same_cats_alone(n):
    assert reward("Sure! " + cats(n), n) < reward(cats(n), n)
    assert reward(cats(n) + " I hope that helps", n) < reward(cats(n), n)


@pytest.mark.parametrize("n", NS)
def test_more_junk_words_score_lower_until_the_cap(n):
    scores = [reward(" ".join(["junk"] * k + [cats(n)]), n) for k in range(0, 12)]
    assert all(a > b for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("n", NS)
def test_overshoot_keeps_falling_across_the_whole_64_token_budget(n):
    scores = [reward(cats(k), n) for k in range(n, 64)]
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert min(scores) > -1.0


def test_n1_non_cat_replies_score_nonpositive():
    assert reward("dog", 1) <= 0.0
    assert reward("", 1) == pytest.approx(-0.25)
    assert reward("I can help with that", 1) < 0.0


def test_near_miss_formats_earn_shaping_but_not_exact():
    for text in ["Cat.", "cat!", "CAT"]:
        s = cat_count_score(text, 1)
        assert not s.exact and 0.0 < s.reward < 1.0
    s = cat_count_score("cat  cat\ncat", 3)
    assert not s.exact and s.reward == pytest.approx(0.7)


def test_character_pieces_earn_nothing():
    assert reward("ca at c a t", 4) < 0.0
    assert reward("exact concatenate", 4) < 0.0


def test_truncation_penalty_only_when_known():
    assert reward(cats(25), 20, stop_reason="length") < reward(cats(25), 20)
    assert reward(cats(20), 20, stop_reason="stop") == pytest.approx(1.0)
    assert reward(cats(20), 20, stop_reason="unknown") == pytest.approx(1.0)


def test_off_by_one_is_well_separated_from_exact():
    for n in NS:
        assert reward(cats(n), n) - reward(cats(max(n - 1, 0)), n) >= 0.3


def test_reward_is_bounded():
    for text in ["", "cat " * 200, "junk " * 200, "cat dog " * 50]:
        for n in NS:
            assert -1.0 <= reward(text, n) <= 1.0


def test_embedded_end_marker_cannot_verify_exact():
    score = cat_count_score("cat<|im_end|> cat", 2)
    assert not score.exact
    assert score.reward < 1.0


@pytest.mark.parametrize("n", NS)
def test_empty_reply_loses_to_any_reply_within_2n_cats(n):
    empty = reward("", n)
    assert empty < 0.0
    for k in range(1, 2 * n + 1):
        assert reward(cats(k), n) > empty


def test_environment_scores_completion_only_and_reports_exact_by_n():
    env = skyrl_gym.make(
        "cat_count",
        env_config=DictConfig({"env_class": "cat_count"}),
        extras={"extra_info": {"n": 4}},
    )
    prompt = "Reply with the word cat exactly 4 times, separated by single spaces. Nothing else."
    env.init([{"role": "user", "content": prompt}])
    step = env.step("cat cat cat")
    assert step["reward"] == pytest.approx(cat_count_score("cat cat cat", 4).reward)
    assert step["verification"].score == 0.0
    assert step["verification"].passed is False
    assert env.get_metrics()["exact_n4"] == 0.0
    assert env.get_metrics()["n_words_n4"] == 3.0
    assert env.get_metrics()["has_cat"] == 1.0


def test_environment_keeps_exact_verification_separate_from_shaped_reward():
    env = skyrl_gym.make(
        "cat_count",
        env_config=DictConfig({"env_class": "cat_count"}),
        extras={"extra_info": {"n": 2}},
    )
    step = env.step("cat cat")
    assert step["reward"] == pytest.approx(1.0)
    assert step["verification"].score == 1.0
    assert step["verification"].passed is True
    assert env.get_metrics()["exact_n2"] == 1.0


def test_environment_penalizes_reported_length_stop():
    env = skyrl_gym.make(
        "cat_count",
        env_config=DictConfig({"env_class": "cat_count"}),
        extras={"extra_info": {"n": 2}},
    )
    env.set_rollout_evidence(RolloutEvidence(stop_reason="length"))
    step = env.step("cat cat")
    assert step["reward"] == pytest.approx(0.9)
    assert step["verification"].score == 1.0
    assert env.get_metrics()["truncated"] == 1.0
