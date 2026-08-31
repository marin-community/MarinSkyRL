import pytest

from skyrl_train.utils.reward_shaping import shape_reward_from_output


CODEFORCES_OUTPUT = """\
test 1: PASS
test 2: FAIL
Results: 13/20 passed (ratio=0.65)
"""

CODEFORCES_NO_SOLUTION_OUTPUT = "No solution found at /app/solution.py|solution.cpp|Solution.java\n"


def test_codeforces_summary_uses_reported_pass_ratio():
    assert shape_reward_from_output(CODEFORCES_OUTPUT, original_reward=0.0, shaper_name="pass_ratio") == 0.65


def test_codeforces_no_solution_is_an_explicit_zero_result():
    assert (
        shape_reward_from_output(
            CODEFORCES_NO_SOLUTION_OUTPUT,
            original_reward=0.0,
            shaper_name="threshold",
            fallback_to_original=False,
        )
        == 0.0
    )


def test_parse_failure_raises_when_fallback_is_disabled():
    with pytest.raises(ValueError, match="could not parse verifier output"):
        shape_reward_from_output(
            "verifier produced no test summary",
            original_reward=1.0,
            shaper_name="pass_ratio",
            fallback_to_original=False,
        )
