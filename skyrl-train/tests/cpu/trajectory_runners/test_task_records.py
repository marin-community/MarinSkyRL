from skyrl_gym.task_records import fold_grades
from taskcompendium.grading_result import GradeResult, Outcome


def test_multiturn_verification_averages_scored_turns_and_requires_all_to_pass():
    results = [
        GradeResult(Outcome.GRADED, 5.0, passed=True, score_min=1.0, score_max=5.0),
        GradeResult(Outcome.UNAVAILABLE, None, "Tool turn has no verdict"),
        GradeResult(Outcome.GRADED, 0.0, passed=False),
    ]

    grade = fold_grades(results)
    assert grade.reward == 0.5
    assert grade.passed is False
    assert grade.diagnostics["num_scored_steps"] == 2
