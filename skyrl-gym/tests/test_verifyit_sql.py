"""Shellbox query execution and private exact comparison on real SQLite databases."""

import json

import pytest
from taskcompendium.grading_result import Outcome


@pytest.fixture
def reference():
    return {
        "schema_sql": "CREATE TABLE t(x INTEGER)",
        "insert_sql": "INSERT INTO t VALUES(1),(1),(2),(3)",
        "reference_sql": "SELECT x FROM t",
        "order_significant": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("verifyit", [False, True])
@pytest.mark.parametrize(
    "query,expected",
    [
        ("SELECT x AS renamed FROM t ORDER BY x DESC", 1),
        ("SELECT DISTINCT x FROM t", 0),
        ("SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3", 0),
        ("DELETE FROM t", 0),
    ],
)
async def test_seeded_sql_preserves_duplicates_perturbation_and_readonly(
    rollout_session, reference, query, expected, verifyit
):
    rollout = await rollout_session(
        "text_to_sql",
        [query],
        {"reward_model": {"ground_truth": json.dumps(reference)}},
        {"verifyit_enabled": verifyit},
    )
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, expected)
    assert rollout.steps[0].transition.reward == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference_sql,candidate,expected",
    [
        ("SELECT 1.0/3", "SELECT 0.333333", 1),
        ("SELECT x FROM t", "SELECT x,x FROM t", 0),
        ("SELECT x FROM t", "SELECT 1e999", 0),
        ("SELECT x'00ff' FROM t", "SELECT x'00ff' FROM t", 1),
    ],
)
async def test_exact_sql_preserves_rounding_columns_and_value_types(
    rollout_session, reference, reference_sql, candidate, expected
):
    reference["reference_sql"] = reference_sql
    rollout = await rollout_session(
        "text_to_sql",
        [candidate],
        {"reward_model": {"ground_truth": json.dumps(reference)}},
        {"verifyit_enabled": True},
    )
    assert rollout.grade.reward == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_sql", ["SELECT absent_column FROM t", "SELECT 1e999"])
async def test_invalid_reference_has_no_grade_even_when_candidate_is_rejected(
    rollout_session, reference, reference_sql
):
    reference["reference_sql"] = reference_sql
    rollout = await rollout_session(
        "text_to_sql",
        ["DELETE FROM t"],
        {"reward_model": {"ground_truth": json.dumps(reference)}},
        {"verifyit_enabled": True},
    )
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.INFRA_ERROR, None)
    assert rollout.steps[0].transition.reward == 0.0
