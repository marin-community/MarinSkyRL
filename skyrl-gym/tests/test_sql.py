"""Multi-turn SQL tools and final grading on a task-local database snapshot."""

import sqlite3

import pytest
from taskcompendium.grading import Outcome

from skyrl_gym.envs.sql.utils import final_sql


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "spider/database/fixture/fixture.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES(1),(1),(2);")
    return path, {"db_path": str(tmp_path)}


def extras(reference="SELECT x FROM t"):
    return {"db_id": "fixture", "data": "spider", "reward_spec": {"ground_truth": reference}}


@pytest.mark.asyncio
@pytest.mark.parametrize("verifyit", [False, True])
@pytest.mark.parametrize("final,expected", [("SELECT DISTINCT x FROM t", 1.0), ("SELECT 99", 0.0)])
async def test_sql_observations_and_set_grade_preserve_the_original_database(
    rollout_session, database, final, expected, verifyit
):
    path, config = database
    rollout = await rollout_session(
        "text2sql",
        ["<think>query</think><sql>SELECT count(*) FROM t</sql>", f"<think>answer</think><solution>{final}</solution>"],
        extras(),
        {**config, "verifyit_enabled": verifyit},
    )
    observation = rollout.steps[0].transition.observations[0]
    assert "3" in observation["content"]
    assert "<reminder>You have 2 turns left to complete the task.</reminder>" in observation["content"]
    assert rollout.steps[1].messages[-2] == observation
    assert [step.transition.reward for step in rollout.steps] == [0.0, expected]
    assert rollout.grade.reward == expected / 2
    assert rollout.loss_mask == (1, 1, 0, 0, 1, 1)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM t").fetchone()[0] == 3


@pytest.mark.asyncio
async def test_sql_write_tool_is_rejected_and_a_later_read_can_succeed(rollout_session, database):
    _, config = database
    rollout = await rollout_session(
        "text2sql",
        ["<think>query</think><sql>DELETE FROM t</sql>", "<think>answer</think><solution>SELECT x FROM t</solution>"],
        extras(),
        config,
    )
    assert rollout.steps[0].transition.observations
    assert rollout.steps[1].transition.reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("verifyit", [False, True])
async def test_sql_invalid_format_retains_negative_reward(rollout_session, database, verifyit):
    _, config = database
    rollout = await rollout_session(
        "text2sql", ["wrong format"], extras(), {**config, "verifyit_enabled": verifyit}, max_turns=1
    )
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, -1.0)
    assert rollout.steps[0].transition.reward == -1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("verifyit", [False, True])
@pytest.mark.parametrize("response", ["wrong format", "<sql>SELECT 1</sql>trailing text"])
async def test_broken_sql_reference_has_no_grade(rollout_session, database, verifyit, response):
    _, config = database
    rollout = await rollout_session(
        "text2sql", [response], extras("SELECT missing FROM t"), {**config, "verifyit_enabled": verifyit}
    )
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.INFRA_ERROR, None)


@pytest.mark.parametrize(
    "response,query",
    [
        ("<think>query</think><solution>SELECT * FROM t</solution>", "SELECT * FROM t"),
        ("<solution>SELECT * FROM t</solution>", None),
        ("<think>query</think><solution>SELECT * FROM t; <sql>query</sql></solution>", None),
        (
            "<observation>rows</observation><think>query</think><solution>SELECT * FROM t</solution>",
            "SELECT * FROM t",
        ),
        (
            "<observation>rows</observation>other text<think>query</think><solution>SELECT * FROM t</solution>",
            None,
        ),
        ("some text<solution>", None),
    ],
)
def test_sql_format_contract_extracts_only_a_final_query(response, query):
    assert final_sql(response) == query
