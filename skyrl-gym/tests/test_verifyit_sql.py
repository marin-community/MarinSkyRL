"""Real SQLite fixtures exercise trusted execution and independent exact composition."""

import json
import sqlite3

import pytest
from verifyit.grade import InvalidTask
from omegaconf import OmegaConf

import skyrl_gym
from skyrl_gym.envs.sqlite_verifyit import score_legacy_sql, score_seeded_sql
from skyrl_gym.envs.text_to_sql.scoring import score
from skyrl_gym.verification import VerificationStatus


@pytest.fixture
def reference():
    return json.dumps(
        {
            "schema_sql": "CREATE TABLE t(x INTEGER)",
            "insert_sql": "INSERT INTO t VALUES(1),(1),(2),(3)",
            "reference_sql": "SELECT x FROM t",
            "order_significant": False,
        }
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("SELECT x AS renamed FROM t ORDER BY x DESC", 1),
        ("SELECT DISTINCT x FROM t", 0),
        ("SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3", 0),
        ("DELETE FROM t", 0),
    ],
)
def test_seeded_sql_preserves_duplicates_perturbation_and_readonly(reference, query, expected):
    assert score_seeded_sql(reference, query)[0] == score(reference, query)[0] == expected


def test_sql_rounding_and_column_count_remain_source_owned(reference):
    task = json.loads(reference)
    task["reference_sql"] = "SELECT 1.0/3"
    assert score_seeded_sql(json.dumps(task), "SELECT 0.333333")[0] == 1
    assert score_seeded_sql(reference, "SELECT x,x FROM t")[0] == 0


def test_seeded_sql_actual_environment_reports_wrong_candidate_and_verifier_failure(reference):
    extras = {"reward_model": {"ground_truth": reference}}
    env = skyrl_gym.make("text_to_sql", env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras)
    assert env.step("SELECT x FROM t")["reward"] == 1
    assert env.step("SELECT 99")["reward"] == 0
    task = json.loads(reference)
    task["reference_sql"] = "SELECT absent_column FROM t"
    bad = skyrl_gym.make(
        "text_to_sql",
        env_config=OmegaConf.create({"verifyit_enabled": True}),
        extras={"reward_model": {"ground_truth": json.dumps(task)}},
    )
    result = bad.step("SELECT x FROM t")
    assert result["reward"] == 0
    assert result["verification"].status is VerificationStatus.ERROR
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"


def test_legacy_sql_keeps_set_semantics_format_projection_and_original_database(tmp_path):
    database = tmp_path / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES(1),(1),(2);")
    response = "<think>query</think><solution>SELECT DISTINCT x FROM t</solution>"
    assert score_legacy_sql(response, "SELECT x FROM t", str(database)) == 1
    assert score_legacy_sql("wrong format", "SELECT x FROM t", str(database)) == -1
    with pytest.raises(InvalidTask, match="reference query failed"):
        score_legacy_sql("wrong format", "SELECT missing FROM t", str(database))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM t").fetchone()[0] == 3


def test_invalid_reference_precedes_rejected_candidate(reference):
    task = json.loads(reference)
    task["reference_sql"] = "SELECT absent_column FROM t"
    with pytest.raises(InvalidTask, match="reference query failed"):
        score_seeded_sql(json.dumps(task), "DELETE FROM t")


def test_nonfinite_sql_results_fail_closed(reference):
    task = json.loads(reference)
    task["reference_sql"] = "SELECT 1e999"
    with pytest.raises(InvalidTask, match="nonfinite"):
        score_seeded_sql(json.dumps(task), "SELECT 1e999")
    assert score_seeded_sql(reference, "SELECT 1e999")[0] == 0


@pytest.mark.parametrize("ordered,score_value", [(True, 0.0), (False, 1.0)])
def test_order_policy_remains_explicit(reference, ordered, score_value):
    task = json.loads(reference)
    task.update(reference_sql="SELECT x FROM t ORDER BY x", order_significant=ordered)
    response = "SELECT x FROM t ORDER BY x DESC"
    assert score(json.dumps(task), response)[0] == score_seeded_sql(json.dumps(task), response)[0] == score_value


@pytest.mark.parametrize("query,reward", [("SELECT NULL AS x FROM t", 1.0), ("SELECT '' AS x FROM t", 0.0)])
def test_null_is_a_value_not_missing_evidence(reference, query, reward):
    task = json.loads(reference)
    task["reference_sql"] = "SELECT NULL AS x FROM t"
    assert score(json.dumps(task), query)[0] == score_seeded_sql(json.dumps(task), query)[0] == reward


def test_malformed_contract_fails_closed_at_actual_framework_boundary():
    env = skyrl_gym.make(
        "text_to_sql",
        env_config=OmegaConf.create({"verifyit_enabled": True}),
        extras={"reward_model": {"ground_truth": "{}"}},
    )
    result = env.step("SELECT 1")
    assert result["reward"] == 0.0
    assert result["verification"].status is VerificationStatus.ERROR
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"


@pytest.mark.parametrize("failure", ["invalid_reference", "missing_database"])
def test_legacy_sql_actual_environment_errors_use_minimum_reward_and_typed_status(tmp_path, failure):
    directory = tmp_path / "spider" / "database" / "fixture"
    directory.mkdir(parents=True)
    database = directory / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES(1);")
    env = skyrl_gym.make(
        "text2sql",
        env_config=OmegaConf.create({"verifyit_enabled": True, "db_path": str(tmp_path)}),
        extras={
            "db_id": "fixture",
            "data": "spider",
            "reward_spec": {
                "ground_truth": "SELECT missing FROM t" if failure == "invalid_reference" else "SELECT x FROM t"
            },
        },
    )
    if failure == "missing_database":
        database.unlink()
    try:
        result = env.step("<think>query</think><solution>SELECT x FROM t</solution>")
        assert result["reward"] == -1
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].score is None
        assert result["verification"].diagnostics["verifyit_status"] == (
            "invalid_task" if failure == "invalid_reference" else "infrastructure_error"
        )
    finally:
        env.close()
