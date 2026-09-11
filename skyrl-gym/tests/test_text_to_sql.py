import json

import pytest
from omegaconf import DictConfig

import skyrl_gym
from skyrl_gym.envs.text_to_sql import scoring

# question: "How many hospitals are there in each state?"
_HOSPITALS = {
    "schema_sql": "CREATE TABLE Hospitals (HospitalID INT, HospitalName TEXT, State TEXT);",
    "insert_sql": (
        "INSERT INTO Hospitals VALUES "
        "(1,'A','CA'),(2,'B','CA'),(3,'C','NY'),(4,'D','NY'),(5,'E','TX'),(6,'F','TX'),(7,'G','TX');"
    ),
    "reference_sql": "SELECT State, COUNT(*) FROM Hospitals GROUP BY State",
    "order_significant": False,
    "table_names": ["Hospitals"],
}
# question: "List incident categories ordered by count, descending."
_ORDERED = {
    "schema_sql": "CREATE TABLE SI (cat TEXT, cnt INT);",
    "insert_sql": "INSERT INTO SI VALUES ('a',30),('b',25),('c',18),('d',15),('e',12),('f',5);",
    "reference_sql": "SELECT cat, cnt FROM SI ORDER BY cnt DESC",
    "order_significant": True,
    "table_names": ["SI"],
}
_LARGE_INTEGER = {
    "schema_sql": "CREATE TABLE Numbers (value INTEGER);",
    "insert_sql": "INSERT INTO Numbers VALUES (1152921504606846976);",
    "reference_sql": "SELECT value FROM Numbers",
    "order_significant": False,
    "table_names": ["Numbers"],
}


def _make(ground_truth: dict) -> object:
    return skyrl_gym.make(
        "text_to_sql",
        env_config=DictConfig({"env_class": "text_to_sql"}),
        extras={"reward_model": {"ground_truth": json.dumps(ground_truth)}},
    )


@pytest.mark.parametrize(
    "ground_truth, response, expected",
    [
        # exact reference -> 1
        (_HOSPITALS, "<solution>SELECT State, COUNT(*) FROM Hospitals GROUP BY State</solution>", 1.0),
        # equivalent rewrite (alias + reordered clauses) -> 1
        (_HOSPITALS, "```sql\nSELECT State, COUNT(*) AS n FROM Hospitals GROUP BY 1\n```", 1.0),
        # wrong: missing GROUP BY -> different row count -> 0
        (_HOSPITALS, "<solution>SELECT State, COUNT(*) FROM Hospitals</solution>", 0.0),
        # wrong dialect / bad column -> candidate error -> 0
        (_HOSPITALS, "<solution>SELECT State, YEAR(HospitalID) FROM Hospitals</solution>", 0.0),
        # hard-coded literal rows -> passes seeded, fails perturbed -> 0
        (
            _HOSPITALS,
            "<solution>SELECT 'CA',2 UNION ALL SELECT 'NY',2 UNION ALL SELECT 'TX',3</solution>",
            0.0,
        ),
        # write attempt -> guard rejects -> 0
        (_HOSPITALS, "<solution>DELETE FROM Hospitals; SELECT 1</solution>", 0.0),
        # multiple statements -> guard rejects -> 0
        (_HOSPITALS, "<solution>SELECT 1; SELECT 2</solution>", 0.0),
        # ordered task: same rows, wrong order -> 0
        (_ORDERED, "<solution>SELECT cat, cnt FROM SI ORDER BY cnt ASC</solution>", 0.0),
        # ordered task: exact reference -> 1
        (_ORDERED, "<solution>SELECT cat, cnt FROM SI ORDER BY cnt DESC</solution>", 1.0),
        # adjacent 64-bit integers remain distinct instead of collapsing through float conversion
        (_LARGE_INTEGER, "<solution>SELECT 1152921504606846977</solution>", 0.0),
    ],
)
def test_step_reward(ground_truth, response, expected):
    env = _make(ground_truth)
    out = env.step(response)
    assert out["reward"] == expected
    assert out["done"] is True


def test_malformed_ground_truth_scores_zero_without_crashing():
    env = skyrl_gym.make(
        "text_to_sql",
        env_config=DictConfig({"env_class": "text_to_sql"}),
        extras={"reward_model": {"ground_truth": "not json"}},
    )
    out = env.step("<solution>SELECT 1</solution>")
    assert out["reward"] == 0.0
    assert out["done"] is True
    assert out["metadata"].get("verifier_error")

    env = _make({**_HOSPITALS, "schema_sql": 42})
    out = env.step("<solution>SELECT 1</solution>")
    assert out["reward"] == 0.0
    assert out["metadata"].get("verifier_error")


def test_grade_reports_infra_for_a_broken_task():
    verdict, _ = scoring.grade(
        {
            "schema_sql": "CREATE TABLE (;",
            "insert_sql": "INSERT INTO t VALUES (1)",
            "reference_sql": "SELECT 1",
            "order_significant": False,
        },
        "SELECT 1",
    )
    assert verdict == scoring.INFRA

    verdict, _ = scoring.grade(
        {
            "schema_sql": "CREATE TABLE keyed (id INTEGER PRIMARY KEY) WITHOUT ROWID;",
            "insert_sql": "INSERT INTO keyed VALUES (1), (2), (3);",
            "reference_sql": "SELECT id FROM keyed",
            "order_significant": False,
        },
        "SELECT id FROM keyed",
    )
    assert verdict == scoring.INFRA

    verdict, _ = scoring.grade(
        {**_HOSPITALS, "reference_sql": "SELECT no_such_col FROM no_such_table"},
        "SELECT 1",
    )
    assert verdict == scoring.INFRA
