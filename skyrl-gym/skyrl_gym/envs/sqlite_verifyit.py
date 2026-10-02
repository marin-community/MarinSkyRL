"""Named SQLite preparation policies; Exact, Schema and ALL own acceptance."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

from harbor_config.errors import error_category
from verifyit.bounded import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.spec import EmptyOutputPolicy, ExactSpec
from verifyit.preparation.sqlite import prepare_sqlite_row_set

from skyrl_gym.envs.text_to_sql import scoring as runtime


class SQLPolicy(StrEnum):
    SEEDED = "skyrl_seeded_round6_multiset_columns_readonly_finite_100000_reference_first_v1"
    LEGACY = "skyrl_legacy_set_numeric_readonly_finite_100000_reference_first_v1"


@dataclass(frozen=True)
class SQLInputs:
    ground_truth: object
    response: str
    database: str | None


@dataclass(frozen=True)
class SQLObservation:
    columns: int
    rows: tuple[tuple, ...]


@dataclass(frozen=True)
class SQLExecution:
    verdict: Reward
    protocol: Reward
    policy: SQLPolicy
    ordered: bool
    inputs: SQLInputs
    references: tuple[SQLObservation, ...]
    candidates: tuple[SQLObservation, ...]

    def diagnostics(self) -> dict:
        # Protected raw references remain in this verifier-owned result, never in
        # observations or rollout metadata. Digests identify the retained inputs.
        return {
            "verifyit_status": "infrastructure_error"
            if self.verdict.status is Status.INFRA_ERROR
            else self.verdict.status.value,
            "policy": str(self.policy),
            "core_status": self.verdict.status.value,
            "ordered": self.ordered,
            "input_sha256": hashlib.sha256(repr(self.inputs).encode()).hexdigest(),
            "verdict": {key: value for key, value in self.verdict.detail.items() if key != "error"},
        }


def structure_sql(ground_truth: object, response: str, database: str | None) -> SQLInputs:
    """Capture immutable raw task text and response before extraction."""
    return SQLInputs(ground_truth, response, database)


def structure_rows(result: tuple[int, list[tuple]]) -> SQLObservation:
    """Retain column count, scalar types, multiplicity and row order."""
    return SQLObservation(result[0], tuple(tuple(row) for row in result[1]))


def prepare_rows(observation: SQLObservation, policy: SQLPolicy, ordered: bool) -> str:
    """Apply the selected source numeric, multiplicity and column-count policy."""
    if policy is SQLPolicy.LEGACY:
        return prepare_sqlite_row_set(observation.rows)
    encoded = []
    for row in observation.rows:
        values = []
        for value in row:
            if isinstance(value, (int, float)):
                if policy is SQLPolicy.SEEDED and isinstance(value, float):
                    value = round(value, 6)
                number = Fraction(value)
                values.append(["number", str(number.numerator), str(number.denominator)])
            elif isinstance(value, bytes):
                values.append(["bytes", value.hex()])
            else:
                values.append([type(value).__name__, value])
        encoded.append(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
    if not ordered:
        encoded.sort()
    return json.dumps(
        [observation.columns, encoded] if policy is SQLPolicy.SEEDED else encoded,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def legacy_protocol(response: str) -> tuple[dict, str]:
    """Observe source tags; Schema decides protocol acceptance."""
    prefix, start, tail = response.partition("<solution>")
    solution, end, _ = tail.partition("</solution>")
    observations = {
        "starts": response.count("<solution>"),
        "ends": tail.count("</solution>"),
        "nested_tags": len(re.findall(r"</?(think|sql|observation)\b", solution, re.I)),
        "thoughts": len(re.findall(r"<think>(.*?)</think>", response, re.S)),
        "observation_followups": [
            prefix[match.end() :].lstrip()[:7].lower() for match in re.finditer(r"</observation>", prefix, re.I)
        ],
    }
    return observations, solution.strip() if start and end else ""


def prepare_candidate(inputs: SQLInputs, policy: SQLPolicy) -> tuple[Reward, str]:
    if policy is SQLPolicy.LEGACY:
        observations, statement = legacy_protocol(inputs.response)
        schema = {
            "type": "object",
            "required": list(observations),
            "properties": {
                "starts": {"const": 1},
                "ends": {"const": 1},
                "nested_tags": {"const": 0},
                "thoughts": {"type": "integer", "minimum": 1},
                "observation_followups": {"type": "array", "items": {"const": "<think>"}},
            },
        }
    else:
        extracted = runtime.extract_sql(inputs.response)
        allowed, statement = runtime.guard_candidate_sql(extracted)
        observations = {"read_only_statement": allowed}
        schema = {
            "type": "object",
            "required": ["read_only_statement"],
            "properties": {"read_only_statement": {"const": True}},
        }
    return grade_json_schema_candidate(schema, observations), statement


def _query_error(error: sqlite3.Error) -> bool:
    # Only SQL rejection is candidate/task data. Unknown codes, disk, memory,
    # locking, corruption and I/O faults must reach the infrastructure boundary.
    code = getattr(error, "sqlite_errorcode", -1)
    return (code & 255) in {
        sqlite3.SQLITE_ERROR,
        sqlite3.SQLITE_AUTH,
        sqlite3.SQLITE_INTERRUPT,
        sqlite3.SQLITE_CONSTRAINT,
        sqlite3.SQLITE_MISMATCH,
    } or isinstance(error, sqlite3.ProgrammingError)


def _bounded_rows(observation: SQLObservation) -> Reward:
    # Schema's finite-number traversal inspects the raw floats, before rounding.
    return grade_json_schema_candidate({"type": "array", "maxItems": 100_000}, [list(row) for row in observation.rows])


def _reference(connection, sql: str) -> SQLObservation:
    try:
        reference = structure_rows(runtime._run_query(connection, sql, read_only=True))
    except sqlite3.Error as error:
        if not _query_error(error):
            raise
        raise InvalidTask("reference query failed") from error
    if _bounded_rows(reference).reward != 1:
        raise InvalidTask("reference result exceeds bounds or contains nonfinite values")
    return reference


def _comparison(
    connection, reference: SQLObservation, statement: str, protocol: Reward, policy: SQLPolicy, ordered: bool
) -> tuple[Reward, SQLObservation]:
    result = SQLObservation(0, ())
    executed = False
    if protocol.reward == 1:
        try:
            result = structure_rows(runtime._run_query(connection, statement, read_only=True))
            executed = True
        except sqlite3.Error as error:
            if not _query_error(error):
                raise
    admission = grade_json_schema_candidate({"type": "boolean", "const": True}, executed)
    bounds = _bounded_rows(result)
    grades = [protocol, admission, bounds]
    if bounds.reward == 1:
        grades.append(
            grade_exact_candidate(
                ExactSpec(
                    expected=(prepare_rows(reference, policy, ordered),),
                    ignore_case=False,
                    ignore_whitespace=False,
                    strip_outer_whitespace=False,
                    empty_output=EmptyOutputPolicy.GRADE,
                ),
                prepare_rows(result, policy, ordered),
            )
        )
    return aggregate_rewards(grades, expected_total=4, policy=Aggregation.ALL), result


def _execute(ground_truth: object, response: str, database: str | None, policy: SQLPolicy) -> SQLExecution:
    inputs = structure_sql(ground_truth, response, database)
    references, candidates = [], []
    ordered = False
    stage = "task_preparation"
    protocol = grade_json_schema_candidate({"const": True}, False)
    try:
        policy = SQLPolicy(policy)
        with ExitStack() as stack:
            if policy is SQLPolicy.SEEDED:
                try:
                    task, creates, inserts, reference_sql = runtime._validated_ground_truth(inputs.ground_truth)
                    connection = runtime.build_db(creates, inserts)
                    stack.callback(connection.close)
                    # Perturbation requires actual hidden rowids, not user columns.
                    if any(row[4] for row in connection.execute("PRAGMA table_list") if row[2] == "table"):
                        raise InvalidTask("seeded perturbation requires rowid tables")
                except ValueError as error:
                    raise InvalidTask("invalid database reference") from error
                except sqlite3.Error as error:
                    if not _query_error(error):
                        raise
                    raise InvalidTask("invalid database reference") from error
                clone = runtime._clone_db(connection)
                stack.callback(clone.close)
                try:
                    runtime.perturb_db(clone)
                except sqlite3.Error as error:
                    if not _query_error(error):
                        raise
                    raise InvalidTask("unsupported database perturbation") from error
                connections, ordered = [connection, clone], task["order_significant"]
            else:
                runtime._QUERY_DEADLINE = 30.0
                connection = sqlite3.connect(Path(inputs.database).resolve().as_uri() + "?mode=ro", uri=True)
                stack.callback(connection.close)
                connections, reference_sql = [connection], inputs.ground_truth
                if not isinstance(reference_sql, str):
                    raise InvalidTask("reference SQL must be text")
            stage = "reference_execution"
            references = [_reference(connection, reference_sql) for connection in connections]
            stage = "candidate_preparation"
            protocol, statement = prepare_candidate(inputs, policy)
            stage = "candidate_execution"
            grades = []
            for connection, reference in zip(connections, references, strict=True):
                grade, result = _comparison(connection, reference, statement, protocol, policy, ordered)
                grades.append(grade)
                candidates.append(result)
            verdict = aggregate_rewards(grades, expected_total=len(connections), policy=Aggregation.ALL)
    except Exception as error:
        verdict = _failure(error, stage)
    return SQLExecution(verdict, protocol, policy, ordered, inputs, tuple(references), tuple(candidates))


def _failure(error: Exception, stage: str) -> Reward:
    return finalize_preparation_failure(
        status=Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR,
        category=error_category(type(error).__name__),
        error_type=type(error).__name__,
        message=str(error),
        stage=stage,
    )


def grade_sql(
    ground_truth: object,
    response: str,
    database: str | None = None,
    *,
    policy: SQLPolicy = SQLPolicy.SEEDED,
    timeout: float = 90,
) -> SQLExecution:
    """Bound capture, extraction, database preparation and core grading together."""
    try:
        return call_bounded(_execute, ground_truth, response, database, policy, timeout=timeout)
    except Exception as error:
        return SQLExecution(
            _failure(error, "bounded_sql_execution"),
            grade_json_schema_candidate({"const": True}, False),
            policy,
            False,
            SQLInputs(ground_truth, response, database),
            (),
            (),
        )


def project_sql_reward(result: SQLExecution) -> float:
    """Project core acceptance to the declared source reward range."""
    if result.policy is SQLPolicy.LEGACY and (
        result.verdict.status is not Status.SCORED or result.protocol.reward == 0
    ):
        return -1.0
    return result.verdict.reward


def score_seeded_sql(ground_truth: str, response: str) -> tuple[float, dict]:
    result = grade_sql(ground_truth, response)
    if result.verdict.status is Status.INVALID_TASK:
        raise InvalidTask(result.verdict.detail["error"])
    if result.verdict.status is not Status.SCORED:
        raise RuntimeError("SQL verification failed")
    return project_sql_reward(result), result.diagnostics()


def score_legacy_sql(response: str, reference: str, database: str) -> float:
    result = grade_sql(reference, response, database, policy=SQLPolicy.LEGACY)
    if result.verdict.status is Status.INVALID_TASK:
        raise InvalidTask(result.verdict.detail["error"])
    if result.verdict.status is not Status.SCORED:
        raise RuntimeError("SQL verification failed")
    return project_sql_reward(result)
