"""Execute source SQLite tools and translate their outputs into core contracts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path


def _canonical(rows: list[tuple], *, multiset: bool, ordered: bool) -> str:
    encoded = []
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, (int, float)):
                try:
                    number = Fraction(value)
                    values.append(["number", str(number.numerator), str(number.denominator)])
                except (ValueError, OverflowError):
                    values.append(["nonfinite", repr(value)])
            elif isinstance(value, bytes):
                values.append(["bytes", value.hex()])
            else:
                values.append([type(value).__name__, value])
        encoded.append(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
    if not multiset:
        encoded = list(set(encoded))
    if not ordered:
        encoded.sort()
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


def _reference(runtime, connection, sql):
    from verifyit.grade import InvalidTask
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate

    try:
        reference = runtime._run_query(connection, sql, read_only=True)
    except (sqlite3.Error, ValueError) as error:
        raise InvalidTask("reference query failed") from error
    verdict = grade_json_schema_candidate(
        {"type": "array", "maxItems": runtime._MAX_RESULT_ROWS}, [list(row) for row in reference[1]]
    )
    if verdict.reward != 1:
        raise InvalidTask("reference result exceeds bounds or contains nonfinite values")
    return reference


def _comparison(runtime, connection, reference, candidate, *, allowed: bool, seeded: bool, ordered: bool):
    from verifyit.grade import Aggregation, aggregate_rewards
    from verifyit.modes.grade_exact import grade_exact_candidate
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate
    from verifyit.spec import ExactSpec

    result = (None, [])
    executed = False
    if allowed:  # Tool authorization; SQLite's read-only authorizer is also active.
        try:
            result = runtime._run_query(connection, candidate, read_only=True)
            executed = True
        except sqlite3.Error:
            pass
    protocol = grade_json_schema_candidate(
        {
            "type": "object",
            "properties": {
                "executed": {"const": True},
                "rows": {"type": "array", "maxItems": runtime._MAX_RESULT_ROWS},
            },
        },
        {"executed": executed, "rows": [list(row) for row in result[1]]},
    )
    values = []
    for columns, rows in (reference, result):
        if seeded:
            normalized = runtime._norm_rows(rows)
            rows = [tuple(cell[1] if cell[0] == "~num" else cell for cell in row) for row in normalized]
        value = _canonical(rows, multiset=seeded, ordered=ordered)
        values.append(json.dumps([columns, value]) if seeded else value)
    equality = grade_exact_candidate(
        ExactSpec(expected=(values[0],), ignore_case=False, ignore_whitespace=False, strip_outer_whitespace=False),
        values[1],
    )
    return aggregate_rewards([protocol, equality], expected_total=2, policy=Aggregation.ALL)


def _execute(data: dict, database: str | None):
    from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards
    from skyrl_gym.envs.text_to_sql import scoring as runtime

    with ExitStack() as stack:
        seeded = data["profile"] == "seeded"
        if seeded:
            try:
                task, creates, inserts, reference_sql = runtime._validated_ground_truth(data["ground_truth"])
                connection = runtime.build_db(creates, inserts)
            except (ValueError, sqlite3.Error) as error:
                raise InvalidTask("invalid database reference") from error
            stack.callback(connection.close)
            clone = runtime._clone_db(connection)
            stack.callback(clone.close)
            runtime.perturb_db(clone)
            connections = [connection, clone]
            ordered = task["order_significant"]
            allowed, statement = runtime.guard_candidate_sql(data["candidate"])
        else:
            runtime._QUERY_DEADLINE = 30.0
            connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
            stack.callback(connection.close)
            connections, reference_sql, ordered = [connection], data["reference_sql"], False
            allowed, statement = data["format_valid"], data["candidate"]
        # Validate every trusted result before interpreting candidate outcomes.
        references = [_reference(runtime, connection, reference_sql) for connection in connections]
        grades = [
            _comparison(runtime, connection, reference, statement, allowed=allowed, seeded=seeded, ordered=ordered)
            for connection, reference in zip(connections, references, strict=True)
        ]
        return aggregate_rewards(grades, expected_total=len(connections), policy=Aggregation.ALL)


def _client(data: dict, database: str | None = None):
    from verifyit.bounded import call_bounded

    return call_bounded(_execute, data, database, timeout=90)


def score_seeded_sql(ground_truth: str, response: str) -> tuple[float, dict]:
    from verifyit.grade import InvalidTask, Status
    from skyrl_gym.envs.text_to_sql.scoring import extract_sql

    try:
        verdict = _client({"profile": "seeded", "ground_truth": ground_truth, "candidate": extract_sql(response)})
    except InvalidTask:
        raise
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, sqlite3.Error) as error:
        raise RuntimeError("SQL verification failed") from error
    if verdict.status is not Status.SCORED:
        raise RuntimeError(f"SQL verification failed ({verdict.status.value})")
    return verdict.reward, {"detail": "match" if verdict.reward else "query_result_mismatch"}


def score_legacy_sql(response: str, reference: str, database: str) -> float:
    from verifyit.grade import InvalidTask, Status
    from skyrl_gym.envs.sql.utils import verify_format_and_extract

    valid, _, candidate, _ = verify_format_and_extract(response)
    try:
        verdict = _client(
            {"profile": "legacy", "reference_sql": reference, "candidate": candidate, "format_valid": valid}, database
        )
    except InvalidTask:
        raise
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, sqlite3.Error) as error:
        raise RuntimeError("SQL verification failed") from error
    if verdict.status is Status.INVALID_TASK:
        raise InvalidTask("SQL trusted contract is invalid")
    if verdict.status is not Status.SCORED:
        raise RuntimeError(f"SQL verification failed ({verdict.status.value})")
    return -1.0 if not valid else verdict.reward
