"""Trusted SQLite execution harness composed with ScriptSpec and strict exact."""

from __future__ import annotations

import json
import math
import os
import shlex
import sqlite3
import sys
from fractions import Fraction
from pathlib import Path


def _canonical(rows: list[tuple], *, multiset: bool, ordered: bool) -> str:
    encoded = []
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, (int, float)):
                number = Fraction(value)
                values.append(["number", str(number.numerator), str(number.denominator)])
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


def _compare(runtime, connection, reference_sql: str, candidate: str, *, profile: str, ordered: bool) -> dict:
    from verifyit.modes.grade_exact import grade_exact_candidate
    from verifyit.spec import ExactSpec

    try:
        reference = runtime._run_query(connection, reference_sql, read_only=True)
        if len(reference[1]) > runtime._MAX_RESULT_ROWS:
            raise ValueError("reference exceeds row limit")
    except (sqlite3.Error, ValueError):
        return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "reference_query_failed"}}
    if any(isinstance(value, float) and not math.isfinite(value) for row in reference[1] for value in row):
        return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "nonfinite_reference_result"}}
    try:
        result = runtime._run_query(connection, candidate, read_only=True)
    except sqlite3.Error:
        return {"status": "scored", "reward": 0.0, "detail": {"reason": "candidate_query_failed"}}
    if any(isinstance(value, float) and not math.isfinite(value) for row in result[1] for value in row):
        return {"status": "scored", "reward": 0.0, "detail": {"reason": "nonfinite_candidate_result"}}
    if len(result[1]) > runtime._MAX_RESULT_ROWS:
        return {"status": "scored", "reward": 0.0, "detail": {"reason": "candidate_exceeds_row_limit"}}
    if profile == "seeded":
        expected_rows = runtime._norm_rows(reference[1])
        candidate_rows = runtime._norm_rows(result[1])
        # Normalized numeric tags preserve Python int/float equality independently.
        expected_rows = [tuple(cell[1] if cell[0] == "~num" else cell for cell in row) for row in expected_rows]
        candidate_rows = [tuple(cell[1] if cell[0] == "~num" else cell for cell in row) for row in candidate_rows]
        expected = json.dumps([reference[0], _canonical(expected_rows, multiset=True, ordered=ordered)])
        actual = json.dumps([result[0], _canonical(candidate_rows, multiset=True, ordered=ordered)])
    else:
        expected = _canonical(reference[1], multiset=False, ordered=False)
        actual = _canonical(result[1], multiset=False, ordered=False)
    spec = ExactSpec(expected=(expected,), ignore_case=False, ignore_whitespace=False, strip_outer_whitespace=False)
    verdict = grade_exact_candidate(spec, actual)
    return {
        "status": verdict.status.value,
        "reward": verdict.reward,
        "detail": {
            "reason": "match" if verdict.reward else "query_result_mismatch",
            "exact_spec": repr(spec),
            "exact_module": grade_exact_candidate.__code__.co_filename,
        },
    }


def _execute(data: dict, tests: Path) -> dict:
    import runtime

    profile = data["profile"]
    if profile == "seeded":
        try:
            reference, creates, inserts, reference_sql = runtime._validated_ground_truth(data["ground_truth"])
            connection = runtime.build_db(creates, inserts)
        except (ValueError, sqlite3.Error):
            return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "invalid_database_reference"}}
        try:
            reference_rows = runtime._run_query(connection, reference_sql, read_only=True)[1]
            if len(reference_rows) > runtime._MAX_RESULT_ROWS or any(
                isinstance(value, float) and not math.isfinite(value) for row in reference_rows for value in row
            ):
                raise ValueError("invalid reference result")
        except (sqlite3.Error, ValueError):
            connection.close()
            return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "reference_query_failed"}}
        accepted, statement = runtime.guard_candidate_sql(data["candidate"])
        if not accepted:
            connection.close()
            return {"status": "scored", "reward": 0.0, "detail": {"reason": "candidate_rejected"}}
        try:
            first = _compare(
                runtime, connection, reference_sql, statement, profile=profile, ordered=reference["order_significant"]
            )
            if first["status"] != "scored" or first["reward"] != 1:
                return first
            perturbed = runtime._clone_db(connection)
            try:
                runtime.perturb_db(perturbed)
                return _compare(
                    runtime,
                    perturbed,
                    reference_sql,
                    statement,
                    profile=profile,
                    ordered=reference["order_significant"],
                )
            finally:
                perturbed.close()
        finally:
            connection.close()
    runtime._QUERY_DEADLINE = 30.0
    runtime._MAX_RESULT_ROWS = 100_000
    connection = sqlite3.connect(f"file:{tests / 'fixture.sqlite'}?mode=ro", uri=True)
    try:
        if not data["format_valid"]:
            # Task validity precedes candidate format reward.
            try:
                runtime._run_query(connection, data["reference_sql"], read_only=True)
            except sqlite3.Error:
                return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "reference_query_failed"}}
            return {"status": "scored", "reward": 0.0, "detail": {"reason": "invalid_format", "format_valid": False}}
        return _compare(runtime, connection, data["reference_sql"], data["candidate"], profile=profile, ordered=False)
    finally:
        connection.close()


def _client(data: dict, database: str | None = None):
    from tempfile import TemporaryDirectory

    from verifyit.grade import run
    from verifyit.spec import ScriptSpec, render_spec

    from skyrl_gym.envs.text_to_sql import scoring

    with TemporaryDirectory(prefix="skyrl-sql-") as directory:
        root = Path(directory)
        (root / "checker.py").write_text(Path(__file__).read_text())
        (root / "checker.sh").write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(root / "checker.py")) + "\n"
        )
        (root / "runtime.py").write_text(Path(scoring.__file__).read_text())
        (root / "data.json").write_text(json.dumps(data, allow_nan=False))
        if database is not None:
            with (
                sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as original,
                sqlite3.connect(root / "fixture.sqlite") as snapshot,
            ):
                original.backup(snapshot)
        (root / "verifier.toml").write_text(
            render_spec(ScriptSpec(path="checker.sh", verdict_file="sql-result.json", timeout=90))
        )
        return run(root / "verifier.toml", root)


def _score_seeded_sql(ground_truth: str, response: str) -> tuple[float, dict]:
    from verifyit.grade import Status

    from skyrl_gym.envs.text_to_sql.scoring import extract_sql

    verdict = _client({"profile": "seeded", "ground_truth": ground_truth, "candidate": extract_sql(response)})
    if verdict.status is not Status.SCORED:
        raise RuntimeError(f"SQL verification failed ({verdict.status.value})")
    return verdict.reward, {"detail": verdict.detail["reason"]}


def _score_legacy_sql(response: str, reference: str, database: str) -> float:
    from verifyit.grade import Status

    from skyrl_gym.envs.sql.utils import verify_format_and_extract

    valid, _, candidate, _ = verify_format_and_extract(response)
    verdict = _client(
        {"profile": "legacy", "reference_sql": reference, "candidate": candidate, "format_valid": valid}, database
    )
    if verdict.status is not Status.SCORED:
        raise RuntimeError(f"SQL verification failed ({verdict.status.value})")
    return -1.0 if not valid else verdict.reward


def score_seeded_sql(ground_truth: str, response: str) -> tuple[float, dict]:
    try:
        return _score_seeded_sql(ground_truth, response)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, sqlite3.Error) as error:
        raise RuntimeError("SQL verification failed") from error


def score_legacy_sql(response: str, reference: str, database: str) -> float:
    try:
        return _score_legacy_sql(response, reference, database)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, sqlite3.Error) as error:
        raise RuntimeError("SQL verification failed") from error


if __name__ == "__main__":
    tests = Path(os.environ["VERIFYIT_TESTS_DIR"])
    logs = Path(os.environ["VERIFYIT_LOGS_DIR"])
    try:
        verdict = _execute(json.loads((tests / "data.json").read_text()), tests)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        verdict = {"status": "infra_error", "reward": 0.0, "detail": {"reason": "sql_runtime_failure"}}
    (logs / "sql-result.json").write_text(json.dumps(verdict, allow_nan=False))
