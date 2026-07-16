"""Unit tests for cloud/iris/literal_proxy_utils.py — RecordProxy wiring pure helpers.

Exercises the pure/disabled paths WITHOUT binding a socket or importing harbor/upath (both
lazy-imported only on the enabled/remote paths): the upstream-origin normalization (the
/v1-doubling guard), per-serve token uniqueness, the local log path, and that the DISABLED
wrapper is a null context manager (byte-identical handoff).

Run:
    python -m pytest cloud/iris/tests/test_literal_proxy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.literal_proxy_utils import (  # noqa: E402
    DEFAULT_LITERAL_PROXY_PORT,
    _slug,
    literal_log_path,
    literal_log_remote_uri,
    literal_proxy_endpoint,
    maybe_serve_literal_proxy,
    serve_token,
    upstream_origin,
)


def test_upstream_origin_strips_path_to_avoid_double_v1():
    assert upstream_origin("http://localhost:8000/v1") == "http://localhost:8000"
    assert upstream_origin("https://h:8443/v1/") == "https://h:8443"
    with pytest.raises(ValueError):
        upstream_origin("localhost:8000/v1")  # not absolute


def test_literal_proxy_endpoint_default_port():
    assert literal_proxy_endpoint() == f"http://127.0.0.1:{DEFAULT_LITERAL_PROXY_PORT}/v1"


def test_slug_is_filesystem_safe():
    assert _slug("rl/qwen3:8b run") == "rl-qwen3-8b-run"
    assert _slug("") == "job"


def test_serve_token_folds_in_task_attempt(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/0:2")
    assert serve_token().endswith("0-2")


def test_literal_log_path_local(tmp_path):
    p = literal_log_path(tmp_path, "myjob", "tok1")
    assert p.parent == tmp_path / "logs"
    assert p.name == "myjob__tok1_literal.jsonl"


def test_literal_log_remote_uri_none_for_local(tmp_path):
    assert literal_log_remote_uri(str(tmp_path), "myjob", "tok1") is None


def test_maybe_serve_disabled_is_null_context_manager():
    upstream = "http://localhost:8000/v1"
    with maybe_serve_literal_proxy(False, upstream, experiments_dir="/tmp/x", job_name="j") as ep:
        assert ep == upstream
