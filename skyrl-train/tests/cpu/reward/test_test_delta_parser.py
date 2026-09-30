"""Stage C (F2) — in-trajectory test-result parser unit tests.

Validates: extract_test_runs parses REAL test-runner stdout (pytest / unittest /
jest) from tool-observation messages, returns ordered (pass, fail) per run, NEVER
parses assistant prose, and falls back to no-signal (empty list) on
unrecognized / garbled / partial output.
"""

import pytest

from skyrl_train.utils.test_delta_parser import extract_test_runs


PYTEST_FAIL = "============== 1 failed, 140 passed in 2.39s =============="
PYTEST_GREEN = "============================= 141 passed in 2.51s ============================="
UNITTEST_FAIL = "Ran 5 tests in 0.003s\n\nFAILED (failures=2, errors=1)"
UNITTEST_OK = "Ran 5 tests in 0.002s\n\nOK"
JEST_FAIL = "Tests:       1 failed, 12 passed, 13 total"
JEST_GREEN = "Tests:       13 passed, 13 total"


def _tool(text):
    return {"role": "tool", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# Framework parsers (pytest / unittest / jest) on real stdout
# ---------------------------------------------------------------------------


def test_pytest_run_parsed():
    runs = extract_test_runs([_user("fix the bug"), _assistant("ok"), _tool(PYTEST_FAIL)])
    assert len(runs) == 1
    r = runs[0]
    assert r.framework == "pytest"
    assert r.passed == 140 and r.failed == 1
    assert r.total_runnable == 141
    assert abs(r.frac_passing - 140 / 141) < 1e-9


def test_unittest_run_parsed():
    runs = extract_test_runs([_tool(UNITTEST_FAIL)])
    assert len(runs) == 1
    assert runs[0].framework == "unittest"
    # Ran 5; failures=2, errors=1 -> 2 passed, 3 effective_failed.
    assert runs[0].passed == 2
    assert runs[0].failed == 3
    assert runs[0].total_runnable == 5


def test_jest_run_parsed():
    runs = extract_test_runs([_tool(JEST_FAIL)])
    assert len(runs) == 1
    assert runs[0].framework == "jest"
    assert runs[0].passed == 12 and runs[0].failed == 1
    assert runs[0].total_runnable == 13


def test_jest_green_parsed():
    runs = extract_test_runs([_tool(JEST_GREEN)])
    assert [(r.framework, r.passed, r.failed, r.total_runnable) for r in runs] == [("jest", 13, 0, 13)]


# ---------------------------------------------------------------------------
# Ordering: a sequence of edit -> test-run observations
# ---------------------------------------------------------------------------


def test_multiple_runs_ordered_with_indices():
    history = [
        _user("task"),
        _assistant("edit 1"),
        _tool(PYTEST_FAIL),  # idx 2: 140/141
        _assistant("edit 2"),
        _tool("============== 141 passed in 2.6s =============="),  # idx 4: green
    ]
    runs = extract_test_runs(history)
    assert len(runs) == 2
    assert runs[0].message_index == 2
    assert runs[1].message_index == 4
    assert runs[0].frac_passing < runs[1].frac_passing
    assert runs[1].frac_passing == 1.0


@pytest.mark.parametrize(
    "history",
    [
        # The assistant claims tests pass; only tool observations are test-runner output.
        pytest.param(
            [_user("task"), _assistant("I ran pytest and got: 141 passed in 2.5s. All tests pass!")],
            id="assistant_prose",
        ),
        pytest.param([_assistant("============== 5 passed in 1.0s ==============")], id="assistant_summary_line"),
        pytest.param(
            [_tool("running 3 tests\ntest tests::a ... ok\ntest result: ok. 3 passed; 0 failed")],
            id="unrecognized_framework",
        ),
        pytest.param([_tool("Segmentation fault (core dumped)\n\x00\x01garbage")], id="garbled"),
        pytest.param([_tool("foo.py  bar.py  README.md")], id="non_test_command"),
        pytest.param(
            [
                _tool(
                    "ERROR collecting test_x.py\nImportError: no module named foo\n"
                    "!!! Interrupted: 1 error during collection !!!"
                )
            ],
            id="collection_error",
        ),
    ],
)
def test_no_test_signal(history):
    assert extract_test_runs(history) == []


def test_list_content_observation():
    # OpenAI-style structured content parts.
    msg = {"role": "tool", "content": [{"type": "text", "text": PYTEST_GREEN}]}
    runs = extract_test_runs([msg])
    assert len(runs) == 1 and runs[0].frac_passing == 1.0
