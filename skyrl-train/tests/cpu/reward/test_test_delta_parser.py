"""Stage C (F2) — in-trajectory test-result parser unit tests.

Validates: extract_test_runs parses REAL test-runner stdout (pytest / unittest /
jest) from tool-observation messages, returns ordered (pass, fail) per run, NEVER
parses assistant prose, and falls back to no-signal (empty list) on
unrecognized / garbled / partial output.

Run:
    pytest tests/cpu/reward/test_test_delta_parser.py
"""

from skyrl_train.utils.test_delta_parser import extract_test_runs
from skyrl_train.utils.reward_shaping import JestOutputParser, get_output_parser


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
    p = JestOutputParser().parse(JEST_GREEN)
    assert p is not None and p.passed == 13 and p.failed == 0 and p.total == 13


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


# ---------------------------------------------------------------------------
# Safety: never parse the model's prose
# ---------------------------------------------------------------------------


def test_assistant_prose_is_never_parsed():
    # The assistant CLAIMS all tests pass; there is no real test-runner output.
    history = [
        _user("task"),
        _assistant("I ran pytest and got: 141 passed in 2.5s. All tests pass!"),
    ]
    runs = extract_test_runs(history)
    assert runs == []


def test_assistant_with_summary_string_ignored():
    # Even if the assistant message text contains a pytest-summary-looking line,
    # it is NOT an observation -> not parsed.
    history = [_assistant("============== 5 passed in 1.0s ==============")]
    assert extract_test_runs(history) == []


# ---------------------------------------------------------------------------
# No-signal fallback: unrecognized / partial / garbled output
# ---------------------------------------------------------------------------


def test_unrecognized_framework_no_signal():
    # cargo test output is not pytest/unittest/jest -> no-signal.
    cargo = "running 3 tests\ntest tests::a ... ok\ntest result: ok. 3 passed; 0 failed"
    runs = extract_test_runs([_tool(cargo)])
    assert runs == []


def test_garbled_output_no_signal():
    runs = extract_test_runs([_tool("Segmentation fault (core dumped)\n\x00\x01garbage")])
    assert runs == []


def test_non_test_command_output_no_signal():
    # Plain ls / cat output must not be mistaken for a test run.
    runs = extract_test_runs([_tool("foo.py  bar.py  README.md")])
    assert runs == []


def test_collection_error_no_signal():
    # pytest collection error -> the pytest parser returns None -> no-signal.
    err = "ERROR collecting test_x.py\nImportError: no module named foo\n!!! Interrupted: 1 error during collection !!!"
    runs = extract_test_runs([_tool(err)])
    assert runs == []


def test_empty_and_malformed_history():
    assert extract_test_runs(None) == []
    assert extract_test_runs([]) == []
    assert extract_test_runs(["not a dict", 42, None]) == []
    assert extract_test_runs([{"role": "tool"}]) == []  # no content


def test_list_content_observation():
    # OpenAI-style structured content parts.
    msg = {"role": "tool", "content": [{"type": "text", "text": PYTEST_GREEN}]}
    runs = extract_test_runs([msg])
    assert len(runs) == 1 and runs[0].frac_passing == 1.0


def test_jest_registered_in_parser_registry():
    assert get_output_parser("jest").name() == "jest"
