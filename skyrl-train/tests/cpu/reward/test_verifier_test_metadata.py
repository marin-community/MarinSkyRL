from skyrl_train.utils.reward_shaping import parse_test_output_with_parser, verifier_test_collection


def test_named_test_lines_preserve_identity_outcome_and_literal_output():
    stdout = """\
test formatting: PASS: title is present
test semantics: FAIL: expected 4, got 3
Results: 1/2 passed
"""

    parsed, parser_name = parse_test_output_with_parser(stdout)
    collection = verifier_test_collection(
        parsed,
        parser_name=parser_name,
        instance_id="task-7",
        repetition_id=2,
    )

    assert collection["complete"] is True
    assert collection["parser"] == "pass_ratio_summary"
    assert [(test["test_id"], test["outcome"], test["output"]) for test in collection["tests"]] == [
        ("test formatting", "passed", "test formatting: PASS: title is present"),
        ("test semantics", "failed", "test semantics: FAIL: expected 4, got 3"),
    ]
    assert all(
        test["trial_id"] == {"instance_id": "task-7", "repetition_id": 2} for test in collection["tests"]
    )
    assert len({test["record_id"] for test in collection["tests"]}) == 2


def test_aggregate_only_output_does_not_fabricate_test_identities():
    parsed, parser_name = parse_test_output_with_parser("Results: 13/20 passed")

    collection = verifier_test_collection(
        parsed,
        parser_name=parser_name,
        instance_id="task-7",
        repetition_id=2,
    )

    assert collection == {"parser": "pass_ratio_summary", "complete": False, "tests": []}


def test_pytest_verbose_lines_are_complete_when_the_summary_agrees():
    stdout = """\
PASSED tests/test_math.py::test_addition
FAILED tests/test_math.py::test_subtraction - assert 3 == 4
========================= 1 failed, 1 passed in 0.10s =========================
"""

    parsed, parser_name = parse_test_output_with_parser(stdout)

    assert parser_name == "pytest"
    assert parsed is not None and parsed.tests_complete
    assert [(test.test_id, test.outcome.value, test.output) for test in parsed.tests] == [
        ("tests/test_math.py::test_addition", "passed", "PASSED tests/test_math.py::test_addition"),
        (
            "tests/test_math.py::test_subtraction",
            "failed",
            "FAILED tests/test_math.py::test_subtraction - assert 3 == 4",
        ),
    ]


def test_standard_pytest_verbose_lines_preserve_the_literal_result_line():
    stdout = """\
tests/test_math.py::test_addition PASSED [ 50%]
tests/test_math.py::test_subtraction FAILED [100%]
========================= 1 failed, 1 passed in 0.10s =========================
"""

    parsed, _ = parse_test_output_with_parser(stdout)

    assert parsed is not None and parsed.tests_complete
    assert [(test.test_id, test.outcome.value, test.output) for test in parsed.tests] == [
        ("tests/test_math.py::test_addition", "passed", "tests/test_math.py::test_addition PASSED [ 50%]"),
        ("tests/test_math.py::test_subtraction", "failed", "tests/test_math.py::test_subtraction FAILED [100%]"),
    ]
