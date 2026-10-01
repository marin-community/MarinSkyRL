"""Safe PRIME/DAPO answer comparison using verifyit's existing grading modes."""

import ast
import math
import operator
import re

from verifyit.grade import InvalidTask, Status
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_math import grade_math_candidate, grade_numeric_candidate
from verifyit.spec import ExactSpec, MathSpec, MathType, NumericSpec


_UNSAFE = re.compile(r"__|[\";`]|\b(?:import|exec|eval|lambda|open|getattr)\b")
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _grade_math(spec, candidate):
    from math_verify.errors import TimeoutException

    try:
        return grade_math_candidate(spec, candidate)
    except TimeoutException as error:
        raise RuntimeError("math verifier timed out") from error


def _arithmetic(text, pi):
    """Interpret only bounded arithmetic nodes; never execute candidate Python."""
    text = re.sub(r"(?<=\d)\\pi", r"*pi", text).replace(r"\pi", "pi")
    tree = ast.parse(text, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("expression too large")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.Name) and node.id == "pi":
            value = pi
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd)
        ):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            value = _OPERATORS[type(node.op)](left, right)
        else:
            raise ValueError("not arithmetic")
        if not math.isfinite(value):
            raise ValueError("nonfinite arithmetic")
        return value

    return visit(tree.body)


def _number(text):
    return float(
        text.replace("{,}", "")
        .replace(",", "")
        .removesuffix(r"\%")
        .removesuffix("%")
        .removeprefix("$")
    )


def _balanced(text):
    text = text.strip()
    # The source allows half-open interval endpoints independently.
    if (text[:1], text[-1:]) in (("[", ")"), ("(", "]")):
        text = text[1:-1]
    stack = []
    closes = {")": "(", "]": "[", "}": "{"}
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in closes:
            if not stack or stack.pop() != closes[char]:
                return False
    return not stack


def unsafe_expression(text):
    return bool(_UNSAFE.search(text))


def normalize_for_compare(text, normalize):
    """Retain source string normalization without invoking its evaluators."""
    if unsafe_expression(text) or not _balanced(text):
        return None
    if text.startswith("Point(") and text.endswith(")"):
        return text[5:]
    # Preserve structural endpoints and matrix LaTeX for typed comparison.
    if (
        r"\begin{pmatrix}" in text
        or ("," in text and text.strip()[:1] in "[{(")
        or r"\pi" in text
    ):
        return text
    try:
        return normalize(text)
    except (ValueError, TypeError, OverflowError):
        return None


def compare_answer(candidate, expected):
    """Return binary correctness; failed verifiers never contribute credit."""
    if not isinstance(candidate, str) or not isinstance(expected, str):
        return False
    if (
        not expected.strip()
        or len(candidate) > 1000
        or _UNSAFE.search(candidate)
        or _UNSAFE.search(expected)
    ):
        return False
    if not _balanced(candidate) or not _balanced(expected):
        return False
    # Validate finite numeric references before any literal success path.
    try:
        target = _number(expected)
        if not math.isfinite(target):
            return False
    except ValueError:
        target = None
    try:
        value = _number(candidate)
        if not math.isfinite(value):
            return False
    except ValueError:
        value = None
    try:
        if target is not None and value is not None:
            for alternative in (target / 100, target, target * 100):
                verdict = grade_numeric_candidate(
                    NumericSpec(
                        expected=alternative,
                        tolerance_abs=1e-4 * max(abs(value), abs(alternative)),
                        tolerance_rel=0,
                    ),
                    value,
                )
                if verdict.status is not Status.SCORED:
                    return False
                if verdict.reward == 1:
                    return True
            return False
        if r"\begin{pmatrix}" in expected and (
            candidate.startswith("[") or candidate.startswith("Matrix(")
        ):
            text = (
                candidate[7:-1]
                if candidate.startswith("Matrix(") and candidate.endswith(")")
                else candidate
            )
            try:
                rows = ast.literal_eval(text)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return False
            if (
                not isinstance(rows, list)
                or not rows
                or not all(isinstance(row, list) and row for row in rows)
            ):
                return False
            if len({len(row) for row in rows}) != 1:
                return False
            if any(
                type(cell) not in (int, float) or not math.isfinite(cell)
                for row in rows
                for cell in row
            ):
                return False
            latex = (
                r"\begin{pmatrix}"
                + r"\\".join("&".join(str(cell) for cell in row) for row in rows)
                + r"\end{pmatrix}"
            )
            verdict = _grade_math(MathSpec(expected=expected), latex)
            return verdict.status is Status.SCORED and verdict.reward == 1
        # Source comma collections are ordered, even when written with braces.
        # Do not let symbolic set equality reorder them.
        if "," in candidate or "," in expected:
            left, right = candidate.strip(), expected.strip()
            brackets = {"[": "]", "(": ")", "{": "}"}
            if left[:1] in brackets or right[:1] in brackets:
                if left[:1] != right[:1] or left[-1:] != right[-1:]:
                    return False
                left, right = left[1:-1], right[1:-1]

            def parts(text):
                depth, start, result = 0, 0, []
                for index, char in enumerate(text):
                    if char in "([{":
                        depth += 1
                    elif char in ")]}":
                        depth -= 1
                    elif char == "," and depth == 0:
                        result.append(text[start:index].strip())
                        start = index + 1
                result.append(text[start:].strip())
                return result

            a, b = parts(left), parts(right)
            if len(a) == 1 or len(a) != len(b):
                return False
            return all(compare_answer(x, y) for x, y in zip(a, b))
        exact = grade_exact_candidate(
            ExactSpec(expected=(expected.replace(" ", ""),)), candidate.replace(" ", "")
        )
        if exact.status is not Status.SCORED:
            return False
        if exact.reward == 1:
            return True
        verdict = _grade_math(MathSpec(expected=expected), candidate)
        if verdict.status is not Status.SCORED:
            return False
        if verdict.reward == 1:
            return True
        if r"\pi" in candidate or r"\pi" in expected:
            for pi in (math.pi, 3.14):
                try:
                    left, right = _arithmetic(candidate, pi), _arithmetic(expected, pi)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                    continue
                verdict = grade_numeric_candidate(
                    NumericSpec(
                        expected=right,
                        tolerance_abs=1e-4 * max(abs(left), abs(right)),
                        tolerance_rel=0,
                    ),
                    left,
                )
                if verdict.status is not Status.SCORED:
                    return False
                if verdict.reward == 1:
                    return True
    except (
        InvalidTask,
        ImportError,
        RuntimeError,
        ValueError,
        TypeError,
        OverflowError,
    ):
        return False
    return False


def score_torl(response, expected, reward_type, boxed_pattern, normalize):
    """Preserve ToRL extraction and signed source shaping without raw-parser false positives."""
    if (
        not isinstance(response, str)
        or not isinstance(expected, str)
        or not expected.strip()
    ):
        return -1.0
    if unsafe_expression(response) or unsafe_expression(expected):
        return -1.0
    matches = boxed_pattern.findall(response)
    candidate = matches[-1][:-1] if matches else ""
    try:
        normalized_candidate, normalized_expected = normalize(candidate), normalize(
            expected
        )
        if not _balanced(normalized_candidate) or not _balanced(normalized_expected):
            return -1.0
        literal = grade_exact_candidate(
            ExactSpec(expected=(expected,), ignore_case=True, ignore_whitespace=False),
            candidate,
        )
        if literal.status is not Status.SCORED:
            return -1.0
        if matches and literal.reward == 1:
            return 1.0
        math_type = MathType.SCALAR
        if "," in normalized_expected:
            normalized_expected = normalized_expected.replace(r"\{", "{").replace(
                r"\}", "}"
            )
            normalized_candidate = normalized_candidate.replace(r"\{", "{").replace(
                r"\}", "}"
            )
            math_type = MathType.LIST
            opening = normalized_expected[:1]
            if opening in "([{":
                if (
                    normalized_candidate[:1] != opening
                    or normalized_candidate[-1:] != normalized_expected[-1:]
                ):
                    normalized_candidate = ""
                else:
                    normalized_candidate = normalized_candidate[1:-1]
                normalized_expected = normalized_expected[1:-1]
        verdict = _grade_math(
            MathSpec(expected=normalized_expected, math_type=math_type),
            normalized_candidate,
        )
        if verdict.status is not Status.SCORED:
            return -1.0
        if matches and verdict.reward == 1:
            return 1.0
        return -0.5 if matches and reward_type == "v2.wformat" else -1.0
    except (
        InvalidTask,
        ImportError,
        RuntimeError,
        ValueError,
        TypeError,
        OverflowError,
    ):
        return -1.0
