"""Unchanged Countdown generator and verifier from score-centering 7c56e9ee.

MIT License

Copyright (c) 2026 Martin Marek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import re

from collections import Counter

from fractions import Fraction

OPERATIONS = ("+", "-", "*", "/")


def _build_random_expr(rng, numbers, operations):
    """Combine `numbers` pairwise in random order/ops -> (Fraction value, expr)."""
    items = [(Fraction(n), str(n)) for n in numbers]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        av, ae = items[i]
        bv, be = items[j]
        op = rng.choice(operations)
        if op == "/" and bv == 0:
            op = "+"
        val = {"+": av + bv, "-": av - bv, "*": av * bv, "/": (av / bv if bv != 0 else av)}[op]
        items = [items[k] for k in range(len(items)) if k not in (i, j)]
        items.append((val, f"({ae} {op} {be})"))
    return items[0]


def generate(rng, num_operands=4, min_number=1, max_number=100, max_target=1000, operations=OPERATIONS, max_tries=200):
    """Generate ONE solvable instance: dict(numbers, target, solution).
    Re-rolls until the built expression has a positive-integer value in range."""
    for _ in range(max_tries):
        numbers = [rng.randint(min_number, max_number) for _ in range(num_operands)]
        val, exp = _build_random_expr(rng, numbers, list(operations))
        if val.denominator == 1 and 1 <= val <= max_target:
            return {"numbers": numbers, "target": int(val), "solution": exp[1:-1] if exp[0] == "(" else exp}
    raise RuntimeError(
        "no in-range integer target found in max_tries; widen max_target or narrow the number range / operations"
    )


SYSTEM_PROMPT = (
    "You are a helpful assistant. You first thinks about the "
    "reasoning process in the mind and then provides the user "
    "with the answer."
)


def instruction(numbers, target):
    return (
        f" Using the numbers {numbers}, create an equation that equals "
        f"{target}. You can use basic arithmetic operations (+, -, *, /) "
        f"and each number can only be used once. Show your work in "
        f"<think> </think> tags. And return the final answer in "
        f"<answer> </answer> tags, for example <answer> (1 + 2) / 3 "
        f"</answer>."
    )


def _messages(problem):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction(problem["numbers"], problem["target"])},
    ]


_ALLOWED = re.compile(r"^[\d\s+\-*/().]+$")


def extract_answer(text):
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m[-1].strip() if m else None


def compute_reward(completion, numbers, target):
    """1.0 iff the <answer> is a valid +,-,*,/ expression that uses each
    provided number exactly once and evaluates to target; else 0.0."""
    expr = extract_answer(completion)
    if expr is None or not _ALLOWED.match(expr) or "**" in expr or "//" in expr:
        return 0.0  # ** and // slip past the char class but aren't allowed ops
    if Counter(int(x) for x in re.findall(r"\d+", expr)) != Counter(numbers):
        return 0.0  # must use each provided number once (also catches bare-target hack)
    try:
        return 1.0 if abs(eval(expr, {"__builtins__": {}}, {}) - target) < 1e-9 else 0.0
    except Exception:
        return 0.0


def _rows(rng, n, **gen_args):
    return [
        {"prompt": _messages(p), "answer": p["solution"], "info": {"numbers": p["numbers"], "target": p["target"]}}
        for p in (generate(rng, **gen_args) for _ in range(n))
    ]
