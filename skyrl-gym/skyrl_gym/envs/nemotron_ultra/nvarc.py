# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI grid and transform verifiers ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

_COLORS = [str(index) for index in range(10)]

_SUBPROCESS_TEMPLATE = r'''
import io
import json
import signal
import sys

def _native(obj):
    import numpy as np
    if isinstance(obj, np.ndarray): return _native(obj.tolist())
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, list): return [_native(x) for x in obj]
    if isinstance(obj, tuple): return tuple(_native(x) for x in obj)
    if isinstance(obj, dict): return {k: _native(v) for k, v in obj.items()}
    return obj

class TransformTimeoutError(Exception): pass
def _timeout_handler(signum, frame):
    raise TransformTimeoutError("transform timed out")

_BANNED_BUILTINS = frozenset({
    "open", "input", "breakpoint", "help", "license", "credits", "copyright",
    "exit", "quit", "vars", "dir", "globals", "locals",
})
_BANNED_MODULES = frozenset({
    "os", "subprocess", "shutil", "pathlib", "builtins", "socket", "urllib",
    "requests", "http", "ftplib", "smtplib", "pickle", "shelve", "marshal",
    "importlib", "pkgutil", "ctypes", "multiprocessing", "threading", "signal",
    "tempfile", "fileinput", "codecs", "pty", "fcntl", "resource", "syslog",
    "asyncio", "concurrent",
})
_original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in _BANNED_MODULES:
        raise ImportError(f"Import of {name!r} is not allowed in sandbox")
    return _original_import(name, globals, locals, fromlist, level)

if isinstance(__builtins__, dict):
    safe = {k: v for k, v in __builtins__.items() if k not in _BANNED_BUILTINS}
else:
    safe = {k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in _BANNED_BUILTINS and not k.startswith("_")}
    safe["__name__"] = "__main__"
    safe["__doc__"] = None
safe["__import__"] = _restricted_import
safe["__builtins__"] = safe

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
try:
    namespace = {"__builtins__": safe}
    exec(%(code)r, namespace)
    if "transform" not in namespace:
        raise ValueError("No 'transform' function defined in code")
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(%(timeout)d)
    try:
        result = namespace["transform"](%(input)s)
    finally:
        signal.alarm(0)
    if hasattr(result, "detach") and callable(result.detach):
        result = result.detach().cpu().tolist()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(json.dumps({"success": True, "result": _native(result)}))
except Exception as error:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(json.dumps({"success": False, "error": f"{type(error).__name__}: {str(error)[:500]}"}))
'''


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _valid_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], list)
        and bool(value[0])
        and all(isinstance(row, list) and len(row) == len(value[0]) for row in value)
        and all(isinstance(cell, int) for row in value for cell in row)
    )


def parse_grid(text: str) -> list[list[int]] | None:
    """Match NVIDIA Board.from_text with the integer color palette."""
    text = _strip_thinking(text)
    if match := re.search(r"\\boxed\{(.+)\}", text, re.DOTALL):
        text = match.group(1)
    text = re.sub(r"[^\s\w]", "", text)
    text = re.sub(r"\b\w+\b", lambda match: match.group(0) if match.group(0) in _COLORS else "", text)
    grid = [[int(cell) for cell in line.split()] for line in text.split("\n") if line.strip()]
    return grid if _valid_grid(grid) else None


def _extract_python(text: str) -> str | None:
    text = _strip_thinking(text)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip() if "def transform" in text else None


def _execute_python(code: str, input_grid: list[list[int]], timeout_seconds: int) -> list[list[int]] | None:
    script = _SUBPROCESS_TEMPLATE % {
        "code": code,
        "input": json.dumps(input_grid),
        "timeout": timeout_seconds,
    }
    try:
        process = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 5,
        )
        if process.returncode != 0 or not process.stdout.strip():
            return None
        result = json.loads(process.stdout.strip())
        value = result.get("result") if result.get("success") else None
        return value if _valid_grid(value) else None
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def grade_nvarc(
    text: str,
    record: dict[str, Any],
    *,
    inductive: bool,
    python_timeout_seconds: int = 30,
) -> tuple[float, dict[str, Any]]:
    if inductive:
        code = _extract_python(text)
        predicted = None if code is None else _execute_python(code, record["test_input"], python_timeout_seconds)
    else:
        predicted = parse_grid(text)
    correct = predicted is not None and predicted == record["expected_output"]
    return float(correct), {
        "agent_mode": "inductive" if inductive else "transductive",
        "extraction_successful": predicted is not None,
        "exact_match": correct,
        "predicted_output": predicted,
    }
