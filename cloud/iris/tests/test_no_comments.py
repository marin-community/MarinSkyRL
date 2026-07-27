"""Enforce no comments in YAML/TOML files and no >2 line comment blocks in Python.

YAML and TOML files must contain zero comment lines (enforced repo-wide).
Python files must not have more than 2 consecutive comment lines (enforced on
files changed in the current branch vs main, so existing code is grandfathered
and cleaned incrementally).
"""

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules", ".opencode"})
_YAML_EXTS = frozenset({".yaml", ".yml"})
_TOML_EXTS = frozenset({".toml"})


def _iter_files(extensions: frozenset[str]):
    for path in sorted(_REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def _changed_py_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=A", "main...HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=10,
        )
        return [(_REPO_ROOT / line.strip()) for line in result.stdout.splitlines() if line.strip().endswith(".py")]
    except Exception:
        return []


def _has_yaml_comment(text: str) -> list[str]:
    violations = []
    in_block_scalar = False
    block_indent = -1
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.lstrip()
        if not in_block_scalar:
            if re.search(r":\s*[|>][-+]?[0-9]*\s*$", line):
                in_block_scalar = True
                block_indent = len(line) - len(stripped)
                continue
        else:
            ci = len(line) - len(stripped) if stripped else block_indent + 1
            if stripped and ci <= block_indent and not stripped.startswith("#"):
                in_block_scalar = False
            else:
                continue
        if stripped.startswith("#"):
            violations.append(f"  L{i}: {line.strip()}")
    return violations


def _has_toml_comment(text: str) -> list[str]:
    violations = []
    in_ml = False
    delim = ""
    for i, line in enumerate(text.split("\n"), 1):
        if in_ml:
            if delim in line:
                in_ml = False
            continue
        if '"""' in line or "'''" in line:
            for d in ['"""', "'''"]:
                if d in line and line.count(d) == 1:
                    in_ml = True
                    delim = d
                    break
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            violations.append(f"  L{i}: {line.strip()}")
    return violations


def _find_long_comment_blocks(text: str, max_consecutive: int = 2) -> list[str]:
    violations = []
    consecutive = 0
    block_start = 0
    _DIRECTIVE_RE = re.compile(r"#\s*(noqa|type:\s*ignore|pragma|isort|file:)", re.I)
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#") and not _DIRECTIVE_RE.match(stripped):
            if consecutive == 0:
                block_start = i
            consecutive += 1
        else:
            if consecutive > max_consecutive:
                violations.append(
                    f"  L{block_start}-{block_start + consecutive - 1}: {consecutive} consecutive comment lines"
                )
            consecutive = 0
    if consecutive > max_consecutive:
        violations.append(f"  L{block_start}-{block_start + consecutive - 1}: {consecutive} consecutive comment lines")
    return violations


class TestNoYamlComments:
    def test_no_comments_in_yaml_files(self):
        all_violations = {}
        for path in _iter_files(_YAML_EXTS):
            v = _has_yaml_comment(path.read_text())
            if v:
                all_violations[str(path.relative_to(_REPO_ROOT))] = v
        if all_violations:
            msg = "YAML files must not contain comments:\n"
            for f, lines in all_violations.items():
                msg += f"\n{f}:\n" + "\n".join(lines) + "\n"
            pytest.fail(msg)


class TestNoTomlComments:
    def test_no_comments_in_toml_files(self):
        all_violations = {}
        for path in _iter_files(_TOML_EXTS):
            v = _has_toml_comment(path.read_text())
            if v:
                all_violations[str(path.relative_to(_REPO_ROOT))] = v
        if all_violations:
            msg = "TOML files must not contain comments:\n"
            for f, lines in all_violations.items():
                msg += f"\n{f}:\n" + "\n".join(lines) + "\n"
            pytest.fail(msg)


class TestNoLongPythonCommentBlocks:
    def test_no_more_than_2_consecutive_comment_lines(self):
        files = _changed_py_files()
        if not files:
            pytest.skip("No Python files changed in this branch")
        all_violations = {}
        for path in files:
            if not path.exists():
                continue
            v = _find_long_comment_blocks(path.read_text())
            if v:
                all_violations[str(path.relative_to(_REPO_ROOT))] = v
        if all_violations:
            msg = "Python files must not have >2 consecutive comment lines:\n"
            total = 0
            for f, lines in all_violations.items():
                msg += f"\n{f}:\n" + "\n".join(lines) + "\n"
                total += len(lines)
            msg += f"\n{total} violation(s) across {len(all_violations)} file(s)"
            pytest.fail(msg)
