"""Every rollout producer keys ``session_id`` on its trajectory.

``session_id`` selects an inference engine and is popped before dispatch, so a wrong key routes
differently and returns the same tokens -- there is no runtime surface to assert against. This
reads the sources instead, and finds the producers by scanning rather than by a fixed list, so a
new one is covered the day it appears.
"""

import ast
from pathlib import Path

import pytest

_RUNNERS = Path(__file__).resolve().parents[2].parent / "skyrl_train" / "trajectory_runners"


def _session_id_assignments(source: Path) -> list[ast.expr]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "session_id"
    ]


def _producers() -> list[Path]:
    return sorted(source for source in _RUNNERS.rglob("*.py") if _session_id_assignments(source))


def _keys_on_the_trajectory(value: ast.expr) -> bool:
    call = value.body if isinstance(value, ast.IfExp) else value
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "to_string"


def test_the_scan_finds_the_known_producers():
    """A scan that silently matches nothing would pass every other check in this module."""
    found = {source.name for source in _producers()}
    assert {"runner.py", "skyrl_gym.py", "step_wise.py"} <= found, found


@pytest.mark.parametrize("producer", _producers(), ids=lambda source: source.name)
def test_every_producer_keys_the_session_on_the_trajectory(producer):
    wrong = [ast.unparse(value) for value in _session_id_assignments(producer) if not _keys_on_the_trajectory(value)]
    assert not wrong, f"{producer.name}: session_id not keyed on the trajectory: {wrong}"
