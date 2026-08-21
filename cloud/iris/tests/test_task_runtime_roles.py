"""Which role each node's Ray metric collector reports under.

`run_head` and `run_worker` both open the collector. Opening both under one role makes every
`GROUP BY role` on an RL run wrong — a ten-node job reports one head and nine workers as ten
controllers.

Run:
    python -m pytest cloud/iris/tests/test_task_runtime_roles.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (_REPO_ROOT, _REPO_ROOT / "skyrl-train"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cloud.iris.task_runtime as task_runtime  # noqa: E402
from skyrl_train import ray_metrics  # noqa: E402

# task_runtime defines its own role constants in an ImportError fallback for environments
# without skyrl_train. Asserting against those would pass while testing nothing, so fail loudly
# here instead of silently checking the wrong module.
assert not hasattr(task_runtime, "_RAY_METRICS_UNAVAILABLE_REASON"), (
    "skyrl_train did not import; this test would assert against task_runtime's fallback constants"
)


def _collector_roles() -> dict[str, str]:
    """The role each `ray_metrics_telemetry` call site opens with, keyed by enclosing function.

    Resolved to the constant's *value*, so renaming the constant does not fail this. Only a
    static read can see which argument a call site passes: the two sites are in `run_head` and
    `run_worker`, and neither is reachable in a unit test without standing up Ray.
    """
    tree = ast.parse(Path(task_runtime.__file__).read_text())
    roles = {}
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            called = node.func if isinstance(node, ast.Call) else None
            if isinstance(called, ast.Name) and called.id == "ray_metrics_telemetry" and node.args:
                argument = node.args[-1]
                if isinstance(argument, ast.Name):
                    roles[function.name] = getattr(task_runtime, argument.id)
                elif isinstance(argument, ast.Constant):
                    roles[function.name] = argument.value
    return roles


def test_the_head_and_a_worker_open_their_collectors_under_different_roles() -> None:
    roles = _collector_roles()

    assert roles["run_head"] == "controller"
    assert roles["run_worker"] == "worker"


def test_the_role_reaches_the_telemetry_lifecycle(monkeypatch) -> None:
    entered: list[str] = []

    class _Owner:
        def collector_or_inert(self, collector):
            del collector
            return _Inert()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class _Inert:
        def start(self) -> None: ...

        def stop(self, *, timeout: float) -> None:
            del timeout

    def fake_process_telemetry(role: str):
        entered.append(role)
        return _Owner()

    monkeypatch.setattr(ray_metrics, "process_telemetry", fake_process_telemetry)

    with ray_metrics.ray_metrics_telemetry("10.0.0.1", 8080, "worker"):
        pass

    assert entered == ["worker"]


def test_the_roles_are_the_shared_vocabulary() -> None:
    # A local spelling would split the vocabulary across producers, so assert against rigging's
    # own enum rather than against a literal — a literal here would only restate the constant.
    from rigging.telemetry import TelemetryRole

    assert task_runtime.CONTROLLER_ROLE == TelemetryRole.CONTROLLER
    assert task_runtime.WORKER_ROLE == TelemetryRole.WORKER
