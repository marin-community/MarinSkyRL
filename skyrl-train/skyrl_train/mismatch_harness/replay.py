"""Re-issue an output-code module's ``call()`` statement by statement and keep every stored buffer.

The module's own Triton kernels (compiled from the archived source) and ``extern_kernels`` calls run
exactly as ``call()`` launches them. Opaque custom ops (``torch.ops.<ns>.<op>``) are routed to
harness implementations, because they depend on vLLM's forward context. A hook may overwrite buffers
before any statement, which is how the harness feeds one region the trainer's tensors.
"""

from __future__ import annotations

import ast
import copy
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from skyrl_train.mismatch_harness.output_code import Launch, LaunchKind, OutputCode

OP_PREFIX = "__harness_op__"

BeforeStatement = Callable[[int, dict[str, Any]], None]


@dataclass(frozen=True)
class StoredValue:
    """A buffer as a launch left it, cloned right after the launch."""

    launch: Launch
    variable: str
    tensor: torch.Tensor
    data_ptr: int
    """Address of the buffer the launch wrote, to match views of it that ``call()`` returns."""


def op_binding(target: str) -> str:
    return OP_PREFIX + target.replace(".", "_")


class _RouteOps(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = ast.unparse(node.func)
        if name.startswith("torch.ops."):
            node.func = ast.Name(id=op_binding(name.removeprefix("torch.ops.")), ctx=ast.Load())
        return node


def load_module(code: OutputCode, name: str) -> types.ModuleType:
    """Execute an output-code module's text, which compiles its Triton kernels."""
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(code.text, module.__file__, "exec"), module.__dict__)
    return module


@dataclass(frozen=True)
class CallResult:
    values: tuple[StoredValue, ...]
    outputs: tuple[Any, ...]
    environment: dict[str, Any]

    def last(self, variable: str) -> torch.Tensor:
        """The last stored value written through a variable name."""
        for value in reversed(self.values):
            if value.variable == variable:
                return value.tensor
        raise KeyError(variable)


def run_call(
    code: OutputCode,
    namespace: Mapping[str, Any],
    arguments: list[Any],
    *,
    ops: Mapping[str, Callable[..., Any]],
    before: BeforeStatement | None = None,
) -> CallResult:
    """Run ``call(arguments)`` one statement at a time.

    ``namespace`` is the loaded module's globals (kernels, ``extern_kernels``, helpers). ``ops`` maps a
    custom-op target such as ``vllm.moe_forward.default`` to the harness implementation.
    """
    environment: dict[str, Any] = dict(namespace)
    environment["args"] = list(arguments)
    for launch in code.launches:
        if launch.kind is LaunchKind.CUSTOM_OP:
            if launch.target not in ops:
                raise KeyError(f"no harness implementation for custom op {launch.target}")
            environment[op_binding(launch.target)] = ops[launch.target]
    launches = {launch.statement: launch for launch in code.launches}
    stored: list[StoredValue] = []
    outputs: tuple[Any, ...] = ()
    for index, statement in enumerate(code.statements):
        if before is not None:
            before(index, environment)
        if isinstance(statement, ast.Return):
            value = eval(compile(ast.Expression(body=statement.value), "<call-return>", "eval"), environment)
            outputs = value if isinstance(value, tuple) else (value,)
            break
        routed = _RouteOps().visit(ast.Module(body=[copy.deepcopy(statement)], type_ignores=[]))
        exec(compile(ast.fix_missing_locations(routed), f"<call-statement-{index}>", "exec"), environment)
        launch = launches.get(index)
        if launch is None:
            continue
        for variable in dict.fromkeys(launch.writes):
            tensor = environment[variable]
            stored.append(
                StoredValue(
                    launch=launch, variable=variable, tensor=tensor.detach().clone(), data_ptr=tensor.data_ptr()
                )
            )
    return CallResult(values=tuple(stored), outputs=outputs, environment=environment)
