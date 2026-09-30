"""Read the Inductor output-code modules that compiled vLLM ran.

An output-code module holds the Triton kernels of one piecewise subgraph and a ``call(args)`` function
that allocates buffers and launches, in order, fused Triton kernels, ``extern_kernels`` GEMMs and opaque
custom ops. This module reads the text only: it lists every launch with the variables it reads and
writes, follows buffer reuse and ``reinterpret_tensor`` views to the storage they share, keeps the
source-node comment Inductor writes above each launch, and, for each Triton kernel, which stored value
is computed from which loads. A store to a ``*bf16`` pointer is the only place a fused kernel rounds;
a store whose value also feeds another store in the same kernel hands that consumer the unrounded fp32
value.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum

TRITON_FACTORY = "async_compile.triton"
EXTERN_MODULE = "extern_kernels"
ALIAS_FUNCTIONS = frozenset({"reinterpret_tensor", "copy_if_misaligned", "alloc_from_pool"})
ALLOCATION_FUNCTIONS = frozenset({"empty_strided_cuda", "empty_strided_cpu", "empty_strided"})
SOURCE_NODES_PATTERN = re.compile(r"#\s*(?:Topologically Sorted|Unsorted) Source Nodes: \[([^\]]*)\]")
LINEAR_NODE_PATTERN = re.compile(r"^linear(?:_(\d+))?$")
SIGNATURE_PATTERN = re.compile(r"'signature': (\{[^}]*\})")
POINTER_PREFIXES = ("in_out_ptr", "out_ptr", "in_ptr")
# Inductor writes this line into every compiled-graph module; standalone kernel files lack it.
GRAPH_MODULE_MARKER = re.compile(r"^# AOT ID: ", re.MULTILINE)


class LaunchKind(StrEnum):
    TRITON = "triton"
    EXTERN = "extern"
    CUSTOM_OP = "custom_op"


@dataclass(frozen=True)
class StoreFlow:
    """One ``tl.store`` of a Triton kernel and the kernel parameters its value is computed from."""

    pointer: str
    value: str
    loads: tuple[str, ...]
    unrounded_from: tuple[str, ...]
    """Pointers of other stores in the same kernel whose values feed this one before their rounding."""


@dataclass(frozen=True)
class KernelSource:
    name: str
    heuristic: str
    params: tuple[str, ...]
    signature: dict[str, str]
    body: str
    stores: tuple[StoreFlow, ...]
    reductions: int

    def pointer_dtype(self, param: str) -> str | None:
        kind = self.signature.get(param)
        return kind[1:] if kind is not None and kind.startswith("*") else None


@dataclass(frozen=True)
class Launch:
    """One kernel, extern GEMM or custom-op launch in ``call()``."""

    statement: int
    kind: LaunchKind
    target: str
    source_nodes: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    text: str
    pointers: tuple[tuple[str, str], ...] = ()
    """For a Triton launch, each pointer parameter with the variable passed for it."""

    @property
    def linear_index(self) -> int | None:
        """The index ``n`` of the ``linear_n`` FX node an extern GEMM implements (``linear`` is 0)."""
        if self.kind is not LaunchKind.EXTERN:
            return None
        indices = [match for node in self.source_nodes if (match := LINEAR_NODE_PATTERN.match(node))]
        if not indices:
            return None
        last = indices[-1]
        return int(last.group(1)) if last.group(1) is not None else 0


@dataclass(frozen=True)
class OutputCode:
    text: str
    kernels: dict[str, KernelSource]
    arguments: tuple[str, ...]
    statements: tuple[ast.stmt, ...]
    launches: tuple[Launch, ...]
    returns: tuple[str, ...]
    storage: dict[str, str]
    """Variable name to the root variable (argument, allocation or op result) whose storage it uses."""

    def root(self, name: str) -> str:
        return self.storage.get(name, name)


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return ""


def _tensor_name(node: ast.expr) -> str | None:
    """The variable a launch argument refers to, looking through alias helpers."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and _call_name(node.func) in ALIAS_FUNCTIONS and node.args:
        return _tensor_name(node.args[0])
    return None


def _copied_value(expr: ast.expr) -> str | None:
    """For ``x.to(...)`` or ``tl.broadcast_to(x, ...)``, the name ``x``; ``"load"`` for a bare ``tl.load``."""
    if not isinstance(expr, ast.Call):
        return None
    called = _call_name(expr.func)
    if called == "tl.load":
        return "load"
    if isinstance(expr.func, ast.Attribute) and expr.func.attr == "to":
        inner = expr.func.value
        if isinstance(inner, ast.Name):
            return inner.id
        return _copied_value(inner)
    if called == "tl.broadcast_to" and expr.args and isinstance(expr.args[0], ast.Name):
        return expr.args[0].id
    return None


def _kernel_stores(function: ast.FunctionDef) -> tuple[tuple[StoreFlow, ...], int]:
    parents: dict[str, set[str]] = {}
    loads: dict[str, set[str]] = {}
    copies: dict[str, str | None] = {}
    stores: list[tuple[int, int, str, str]] = []
    reductions = 0

    def names(expr: ast.AST) -> set[str]:
        return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}

    def loaded_pointers(expr: ast.AST) -> set[str]:
        found = set()
        for node in ast.walk(expr):
            if isinstance(node, ast.Call) and _call_name(node.func) == "tl.load" and node.args:
                found |= {name for name in names(node.args[0]) if name.startswith(POINTER_PREFIXES)}
        return found

    def exact(value: str) -> bool:
        """Whether a value is a loaded input passed through casts and broadcasts without arithmetic."""
        seen = set()
        while value not in seen:
            seen.add(value)
            source = copies.get(value)
            if source is None:
                return False
            if source == "load":
                return True
            value = source
        return False

    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            parents[target] = names(node.value) - {target}
            loads[target] = loaded_pointers(node.value)
            copies[target] = _copied_value(node.value)
            reductions += sum(
                1
                for call in ast.walk(node.value)
                if isinstance(call, ast.Call) and _call_name(call.func) in {"tl.sum", "triton_helpers.max2"}
            )
        elif isinstance(node, ast.Call) and _call_name(node.func) == "tl.store" and len(node.args) >= 2:
            pointer = next(iter(sorted(n for n in names(node.args[0]) if n.startswith(POINTER_PREFIXES))), "")
            value = node.args[1]
            text = value.id if isinstance(value, ast.Name) else ast.unparse(value)
            stores.append((node.lineno, node.col_offset, pointer, text))
    stores.sort()

    def ancestry(value: str) -> tuple[set[str], set[str]]:
        seen: set[str] = set()
        pointers: set[str] = set()
        frontier = [value]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            pointers |= loads.get(current, set())
            frontier.extend(parents.get(current, ()))
        return seen, pointers

    flows = []
    for _, _, pointer, value in stores:
        seen, pointers = ancestry(value)
        unrounded = tuple(
            sorted(
                {
                    other
                    for _, _, other, other_value in stores
                    if other_value != value and other_value in seen and not exact(other_value)
                }
            )
        )
        flows.append(StoreFlow(pointer=pointer, value=value, loads=tuple(sorted(pointers)), unrounded_from=unrounded))
    return tuple(flows), reductions


def parse_kernel(name: str, source: str) -> KernelSource:
    """Parse one Triton kernel source string emitted by Inductor."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    heuristic = ""
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call):
            called = _call_name(decorator.func)
            if called.startswith("triton_heuristics."):
                heuristic = called.removeprefix("triton_heuristics.")
    signature_match = SIGNATURE_PATTERN.search(source)
    signature = ast.literal_eval(signature_match.group(1)) if signature_match else {}
    stores, reductions = _kernel_stores(function)
    return KernelSource(
        name=name,
        heuristic=heuristic,
        params=tuple(argument.arg for argument in function.args.args),
        signature=signature,
        body=ast.get_source_segment(source, function) or "",
        stores=stores,
        reductions=reductions,
    )


def _flatten(statements: list[ast.stmt]) -> list[ast.stmt]:
    flat: list[ast.stmt] = []
    for statement in statements:
        if isinstance(statement, ast.With):
            flat.extend(_flatten(statement.body))
        else:
            flat.append(statement)
    return flat


def _call_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            return node
        if isinstance(node, ast.ClassDef) and node.name == "Runner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "call":
                    return item
    raise ValueError("output code has no call() function")


def _source_node_comments(text: str) -> list[tuple[int, tuple[str, ...]]]:
    comments = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = SOURCE_NODES_PATTERN.search(line)
        if match and line.lstrip().startswith("#"):
            nodes = tuple(part.strip() for part in match.group(1).split(",") if part.strip())
            comments.append((number, nodes))
    return comments


def is_graph_module(text: str) -> bool:
    """Whether a cached ``.py`` file is a compiled subgraph (with ``call()``) rather than one kernel's source."""
    return GRAPH_MODULE_MARKER.search(text) is not None


def parse_output_code(text: str) -> OutputCode:
    """Parse one output-code module: its kernels, ``call()`` statements and launches."""
    tree = ast.parse(text)
    kernels: dict[str, KernelSource] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == TRITON_FACTORY
        ):
            name = node.value.args[0].value
            kernels[name] = parse_kernel(name, node.value.args[1].value)

    call = _call_function(tree)
    statements = _flatten(call.body)
    comments = _source_node_comments(text)
    arguments: tuple[str, ...] = ()
    storage: dict[str, str] = {}
    launches: list[Launch] = []
    returns: tuple[str, ...] = ()
    previous_line = call.lineno

    def root(name: str) -> str:
        return storage.get(name, name)

    for index, statement in enumerate(statements):
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name):
            source = statement.value.id
            for target in statement.targets:
                if isinstance(target, ast.Tuple) and source == "args":
                    arguments = tuple(element.id for element in target.elts if isinstance(element, ast.Name))
                    storage.update({argument: argument for argument in arguments})
                elif isinstance(target, ast.Name):
                    storage[target.id] = root(source)
            continue
        if isinstance(statement, ast.Return) and statement.value is not None:
            elements = statement.value.elts if isinstance(statement.value, ast.Tuple) else [statement.value]
            returns = tuple(name for element in elements if (name := _tensor_name(element)) is not None)
            continue
        call_node = statement.value if isinstance(statement, (ast.Expr, ast.Assign)) else None
        if not isinstance(call_node, ast.Call):
            continue
        called = _call_name(call_node.func)
        targets = (
            [t.id for t in statement.targets if isinstance(t, ast.Name)] if isinstance(statement, ast.Assign) else []
        )
        if called in ALLOCATION_FUNCTIONS:
            storage.update({target: target for target in targets})
            continue
        if called in ALIAS_FUNCTIONS:
            aliased = _tensor_name(call_node.args[0]) if call_node.args else None
            for target in targets:
                storage[target] = root(aliased) if aliased is not None else target
            continue

        kind: LaunchKind | None = None
        reads: list[str] = []
        writes: list[str] = []
        pointers: list[tuple[str, str]] = []
        target_name = called
        if called.endswith(".run") and called.removesuffix(".run") in kernels:
            kind = LaunchKind.TRITON
            kernel = kernels[called.removesuffix(".run")]
            target_name = kernel.name
            for param, argument in zip(kernel.params, call_node.args, strict=False):
                variable = _tensor_name(argument)
                if variable is None or not param.startswith(POINTER_PREFIXES):
                    continue
                pointers.append((param, variable))
                if param.startswith(("in_ptr", "in_out_ptr")):
                    reads.append(variable)
                if param.startswith(("out_ptr", "in_out_ptr")):
                    writes.append(variable)
        elif called.startswith(f"{EXTERN_MODULE}."):
            kind = LaunchKind.EXTERN
            target_name = called.removeprefix(f"{EXTERN_MODULE}.")
            reads = [name for argument in call_node.args if (name := _tensor_name(argument)) is not None]
            writes = [
                name
                for keyword in call_node.keywords
                if keyword.arg == "out" and (name := _tensor_name(keyword.value)) is not None
            ]
        elif called.startswith("torch.ops."):
            kind = LaunchKind.CUSTOM_OP
            target_name = called.removeprefix("torch.ops.")
            reads = [name for argument in call_node.args if (name := _tensor_name(argument)) is not None]
            writes = list(targets)
            storage.update({target: target for target in targets})
        if kind is None:
            continue
        nodes = tuple(nodes for number, nodes in comments if previous_line < number < statement.lineno)
        previous_line = statement.lineno
        launches.append(
            Launch(
                statement=index,
                kind=kind,
                target=target_name,
                source_nodes=nodes[-1] if nodes else (),
                reads=tuple(reads),
                writes=tuple(writes),
                text=ast.unparse(statement),
                pointers=tuple(pointers),
            )
        )
    return OutputCode(
        text=text,
        kernels=kernels,
        arguments=arguments,
        statements=tuple(statements),
        launches=tuple(launches),
        returns=returns,
        storage=storage,
    )
