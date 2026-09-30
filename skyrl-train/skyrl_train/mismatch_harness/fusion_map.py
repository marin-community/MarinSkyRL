"""List every launch of compiled vLLM's Grug subgraphs with what it reads and where it rounds to bf16.

Usage: ``python -m skyrl_train.mismatch_harness.fusion_map <output-code dir or files...>`` prints a
Markdown table per subgraph. Inside a fused Triton kernel every intermediate stays fp32 (Inductor drops
``.to(bfloat16)`` casts between fused ops); a value is rounded only where the kernel stores it to a
``*bf16`` pointer. A store marked "reads unrounded" feeds another store of the same kernel before its own
rounding, so that consumer sees the fp32 value.
"""

from __future__ import annotations

import sys
from pathlib import Path

from skyrl_train.mismatch_harness.output_code import LaunchKind, OutputCode, is_graph_module, parse_output_code
from skyrl_train.mismatch_harness.pieces import Piece, classify


def _describe(code: OutputCode, piece: Piece, name: str, statement: int, roles_at: dict) -> str:
    """A variable read at ``statement`` as its argument role or the stored value it holds."""
    for role, argument in {**piece.parameters, **piece.activations}.items():
        if argument == name:
            return role
    root = code.root(name)
    for launch in reversed(code.launches):
        if launch.statement >= statement:
            continue
        for written in launch.writes:
            if code.root(written) == root:
                return roles_at.get((launch.statement, written), f"{root}@{launch.statement}")
    return root


def render_piece(piece: Piece) -> list[str]:
    code = piece.code
    roles_at = {(ref.statement, ref.variable): role for role, ref in piece.values.items()}
    gemm_role = {launch.statement: role for role, launch in piece.gemms.items()}
    lines = [
        f"### {piece.kind} piece{' with RoPE' if piece.rope else ''}",
        "",
        "| # | launch | fuses (source nodes) | reads | stores |",
        "|---|---|---|---|---|",
    ]
    for number, launch in enumerate(code.launches):
        reads = ", ".join(
            _describe(code, piece, name, launch.statement, roles_at) for name in dict.fromkeys(launch.reads)
        )
        if launch.kind is LaunchKind.TRITON:
            kernel = code.kernels[launch.target]
            stores = []
            passed = dict(launch.pointers)
            for store in kernel.stores:
                variable = passed[store.pointer]
                role = roles_at.get((launch.statement, variable), code.root(variable))
                dtype = kernel.pointer_dtype(store.pointer)
                note = f"; reads unrounded {', '.join(store.unrounded_from)}" if store.unrounded_from else ""
                stores.append(f"{role} ({dtype}){note}")
            target = f"`{launch.target}` ({kernel.heuristic}, {kernel.reductions} reductions)"
        else:
            target = f"`{launch.target}` {gemm_role.get(launch.statement, '')}".strip()
            stores = [
                f"{roles_at.get((launch.statement, variable), code.root(variable))}" for variable in launch.writes
            ]
        nodes = ", ".join(launch.source_nodes)
        lines.append(f"| {number} | {target} | {nodes} | {reads} | {'; '.join(stores)} |")
    lines.append("")
    return lines


def render(paths: list[Path]) -> str:
    lines = ["# Compiled vLLM Grug subgraphs: launches and bf16 stores", ""]
    for path in paths:
        text = path.read_text()
        if not is_graph_module(text):
            continue
        piece = classify(parse_output_code(text))
        lines.append(f"Source: `{path.name}`")
        lines.append("")
        lines += render_piece(piece)
    return "\n".join(lines)


def main() -> None:
    paths: list[Path] = []
    for argument in sys.argv[1:]:
        path = Path(argument)
        paths += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    print(render(paths))


if __name__ == "__main__":
    main()
