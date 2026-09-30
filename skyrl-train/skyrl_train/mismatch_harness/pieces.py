"""Name the arguments and stored values of one compiled vLLM Grug subgraph by their layer role.

vLLM splits the compiled Grug forward at every attention op, so one piecewise subgraph holds the part
of layer ``L-1`` after attention and the part of layer ``L`` before it (``MIDDLE``); the first holds the
embedding and layer 0 before attention (``FIRST``) and the last holds the last layer after attention
and the final norm (``LAST``). Dynamo numbers each ``F.linear`` node of a subgraph in the order the
model code calls it, so the ``linear_n`` source node of each extern GEMM names its weight. Every other
role follows from dataflow: the value a GEMM reads is whatever the last launch before it wrote into
that storage, and norm weights and activation inputs are the arguments the matching launches read.

Parameter roles use the Hugging Face names of ``GrugMoeForCausalLM`` relative to a layer: ``prev.`` is
the layer whose post-attention half the piece holds, ``next.`` the layer whose pre-attention half it
holds. Storage is reused inside ``call()``, so a stored value is named by the launch that wrote it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum

from skyrl_train.mismatch_harness.output_code import Launch, LaunchKind, OutputCode


class PieceKind(StrEnum):
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


PRE_ATTENTION_GEMMS = (
    "next.attn_gated_norm.down_proj",
    "next.attn_gated_norm.up_proj",
    "next.self_attn.q_proj",
    "next.self_attn.k_proj",
    "next.self_attn.v_proj",
)
POST_ATTENTION_GEMMS = (
    "prev.self_attn.attn_gate",
    "prev.self_attn.o_proj",
    "prev.mlp_gated_norm.down_proj",
    "prev.mlp_gated_norm.up_proj",
    "prev.mlp.router",
    "prev.shared_expert.gate_proj",
    "prev.shared_expert.up_proj",
    "prev.shared_expert.down_proj",
)
GEMM_ROLES = {
    PieceKind.FIRST: ("embed_gated_norm.down_proj", "embed_gated_norm.up_proj", *PRE_ATTENTION_GEMMS),
    PieceKind.MIDDLE: (*POST_ATTENTION_GEMMS, *PRE_ATTENTION_GEMMS),
    PieceKind.LAST: (*POST_ATTENTION_GEMMS, "final_gated_norm.down_proj", "final_gated_norm.up_proj"),
}
MOE_OP = "vllm.moe_forward.default"


@dataclass(frozen=True)
class ExampleArgument:
    """An argument as the module's ``get_args()`` builds it for benchmarking (sizes at the size hint)."""

    shape: tuple[int, ...] | None
    dtype: str | None


@dataclass(frozen=True)
class ValueRef:
    """The value a launch stored through ``variable`` (read from the replay's stored values)."""

    statement: int
    variable: str


@dataclass(frozen=True)
class Piece:
    code: OutputCode
    kind: PieceKind
    rope: bool
    gemms: dict[str, Launch]
    parameters: dict[str, str]
    """Parameter role (``prev.self_attn.o_proj.weight``) to the ``call()`` argument that carries it."""
    activations: dict[str, str]
    """Activation role (``prev.attn_out``, ``next.positions``) to the ``call()`` argument."""
    values: dict[str, ValueRef]
    """Stored-value role (``prev.mlp_in``, ``next.residual``) to the launch that stores it."""
    examples: dict[str, ExampleArgument]

    @property
    def moe(self) -> Launch | None:
        return next((launch for launch in self.code.launches if launch.target == MOE_OP), None)


def example_arguments(text: str) -> dict[str, ExampleArgument]:
    """Shapes and dtypes of ``call()`` arguments from the module's ``get_args()``."""
    tree = ast.parse(text)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args"), None)
    examples: dict[str, ExampleArgument] = {}
    if function is None:
        return examples
    for statement in function.body:
        if not (isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)):
            continue
        name = statement.targets[0].id
        value = statement.value
        if isinstance(value, ast.Call) and ast.unparse(value.func) == "rand_strided":
            dtype = next(
                (
                    ast.unparse(keyword.value).removeprefix("torch.")
                    for keyword in value.keywords
                    if keyword.arg == "dtype"
                ),
                None,
            )
            examples[name] = ExampleArgument(shape=tuple(ast.literal_eval(value.args[0])), dtype=dtype)
        else:
            examples[name] = ExampleArgument(shape=None, dtype=None)
    return examples


def _kind(code: OutputCode) -> PieceKind:
    if any("embedding" in launch.source_nodes for launch in code.launches):
        return PieceKind.FIRST
    if len(code.returns) == 1:
        return PieceKind.LAST
    return PieceKind.MIDDLE


class _Dataflow:
    def __init__(self, code: OutputCode):
        self.code = code

    def writer(self, variable: str, before: int) -> ValueRef:
        """The launch that last wrote ``variable``'s storage before statement ``before``."""
        root = self.code.root(variable)
        for launch in reversed(self.code.launches):
            if launch.statement >= before:
                continue
            for written in launch.writes:
                if self.code.root(written) == root:
                    return ValueRef(launch.statement, written)
        raise KeyError(f"no launch writes {variable} before statement {before}")

    def launch(self, ref: ValueRef) -> Launch:
        return next(launch for launch in self.code.launches if launch.statement == ref.statement)

    def gemm_input(self, gemm: Launch) -> str:
        return gemm.reads[1] if gemm.target == "addmm" else gemm.reads[0]

    def pointer_reads(self, launch: Launch, prefix: str) -> list[str]:
        """Variables a Triton launch passes for pointer parameters starting with ``prefix``."""
        kernel = self.code.kernels[launch.target]
        read_params = [param for param in kernel.params if param.startswith(("in_ptr", "in_out_ptr"))]
        return [name for name, param in zip(launch.reads, read_params, strict=True) if param.startswith(prefix)]


def classify(code: OutputCode) -> Piece:
    """Assign layer roles to one parsed output-code module."""
    kind = _kind(code)
    examples = example_arguments(code.text)
    arguments = set(code.arguments)
    flow = _Dataflow(code)
    roles = GEMM_ROLES[kind]
    gemms: dict[str, Launch] = {}
    for launch in code.launches:
        if launch.kind is not LaunchKind.EXTERN:
            continue
        index = launch.linear_index
        if index is None or index >= len(roles):
            raise ValueError(f"extern launch {launch.text!r} has no linear source node within the {kind} roles")
        gemms[roles[index]] = launch
    missing = set(roles) - set(gemms)
    if missing:
        raise ValueError(f"{kind} piece is missing GEMMs {sorted(missing)}")

    def vector_arguments(launch: Launch) -> list[str]:
        return [
            name
            for name in launch.reads
            if name in arguments and examples[name].shape is not None and len(examples[name].shape) == 1
        ]

    def weight_argument(gemm: Launch) -> str:
        """The argument holding a GEMM's weight, through a kernel that copies it into a padded buffer."""
        weight = gemm.reads[-1]
        if weight in arguments:
            return weight
        copy = flow.launch(flow.writer(weight, gemm.statement))
        sources = [name for name in copy.reads if name in arguments]
        if copy.kind is not LaunchKind.TRITON or len(sources) != 1:
            raise ValueError(f"cannot trace the weight of {gemm.text!r} to one argument")
        return sources[0]

    parameters = {f"{role}.weight": weight_argument(gemm) for role, gemm in gemms.items()}
    activations: dict[str, str] = {}
    values: dict[str, ValueRef] = {}

    def gemm_values(prefix: str, down: str, up: str) -> ValueRef:
        """Record a gated norm's GEMM chain; return the norm output it gates."""
        down_gemm, up_gemm = gemms[down], gemms[up]
        norm = flow.writer(flow.gemm_input(down_gemm), down_gemm.statement)
        values[f"{prefix}_rms"] = norm
        values[f"{prefix}_gate_down"] = ValueRef(down_gemm.statement, down_gemm.writes[0])
        values[f"{prefix}_gate_act"] = flow.writer(flow.gemm_input(up_gemm), up_gemm.statement)
        values[f"{prefix}_gate_up"] = ValueRef(up_gemm.statement, up_gemm.writes[0])
        return norm

    if kind is PieceKind.FIRST:
        embedding = next(launch for launch in code.launches if "embedding" in launch.source_nodes)
        activations["token_ids"] = next(name for name in embedding.reads if examples[name].dtype == "int32")
        parameters["embed_tokens.weight"] = next(
            name
            for name in embedding.reads
            if name in arguments and examples[name].shape is not None and len(examples[name].shape) == 2
        )
        parameters["embed_norm.weight"] = next(
            name for name in vector_arguments(embedding) if examples[name].dtype != "int32"
        )
        gemm_values("embed", "embed_gated_norm.down_proj", "embed_gated_norm.up_proj")
    else:
        attn_gate = gemms["prev.self_attn.attn_gate"]
        o_proj = gemms["prev.self_attn.o_proj"]
        activations["prev.attn_in"] = attn_gate.reads[0]
        activations["prev.residual"] = o_proj.reads[0]
        values["prev.attn_gate"] = ValueRef(attn_gate.statement, attn_gate.writes[0])
        xsa_ref = flow.writer(o_proj.reads[1], o_proj.statement)
        values["prev.xsa_gate"] = xsa_ref
        for name in flow.launch(xsa_ref).reads:
            if name in arguments:
                activations["prev.attn_out" if len(examples[name].shape) == 3 else "prev.value"] = name
        values["prev.residual_after_attention"] = ValueRef(o_proj.statement, o_proj.writes[0])
        post_norm = gemm_values("prev.mlp", "prev.mlp_gated_norm.down_proj", "prev.mlp_gated_norm.up_proj")
        parameters["prev.post_attention_layernorm.weight"] = vector_arguments(flow.launch(post_norm))[0]
        moe = next((launch for launch in code.launches if launch.target == MOE_OP), None)
        if moe is None:
            raise ValueError("post-attention piece has no MoE op")
        router = gemms["prev.mlp.router"]
        activations["prev.layer_name"] = next(name for name in moe.reads if name in arguments)
        values["prev.mlp_in"] = flow.writer(moe.reads[0], moe.statement)
        values["prev.router_input"] = flow.writer(flow.gemm_input(router), router.statement)
        values["prev.router_logits"] = ValueRef(router.statement, router.writes[0])
        values["prev.routed"] = ValueRef(moe.statement, moe.writes[0])
        for role in ("gate_proj", "up_proj", "down_proj"):
            gemm = gemms[f"prev.shared_expert.{role}"]
            values[f"prev.shared_{role}"] = ValueRef(gemm.statement, gemm.writes[0])
        down = gemms["prev.shared_expert.down_proj"]
        values["prev.shared_act"] = flow.writer(flow.gemm_input(down), down.statement)

    if kind is PieceKind.LAST:
        final_norm = gemm_values("final", "final_gated_norm.down_proj", "final_gated_norm.up_proj")
        parameters["final_norm.weight"] = vector_arguments(flow.launch(final_norm))[0]
        values["final_hidden"] = flow.writer(code.returns[0], len(code.statements))
        return Piece(code, kind, False, gemms, parameters, activations, values, examples)

    input_norm = gemm_values("next.attn", "next.attn_gated_norm.down_proj", "next.attn_gated_norm.up_proj")
    norm_launch = flow.launch(input_norm)
    parameters["next.input_layernorm.weight"] = vector_arguments(norm_launch)[0]
    residual = flow.pointer_reads(norm_launch, "in_out_ptr")
    if not residual:
        raise ValueError("the layer-input norm kernel does not store the residual stream")
    values["next.residual"] = ValueRef(norm_launch.statement, residual[0])
    q_proj = gemms["next.self_attn.q_proj"]
    values["next.attn_in"] = flow.writer(flow.gemm_input(q_proj), q_proj.statement)
    for role in ("q_proj", "k_proj", "v_proj"):
        gemm = gemms[f"next.self_attn.{role}"]
        values[f"next.{role}"] = ValueRef(gemm.statement, gemm.writes[0])
    positions = [
        name
        for name, example in examples.items()
        if name in arguments and example.dtype == "int64" and example.shape is not None and len(example.shape) == 1
    ]
    rope = bool(positions)
    if rope:
        activations["next.positions"] = positions[0]
        rotary = next(launch for launch in code.launches if positions[0] in launch.reads)
        activations["next.cos_sin_cache"] = next(
            name
            for name in rotary.reads
            if name in arguments and name != positions[0] and examples[name].dtype == "bfloat16"
        )
    return Piece(code, kind, rope, gemms, parameters, activations, values, examples)
