"""Capture the Inductor kernels a decode-invariant vLLM engine compiles for Grug into ``inductor_kernels.py``.

Run on an H100 with this repository's vLLM wheel, from the repository root:

    python -m skyrl_train.models.grug_exact.capture_inductor_kernels --dump /tmp/grug-kernels \\
        --output skyrl-train/skyrl_train/models/grug_exact/inductor_kernels.py

The tool writes a Snowball-shaped Grug checkpoint with few narrow experts, serves it in this process with vLLM as
the decode-invariant engine (``decode_invariant.engine``'s kwargs), reads the Inductor output code vLLM's compile
wrote under a fresh cache root, writes every captured kernel, wrapper and the capture's provenance to ``--dump`` (and
prints the dump, compressed and base64-encoded, so a job log carries it), then selects the kernels the trainer runs
by role and writes them to ``--output``. ``--from-dump DIR`` renders ``--output`` from an earlier dump instead of
serving.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
import re
import shlex
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from skyrl_train.inference_engines.vllm.decode_invariant.engine import (
    ALL2ALL_BACKEND,
    DECODE_INVARIANT_ATTENTION_BACKEND,
    INDUCTOR_COMPILE_CONFIG,
    MOE_BACKEND,
    decode_invariant_engine_init_kwargs,
)
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM

# The Grug model whose kernels the trainer runs: Snowball's attention geometry, hidden width, shared-expert width,
# norm epsilon and rotary table length (the kernels hard-code them), with few narrow routed experts, which run as a
# custom op.
CAPTURE_MODEL = dict(
    vocab_size=128256,
    max_position_embeddings=65536,
    num_hidden_layers=4,
    num_local_experts=8,
    intermediate_size=128,
    initializer_range=0.02,
)
# The model-config attributes the compiled kernels hard-code; ``config/numerics.py`` checks a policy against them.
COMPILED_MODEL_ATTRIBUTES = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "shared_expert_intermediate_size",
    "qk_mult",
    "qk_mult_long_scale",
    "rms_norm_eps",
    "max_position_embeddings",
)
PROMPT_LENGTHS = (1, 7, 300, 1500)
RESPONSE_TOKENS = 4


class KernelRole(NamedTuple):
    """How to find one kernel the trainer runs: a pattern on its Inductor name (the fused operations) and the
    parameter names its signature must and must not have."""

    pattern: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


ROLES: dict[str, KernelRole] = {
    "EMBEDDING_NORM": KernelRole(r"^triton_red_fused__to_copy_embedding_rms_norm_\d+$"),
    "EMBEDDING_PRODUCT_NORM": KernelRole(r"^triton_red_fused_mul_rms_norm_sigmoid_\d+$"),
    "GATE_SILU": KernelRole(r"^triton_poi_fused_silu_\d+$"),
    "GATED_PRODUCT": KernelRole(r"^triton_poi_fused_mul_sigmoid_\d+$"),
    "RMS_NORM": KernelRole(r"^triton_red_fused_rms_norm_\d+$"),
    "RESIDUAL_NORM": KernelRole(r"^triton_red_fused_add_rms_norm_\d+$", required=("out_ptr1",)),
    "FINAL_NORM": KernelRole(r"^triton_red_fused_add_rms_norm_\d+$", forbidden=("out_ptr1",)),
    "SHARED_ACTIVATION": KernelRole(r"^triton_poi_fused_mul_silu_\d+$"),
    "XSA_HEAD_GATE": KernelRole(r"mul_sigmoid_sub_sum"),
    "HEAD_SQUARE_SUMS": KernelRole(r"^triton_per_fused__to_copy_mean_pow_view_\d+$"),
    "KEY_ROPE": KernelRole(r"index_select_mean_mul_pow_rsqrt_slice_split_sub", required=("out_ptr1",)),
    "QUERY_ROPE": KernelRole(r"index_select_mean_mul_pow_rsqrt_slice_split_sub", forbidden=("out_ptr1",)),
    "QUERY_SCALE": KernelRole(r"^triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_slice_view_\d+$"),
    "KEY_NORM": KernelRole(r"^triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_view_\d+$", required=("in_out_ptr0",)),
    "QUERY_NORM_SCALE": KernelRole(r"^triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_view_\d+$", required=("ks0",)),
}
KERNEL_PATTERN = re.compile(r"(\w+) = async_compile\.triton\('(\w+)', '''(.*?)''', device_str='cuda'\)", re.DOTALL)
SIGNATURE_PATTERN = re.compile(r"\ndef (\w+)\((.*?)\):")
DUMP_MARKER = "===== dump.tar.gz.b64"
DUMP_END_MARKER = "===== end of dump"
DUMP_LINE = 120


def kernel_parameters(source: str) -> tuple[str, ...]:
    """The parameter names of the Triton kernel defined in ``source``; empty when it defines none."""
    match = SIGNATURE_PATTERN.search(source)
    return tuple(part.split(":")[0].strip() for part in match.group(2).split(",")) if match else ()


@dataclass(frozen=True)
class Provenance:
    """Where a capture's kernels came from: the capture date and command, the engine's library versions and GPU,
    the model config values the kernels hard-code, and the engine settings."""

    date: str
    command: str
    versions: dict[str, str]
    compiled_model: dict[str, object]
    inductor_compile_config: dict[str, object]
    engine_init_kwargs: dict[str, object]


@dataclass(frozen=True)
class CapturedKernel:
    name: str
    source: str
    launches: tuple[str, ...]

    @property
    def body(self) -> str:
        """The kernel's code after ``@triton.jit`` with its name's Inductor index removed, which two graph pieces
        compiling the same operations share."""
        return re.sub(r"_\d+\(", "(", self.source[self.source.index("@triton.jit") :], count=1)

    @property
    def parameters(self) -> tuple[str, ...]:
        return kernel_parameters(self.source)

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(self.parameters)})"


def write_checkpoint(path: Path) -> GrugMoeConfig:
    config = GrugMoeConfig(**CAPTURE_MODEL)
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            # A decode-invariant engine refuses router weights bf16 cannot represent.
            layer.mlp.router.weight.copy_(layer.mlp.router.weight.to(torch.bfloat16))
    model.save_pretrained(path, safe_serialization=True)
    return config


def serve(checkpoint: Path, max_model_len: int) -> dict[str, str]:
    """Serve ``checkpoint`` as a decode-invariant engine and return the versions it ran with."""
    import vllm  # noqa: PLC0415 - imported after the cache root is set in the environment
    import vllm.vllm_flash_attn  # noqa: PLC0415,F401 - loads the FA3 op the attention backend calls

    engine = vllm.LLM(
        model=str(checkpoint),
        dtype="bfloat16",
        seed=0,
        skip_tokenizer_init=True,
        enforce_eager=False,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=len(PROMPT_LENGTHS),
        gpu_memory_utilization=0.5,
        attention_backend=DECODE_INVARIANT_ATTENTION_BACKEND,
        kernel_config={"moe_backend": MOE_BACKEND},
        all2all_backend=ALL2ALL_BACKEND,
        **decode_invariant_engine_init_kwargs(),
    )
    generator = torch.Generator().manual_seed(5)
    prompts = [
        vllm.TokensPrompt(prompt_token_ids=torch.randint(10, 1000, (length,), generator=generator).tolist())
        for length in PROMPT_LENGTHS
    ]
    engine.generate(prompts, vllm.SamplingParams(max_tokens=RESPONSE_TOKENS, temperature=0.0, detokenize=False))
    import triton  # noqa: PLC0415 - a GPU-only extra

    return {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(),
    }


def captured_kernels(cache_root: Path) -> tuple[list[CapturedKernel], list[str]]:
    """Every Triton kernel in the Inductor output code under ``cache_root`` with a body no earlier kernel has, with
    the lines of the output code that launch it, and the output code (the wrappers) itself."""
    wrappers = sorted(
        path for path in (cache_root / "torch_compile_cache").rglob("*.py") if "inductor_cache" in path.parts
    )
    kernels: list[CapturedKernel] = []
    texts = []
    for path in wrappers:
        text = path.read_text()
        found = KERNEL_PATTERN.findall(text)
        if not found:
            continue
        texts.append(f"# {path}\n{text}")
        for variable, name, source in found:
            if variable != name:
                raise ValueError(f"{path}: kernel {name} is bound to {variable}")
            launches = tuple(line.strip() for line in text.splitlines() if f"{name}.run(" in line)
            kernels.append(CapturedKernel(name, source, launches))
    return distinct(kernels), texts


def distinct(kernels: list[CapturedKernel]) -> list[CapturedKernel]:
    """The kernels whose body no earlier kernel has."""
    kept: list[CapturedKernel] = []
    for kernel in kernels:
        if all(kernel.body != known.body for known in kept):
            kept.append(kernel)
    return kept


def write_dump(dump: Path, kernels: list[CapturedKernel], wrappers: list[str], provenance: Provenance) -> None:
    (dump / "kernels").mkdir(parents=True, exist_ok=True)
    (dump / "wrappers").mkdir(parents=True, exist_ok=True)
    for index, kernel in enumerate(kernels):
        (dump / "kernels" / f"{index:02d}_{kernel.name}.py").write_text(kernel.source)
        (dump / "kernels" / f"{index:02d}_{kernel.name}.launches").write_text("\n".join(kernel.launches))
    for index, wrapper in enumerate(wrappers):
        (dump / "wrappers" / f"{index:02d}.py").write_text(wrapper)
    (dump / "provenance.json").write_text(json.dumps(asdict(provenance), indent=2))


def read_dump(dump: Path) -> tuple[list[CapturedKernel], Provenance]:
    kernels = []
    for path in sorted((dump / "kernels").glob("*.py")):
        launches = tuple(path.with_suffix(".launches").read_text().splitlines())
        kernels.append(CapturedKernel(path.stem.split("_", 1)[1], path.read_text(), launches))
    return distinct(kernels), Provenance(**json.loads((dump / "provenance.json").read_text()))


def print_dump(dump: Path) -> None:
    """Print the dump as a compressed, base64-encoded tar in short lines, between markers."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(dump, arcname=".")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    print(DUMP_MARKER)
    for start in range(0, len(encoded), DUMP_LINE):
        print(encoded[start : start + DUMP_LINE])
    print(DUMP_END_MARKER)


def select_roles(kernels: list[CapturedKernel]) -> dict[str, CapturedKernel]:
    """Each role's kernel; raise when a role matches no kernel or several."""
    roles = {}
    problems = []
    for role, spec in ROLES.items():
        matches = [
            kernel
            for kernel in kernels
            if re.search(spec.pattern, kernel.name)
            and all(name in kernel.parameters for name in spec.required)
            and not any(name in kernel.parameters for name in spec.forbidden)
        ]
        if len(matches) != 1:
            problems.append(f"{role}: {[kernel.signature for kernel in matches]}")
            continue
        roles[role] = matches[0]
    if problems:
        raise ValueError("roles without exactly one kernel: " + "; ".join(problems))
    return roles


def _literal(value: object) -> str:
    """``value`` as Python source in the repository's formatting (double-quoted strings)."""
    return json.dumps(value).replace("true", "True").replace("false", "False").replace("null", "None")


def render(roles: dict[str, CapturedKernel], provenance: Provenance) -> str:
    compiled = provenance.compiled_model
    versions = provenance.versions
    lines = [
        '"""Compiled vLLM\'s Inductor kernels for Grug, generated by ``capture_inductor_kernels``; do not edit.',
        "",
        f"Generated on {provenance.date} by:",
        f"    {provenance.command}",
        "from the Inductor output code of a decode-invariant vLLM engine serving a Grug model with",
        f"    {compiled}",
        f"on {versions['gpu']} (compute capability {versions['compute_capability']}) with vLLM {versions['vllm']}, "
        f"torch {versions['torch']}, Triton {versions['triton']}, CUDA {versions['cuda']}, and the Inductor settings",
        f"    {provenance.inductor_compile_config}.",
        "Each kernel keeps its Inductor name and source; the comment above it shows one of the engine's launches.",
        '"""',
        "",
        "COMPILED_MODEL = {",
        *(f"    {_literal(name)}: {_literal(value)}," for name, value in compiled.items()),
        "}",
        f"INDUCTOR_COMPILE_CONFIG = {_literal(provenance.inductor_compile_config)}",
        "",
    ]
    for role, kernel in roles.items():
        if '"""' in kernel.source:
            raise ValueError(f"{kernel.name} contains a triple quote")
        for launch in kernel.launches[:1]:
            lines.append(f"# {launch}")
        lines.append(f"{role} = (")
        lines.append(f"    {_literal(kernel.name)},")
        lines.append(f'    r"""{kernel.source}""",')
        lines.append(")")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("inductor_kernels.py"))
    parser.add_argument("--dump", type=Path, default=None, help="write every captured kernel and wrapper here")
    parser.add_argument("--from-dump", type=Path, default=None, help="render --output from this earlier dump")
    parser.add_argument("--max-model-len", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.from_dump is not None:
        kernels, provenance = read_dump(arguments.from_dump)
    else:
        command = " ".join(shlex.quote(part) for part in ["python", "-m", __spec__.name, *sys.argv[1:]])
        with tempfile.TemporaryDirectory(prefix="grug-exact-capture-") as scratch:
            scratch = Path(scratch)
            cache_root = scratch / "vllm-cache"
            os.environ["VLLM_CACHE_ROOT"] = str(cache_root)
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            config = write_checkpoint(scratch / "model")
            versions = serve(scratch / "model", arguments.max_model_len)
            kernels, wrappers = captured_kernels(cache_root)
        provenance = Provenance(
            date=datetime.date.today().isoformat(),
            command=command,
            versions=versions,
            compiled_model={name: getattr(config, name) for name in COMPILED_MODEL_ATTRIBUTES},
            inductor_compile_config=dict(INDUCTOR_COMPILE_CONFIG),
            engine_init_kwargs=decode_invariant_engine_init_kwargs(),
        )
        print(f"captured {len(kernels)} kernels:")
        for kernel in kernels:
            print(f"  {kernel.signature}")
            for launch in kernel.launches:
                print(f"      {launch}")
        if arguments.dump is not None:
            write_dump(arguments.dump, kernels, wrappers, provenance)
            print_dump(arguments.dump)
    roles = select_roles(kernels)
    arguments.output.write_text(render(roles, provenance))
    print(f"wrote {len(roles)} kernels to {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
