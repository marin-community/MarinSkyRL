#!/usr/bin/env python
"""De-fuse a MarinSkyRL/TorchTitan grouped-MoE checkpoint into HF Qwen3-MoE layout.

WHY THIS EXISTS
---------------
The RL disk-export path (``FSDPStrategy.save_hf_model`` in
``skyrl_train/distributed/fsdp_strategy.py``) collects the FSDP2 full state dict of
the *grouped-swapped* model and hands it straight to ``model.save_pretrained(...)``
WITHOUT running the grouped->HF weight remap that the *weight-sync-to-vLLM* path
(``FSDPWeightExtractor`` in ``workers/fsdp/fsdp_worker.py``) applies. So every MoE
RL checkpoint saved this way lands on disk with the FUSED grouped-expert layout:

    model.layers.{i}.mlp.moe.experts.w1   # (num_experts, moe_dim, dim)  == gate_proj
    model.layers.{i}.mlp.moe.experts.w3   # (num_experts, moe_dim, dim)  == up_proj
    model.layers.{i}.mlp.moe.experts.w2   # (num_experts, dim, moe_dim)  == down_proj
    model.layers.{i}.mlp.moe.router.gate.weight

but ``config.json`` says ``model_type: qwen3_moe``, so vLLM's HF loader
(``vllm/model_executor/models/qwen3_moe.py`` load_weights) expects the PER-EXPERT
layout and KeyErrors on ``experts.w1``. This script rewrites such a checkpoint into
the per-expert HF layout vLLM can load:

    model.layers.{i}.mlp.experts.{j}.gate_proj.weight   <- w1[j]   # [moe_dim, dim]
    model.layers.{i}.mlp.experts.{j}.up_proj.weight     <- w3[j]   # [moe_dim, dim]
    model.layers.{i}.mlp.experts.{j}.down_proj.weight   <- w2[j]   # [dim, moe_dim]
    model.layers.{i}.mlp.gate.weight                    <- router.gate.weight

VERIFIED MAPPING (w1=gate, w3=up, w2=down; NO per-expert transpose)
-------------------------------------------------------------------
Source of truth: ``skyrl_train/models/layers/moe.py``.

  * ``GroupedExperts`` docstring (the parameter contract):
        w1: (num_experts, hidden_dim, dim)  -- gate_proj
        w3: (num_experts, hidden_dim, dim)  -- up_proj
        w2: (num_experts, dim, hidden_dim)  -- down_proj
  * ``_run_experts_for_loop`` (the EP=1 PARITY oracle, docstring says it
    "numerically matches HF eager down(silu(gate(x)) * up(x))"):
        h = silu(x @ w1[j].T);  h = h * (x @ w3[j].T);  out = h @ w2[j].T
    i.e. ``x @ w1[j].T`` IS ``nn.Linear(gate_proj)(x)``, so w1[j] already has the
    HF ``[out_features, in_features]`` orientation of ``gate_proj.weight`` -- a
    plain per-expert SLICE, NO transpose. Same for w3->up_proj, w2->down_proj.
  * ``moe_weight_remap.convert_tt_layer_to_hf`` already encodes exactly this
    (w1[j]->gate_proj, w3[j]->up_proj, w2[j]->down_proj, router->mlp.gate). We
    REUSE that function so this script stays in lockstep with the trainer.

The only extra step vs the in-trainer converter is stripping the ``GroupedMoEShim``
``.mlp.moe.`` -> ``.mlp.`` prefix (matches ``FSDPWeightExtractor._strip_grouped_prefix``),
because the on-disk keys carry the live shim path while ``convert_tt_layer_to_hf``
matches on the post-strip ``.mlp.experts.*`` / ``.mlp.router.gate.weight`` names.

USAGE
-----
    python convert_fused_moe_to_hf.py \
        --src <fused_ckpt_dir_or_hf_repo_id> \
        --dst <output_dir> \
        [--dtype preserve]

    # self-test (no I/O; synthetic tensors prove the de-fuse is numerically exact)
    python convert_fused_moe_to_hf.py --self-test

Cluster follow-up (NOT done here): a full vLLM load-test of the emitted dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Reuse the trainer's verified grouped->HF converter WITHOUT importing the      #
# skyrl_train package (its models.layers.moe imports torchtitan, unavailable on #
# the Mac). moe_weight_remap.py itself imports only ``torch``, so we load that   #
# single module by file path.                                                   #
# --------------------------------------------------------------------------- #
_REMAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "skyrl_train"
    / "models"
    / "layers"
    / "moe_weight_remap.py"
)


def _load_remap_module():
    spec = importlib.util.spec_from_file_location("_skyrl_moe_weight_remap", _REMAP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_SHIM_SEG = ".mlp.moe."
_FSDP_SEG = "._fsdp_wrapped_module."


def _strip_grouped_prefix(name: str) -> str:
    """Normalize a live grouped-swapped key to the converter's expected form.

    Mirrors ``FSDPWeightExtractor._strip_grouped_prefix``:
      ``...layers.{i}.mlp.moe.experts.w1``        -> ``...layers.{i}.mlp.experts.w1``
      ``...layers.{i}.mlp.moe.router.gate.weight`` -> ``...mlp.router.gate.weight``
    Also drops any FSDP ``_fsdp_wrapped_module`` segments defensively.
    """
    name = name.replace(_FSDP_SEG, ".")
    name = name.replace(_SHIM_SEG, ".mlp.")
    return name


def defuse_state_dict(state_dict: dict, remap_mod) -> dict:
    """Return a NEW state dict in per-expert HF layout.

    1. Strip the shim/FSDP prefix on every key.
    2. Run the trainer's ``convert_tt_to_hf_moe`` (splits w1/w2/w3 -> per-expert
       gate_proj/up_proj/down_proj, renames router -> mlp.gate). In-place on a copy.
    """
    stripped = {}
    for k, v in state_dict.items():
        stripped[_strip_grouped_prefix(k)] = v
    remap_mod.convert_tt_to_hf_moe(stripped)  # in-place
    return stripped


# --------------------------------------------------------------------------- #
# Self-test: prove the de-fuse + mapping is numerically exact on tiny tensors.  #
# --------------------------------------------------------------------------- #


def self_test() -> None:
    remap_mod = _load_remap_module()
    torch.manual_seed(0)

    num_experts, dim, moe_dim = 4, 8, 6
    n_layers = 2

    # Build a KNOWN per-expert HF ground truth, then FUSE it the way the trainer's
    # GroupedExperts holds it (w1<-gate, w3<-up, w2<-down; per-expert stack on dim 0),
    # emit it under the on-disk shim names, run our de-fuse, and assert we recover
    # the exact per-expert HF tensors bit-for-bit.
    gt = {}  # ground-truth per-expert HF tensors
    fused = {}  # what the fused on-disk checkpoint holds
    for i in range(n_layers):
        w1 = torch.randn(num_experts, moe_dim, dim)  # gate_proj stack [E, moe, dim]
        w3 = torch.randn(num_experts, moe_dim, dim)  # up_proj stack
        w2 = torch.randn(num_experts, dim, moe_dim)  # down_proj stack [E, dim, moe]
        gate = torch.randn(num_experts, dim)  # router
        # On-disk fused (shim) names:
        fused[f"model.layers.{i}.mlp.moe.experts.w1"] = w1.clone()
        fused[f"model.layers.{i}.mlp.moe.experts.w3"] = w3.clone()
        fused[f"model.layers.{i}.mlp.moe.experts.w2"] = w2.clone()
        fused[f"model.layers.{i}.mlp.moe.router.gate.weight"] = gate.clone()
        # Ground-truth per-expert HF names (gate_proj<-w1[j], up_proj<-w3[j], down_proj<-w2[j]):
        for j in range(num_experts):
            gt[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
            gt[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = w3[j]
            gt[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = w2[j]
        gt[f"model.layers.{i}.mlp.gate.weight"] = gate
    # a couple of passthrough (non-MoE) keys must survive untouched
    fused["model.embed_tokens.weight"] = torch.randn(10, dim)
    gt["model.embed_tokens.weight"] = fused["model.embed_tokens.weight"]
    fused["model.layers.0.self_attn.q_proj.weight"] = torch.randn(dim, dim)
    gt["model.layers.0.self_attn.q_proj.weight"] = fused["model.layers.0.self_attn.q_proj.weight"]

    out = defuse_state_dict(fused, remap_mod)

    # (1) key sets identical
    assert set(out.keys()) == set(gt.keys()), (
        f"key mismatch\n  extra: {set(out) - set(gt)}\n  missing: {set(gt) - set(out)}"
    )
    # (2) every tensor bit-for-bit equal + shapes as HF expects
    for k, v in gt.items():
        assert out[k].shape == v.shape, f"shape mismatch {k}: {out[k].shape} != {v.shape}"
        assert torch.equal(out[k], v), f"value mismatch at {k}"

    # (3) numerical SwiGLU parity: fused-forward (x@w1.T; *x@w3.T; @w2.T) MUST equal
    #     HF per-expert forward down_proj(silu(gate_proj(x)) * up_proj(x)).
    import torch.nn.functional as F

    x = torch.randn(3, dim)
    e = 0
    w1 = fused["model.layers.0.mlp.moe.experts.w1"][e]
    w3 = fused["model.layers.0.mlp.moe.experts.w3"][e]
    w2 = fused["model.layers.0.mlp.moe.experts.w2"][e]
    fused_fwd = (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    gp = out["model.layers.0.mlp.experts.0.gate_proj.weight"]
    up = out["model.layers.0.mlp.experts.0.up_proj.weight"]
    dn = out["model.layers.0.mlp.experts.0.down_proj.weight"]
    hf_fwd = F.linear(F.silu(F.linear(x, gp)) * F.linear(x, up), dn)
    assert torch.allclose(fused_fwd, hf_fwd, atol=1e-6), "SwiGLU forward parity failed"

    print("SELF-TEST PASS:")
    print(f"  layers={n_layers} experts={num_experts} dim={dim} moe_dim={moe_dim}")
    print(f"  de-fused {len(gt)} keys, all bit-exact vs per-expert HF ground truth")
    print("  SwiGLU forward parity (fused == HF per-expert): OK")
    print("  shapes: gate_proj/up_proj [moe_dim, dim], down_proj [dim, moe_dim] (HF [out,in])")


# --------------------------------------------------------------------------- #
# Full checkpoint conversion (safetensors sharded -> safetensors sharded)       #
# --------------------------------------------------------------------------- #

_PASSTHROUGH_COPY = ("tokenizer", "generation_config", "special_tokens", "vocab", "merges", "added_tokens", "chat_template")


def _resolve_src(src: str, cache_dir: str | None) -> Path:
    """Return a local dir for ``src`` (HF repo id -> snapshot download; dir -> as-is)."""
    p = Path(src)
    if p.exists() and p.is_dir():
        return p
    # treat as HF repo id
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    local = snapshot_download(
        repo_id=src,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"],
    )
    return Path(local)


def convert_checkpoint(src: str, dst: str, dtype: str = "preserve", cache_dir: str | None = None) -> None:
    from safetensors.torch import load_file, save_file

    remap_mod = _load_remap_module()
    src_dir = _resolve_src(src, cache_dir)
    dst_dir = Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    target_dtype = None
    if dtype != "preserve":
        target_dtype = getattr(torch, dtype)

    # 1) Load full state dict from all shards.
    index_path = src_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = [p.name for p in src_dir.glob("*.safetensors")]
    full_sd: dict = {}
    for sf in shard_files:
        full_sd.update(load_file(str(src_dir / sf)))
    print(f"[convert] loaded {len(full_sd)} tensors from {len(shard_files)} shard(s)")

    n_fused = sum(1 for k in full_sd if k.endswith("mlp.moe.experts.w1"))
    if n_fused == 0:
        print("[convert] WARNING: no fused ``mlp.moe.experts.w1`` keys found — is this already HF layout?")

    # 2) De-fuse.
    out_sd = defuse_state_dict(full_sd, remap_mod)
    if target_dtype is not None:
        out_sd = {k: (v.to(target_dtype) if v.is_floating_point() else v) for k, v in out_sd.items()}
    print(f"[convert] de-fused -> {len(out_sd)} tensors ({n_fused} fused MoE layers expanded)")

    # 3) Shard + write safetensors (~5GB/shard) + index.
    _save_sharded(out_sd, dst_dir)

    # 4) Copy config.json (keep model_type qwen3_moe) + tokenizer/aux files.
    _copy_aux(src_dir, dst_dir)
    print(f"[convert] DONE -> {dst_dir}")


def _save_sharded(state_dict: dict, dst_dir: Path, max_shard_bytes: int = 5_000_000_000) -> None:
    from safetensors.torch import save_file

    items = list(state_dict.items())
    shards: list[dict] = [{}]
    sizes = [0]
    for k, v in items:
        v = v.contiguous()
        nbytes = v.numel() * v.element_size()
        if sizes[-1] > 0 and sizes[-1] + nbytes > max_shard_bytes:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = v
        sizes[-1] += nbytes

    n = len(shards)
    weight_map = {}
    total = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(dst_dir / fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        total += sizes[i - 1]
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"[convert] wrote {n} shard(s), {len(weight_map)} tensors, {total/1e9:.2f} GB")


def _copy_aux(src_dir: Path, dst_dir: Path) -> None:
    # config.json verbatim (model_type stays qwen3_moe; per-expert layout is what
    # the config already describes via num_experts / moe_intermediate_size).
    for p in src_dir.iterdir():
        if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json":
            continue
        if p.name == "config.json" or p.suffix in (".txt", ".model") or any(
            t in p.name for t in _PASSTHROUGH_COPY
        ) or p.name.endswith(".json"):
            shutil.copy2(p, dst_dir / p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="fused-layout checkpoint dir OR HF repo id")
    ap.add_argument("--dst", help="output dir for the HF per-expert checkpoint")
    ap.add_argument("--dtype", default="preserve", help="preserve (default) | bfloat16 | float16 | float32")
    ap.add_argument("--cache-dir", default=None, help="HF snapshot cache dir (for repo-id src)")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic numerical self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.src or not args.dst:
        ap.error("--src and --dst are required (or use --self-test)")
    convert_checkpoint(args.src, args.dst, dtype=args.dtype, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
