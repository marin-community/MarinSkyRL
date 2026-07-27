#!/usr/bin/env python
"""Consolidate a sharded RL training checkpoint into a single HF safetensors model.

WHY THIS EXISTS
---------------
``FSDPStrategy.save_checkpoint`` writes the *training* checkpoint as one file per
rank, holding an FSDP SHARDED state dict::

    model_world_size_{W}_rank_{R}.pt      # DTensor shards of the model
    optim_world_size_{W}_rank_{R}.pt      # optimizer state (not needed here)
    extra_state_world_size_{W}_rank_{R}.pt

The *inference* artifact is written by a different path, ``save_hf_model``, which
gathers a full state dict and calls ``save_pretrained``. When that path does not
run — it is gated on ``hf_save_interval`` and has failed outright before — the run
finishes with training checkpoints and no loadable model. Two completed sweep runs
are in exactly that state: 80/80 steps banked, nothing publishable.

This script closes that gap offline: it reassembles the per-rank shards into full
tensors and writes an HF-layout safetensors directory, with no GPUs and no process
group.

WHAT IT DOES NOT DO
-------------------
It does not reshard, retrain, or convert optimizer state. It reads model shards
only. If the checkpoint's MoE weights are in the fused grouped layout, pass
``--defuse-moe`` and the conversion is delegated to the reviewed implementation in
``convert_fused_moe_to_hf.py`` rather than reimplemented here.

START WITH ``--inspect``
------------------------
Reassembly depends on how each parameter was sharded, and that varies with the
FSDP/EP/TP mesh a run used. ``--inspect`` reports what is actually on disk —
world size, parameter count, dtypes, and the placement of every distinct shard
pattern — and writes nothing. Run it first and read the report. A checkpoint whose
placements this script cannot reassemble is reported as such and the run aborts;
it never writes a partially-reassembled model, because a silently wrong set of
weights is far more expensive than a failed conversion.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

RANK_FILE_RE = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")


def find_shards(ckpt_dir: Path) -> Tuple[int, Dict[int, Path]]:
    """Return (world_size, {rank: path}) for the model shards in ckpt_dir."""
    shards: Dict[int, Path] = {}
    world_sizes = set()
    for p in sorted(ckpt_dir.iterdir()):
        m = RANK_FILE_RE.match(p.name)
        if not m:
            continue
        world_sizes.add(int(m.group(1)))
        shards[int(m.group(2))] = p

    if not shards:
        raise SystemExit(
            f"No model_world_size_*_rank_*.pt files in {ckpt_dir}.\n"
            f"Point --src at the global_step_N directory that holds the rank files."
        )
    if len(world_sizes) != 1:
        raise SystemExit(f"Mixed world sizes in {ckpt_dir}: {sorted(world_sizes)}. Refusing to guess.")

    world_size = world_sizes.pop()
    missing = sorted(set(range(world_size)) - set(shards))
    if missing:
        raise SystemExit(
            f"Checkpoint is incomplete: world_size={world_size} but ranks {missing} are absent.\n"
            f"Every rank file is required — a missing rank means missing weights, not smaller weights."
        )
    return world_size, shards


def load_shard(path: Path) -> Dict[str, Any]:
    """Load one rank file on CPU. Deliberately not weights_only: the values are DTensors."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Some writers nest the state dict under a key; accept either shape.
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def describe(value: Any) -> str:
    """One-line description of a shard value, for the inspect report."""
    placements = getattr(value, "placements", None)
    if placements is not None:
        local = value.to_local()
        return f"DTensor placements={placements} local={tuple(local.shape)} global={tuple(value.shape)}"
    if torch.is_tensor(value):
        return f"Tensor {tuple(value.shape)}"
    return type(value).__name__


def inspect(ckpt_dir: Path) -> None:
    world_size, shards = find_shards(ckpt_dir)
    print(f"checkpoint : {ckpt_dir}")
    print(f"world_size : {world_size}")
    print(f"rank files : {len(shards)}")

    rank0 = load_shard(shards[0])
    print(f"parameters : {len(rank0)}")

    dtypes = defaultdict(int)
    patterns = defaultdict(list)
    for k, v in rank0.items():
        t = v.to_local() if hasattr(v, "to_local") else v
        if torch.is_tensor(t):
            dtypes[str(t.dtype)] += 1
        patterns[describe(v).split(" local=")[0]].append(k)

    print(f"dtypes     : {dict(dtypes)}")
    print("\nshard patterns (rank 0):")
    for pat, keys in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(keys):5d}  {pat}")
        print(f"         e.g. {keys[0]}")

    fused = [k for k in rank0 if ".experts.w1" in k or ".moe.experts." in k]
    if fused:
        print(f"\nFused grouped-MoE weights present ({len(fused)} keys, e.g. {fused[0]}).")
        print("Pass --defuse-moe so the output is loadable by the HF/vLLM per-expert loader.")


def full_tensor(key: str, per_rank: List[Any]) -> torch.Tensor:
    """Reassemble one parameter from its per-rank shards.

    Handles the placements this trainer actually produces: a parameter is either
    replicated (every rank holds an identical copy) or sharded along one dim. Any
    other placement raises, because guessing a reassembly order is how a checkpoint
    gets silently corrupted.
    """
    first = per_rank[0]

    if not hasattr(first, "placements"):
        if torch.is_tensor(first):
            return first  # plain tensor, already whole
        raise TypeError(f"{key}: unsupported shard value {type(first).__name__}")

    placements = first.placements
    shard_dims = [p.dim for p in placements if p.is_shard()]

    if not shard_dims:
        return first.to_local()  # fully replicated

    if len(set(shard_dims)) > 1:
        raise NotImplementedError(
            f"{key}: sharded on multiple dims {sorted(set(shard_dims))}. "
            f"Reassembly order is mesh-dependent and is not inferred here."
        )

    dim = shard_dims[0]
    expected = tuple(first.shape)

    # Ranks are visited in order and their shards concatenated until the parameter
    # is whole. On a multi-dimensional mesh the later ranks hold replicas of shards
    # already collected, so stopping at the declared size is what keeps a replicated
    # dimension from multiplying the tensor.
    parts: List[torch.Tensor] = []
    filled = 0
    for v in per_rank:
        if filled >= expected[dim]:
            break
        local = v.to_local()
        parts.append(local)
        filled += local.shape[dim]

    merged = torch.cat(parts, dim=dim)
    if tuple(merged.shape) != expected:
        raise ValueError(
            f"{key}: reassembled to {tuple(merged.shape)} but the DTensor declares {expected}. "
            f"Refusing to write a mismatched parameter."
        )
    return merged


def consolidate(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    world_size, shards = find_shards(ckpt_dir)
    print(f"[consolidate] world_size={world_size}, loading {len(shards)} rank files")

    loaded = {r: load_shard(p) for r, p in sorted(shards.items())}
    keys = list(loaded[0].keys())

    for r, sd in loaded.items():
        if set(sd.keys()) != set(keys):
            raise SystemExit(f"rank {r} has a different parameter set than rank 0. Refusing to merge.")

    out: Dict[str, torch.Tensor] = {}
    for i, k in enumerate(keys):
        out[k] = full_tensor(k, [loaded[r][k] for r in sorted(loaded)])
        if (i + 1) % 100 == 0:
            print(f"[consolidate] {i + 1}/{len(keys)} parameters")

    print(f"[consolidate] {len(out)} parameters reassembled")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="checkpoint dir holding model_world_size_*_rank_*.pt")
    ap.add_argument("--dst", help="output dir for the HF safetensors model")
    ap.add_argument("--inspect", action="store_true", help="report the on-disk layout and exit without writing")
    ap.add_argument(
        "--defuse-moe", action="store_true", help="also de-fuse grouped MoE weights to HF per-expert layout"
    )
    ap.add_argument("--aux-from", help="dir or HF repo id to copy config.json/tokenizer from")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    if args.inspect:
        inspect(src)
        return

    if not args.dst:
        raise SystemExit("--dst is required unless --inspect is given")

    state = consolidate(src)

    sys.path.insert(0, str(Path(__file__).parent))
    from convert_fused_moe_to_hf import _copy_aux, _save_sharded  # noqa: E402

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if args.defuse_moe:
        from convert_fused_moe_to_hf import _load_remap_module, defuse_state_dict  # noqa: E402

        print("[defuse] converting grouped MoE weights to HF per-expert layout")
        state = defuse_state_dict(state, _load_remap_module())

    _save_sharded(state, dst)

    aux = Path(args.aux_from) if args.aux_from else src
    if aux.is_dir():
        _copy_aux(aux, dst)
    else:
        print(f"[aux] {aux} is not a local dir; copy config.json and tokenizer files into {dst} before publishing")

    cfg = dst / "config.json"
    if not cfg.exists():
        print(
            f"\nWARNING: {cfg} is missing. The weights are written but the directory is not "
            f"loadable until config.json and the tokenizer files are present. Use --aux-from."
        )
    print(f"\n[done] {len(state)} tensors -> {dst}")
    print(json.dumps({"tensors": len(state), "dst": str(dst)}, indent=1))


if __name__ == "__main__":
    main()
