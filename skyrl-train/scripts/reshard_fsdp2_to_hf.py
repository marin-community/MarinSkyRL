#!/usr/bin/env python3
"""Consolidate a SkyRL FSDP2 sharded checkpoint into Hugging Face safetensors.

SkyRL saves each rank as a ``SHARDED_STATE_DICT`` of one-dimensional ``Shard(0)``
DTensors. This script reconstructs each tensor by streaming local shards in rank
order, writes the resulting model, uploads it to Hugging Face, and verifies the
saved artifact. It does not need a process group or GPU to read the rank files.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import types

import torch
import torch._tensor as _tt
import torch._utils as _tu
import s3fs
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ----- stub unpickler: extract per-rank local Shard(0) tensors on any torch ------
def _install_stubs():
    """Make torch.load able to reconstruct the newer-torch DTensor pickle here by
    stubbing the classes/module this (older) torch lacks. The DTensor OUTER is a
    placeholder (we never use its global storage); the LOCAL tensor and the spec
    (placements) come through ``state[1]`` intact as real objects."""
    if "torch.distributed._mesh_layout" not in sys.modules:
        m = types.ModuleType("torch.distributed._mesh_layout")

        class _Cap:
            def __init__(self, *a, **k):
                self.a, self.k = a, k

            def __setstate__(self, s):
                self.state = s

        m._MeshLayout = _Cap
        sys.modules["torch.distributed._mesh_layout"] = m
        import torch.distributed.tensor._dtensor_spec as dspec

        if not hasattr(dspec, "ShardOrderEntry"):
            dspec.ShardOrderEntry = _Cap

    def _cap_wrap(cls, dtype, size, stride, requires_grad, backward_hooks, *rest):
        t = torch.empty(0)
        t._g = (tuple(size), str(dtype))
        return t

    _tu._rebuild_wrapper_subclass = _cap_wrap


def _extract_shard(path):
    """Return {key: (local_tensor, placements, global_shape)} for one rank file."""
    out = {}

    def _cap_from_type(func, new_type, args, state):
        inner = func(*args)  # placeholder outer DTensor
        d = state[1] if isinstance(state, tuple) and len(state) > 1 and isinstance(state[1], dict) else {}
        lt = d.get("_local_tensor")
        spec = d.get("_spec")
        pl = getattr(spec, "placements", None)
        pl_info = [(type(p).__name__, getattr(p, "dim", None)) for p in pl] if pl else None
        tm = getattr(spec, "tensor_meta", None) if spec is not None else None
        gshape = tuple(getattr(tm, "shape")) if tm is not None and hasattr(tm, "shape") else None
        _cap_from_type.buf.append((lt, pl_info, gshape))
        return inner

    _cap_from_type.buf = []
    _tt._rebuild_from_type_v2 = _cap_from_type
    _tu._rebuild_from_type_v2 = _cap_from_type
    sd = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(sd.keys())
    assert len(keys) == len(_cap_from_type.buf), f"{len(keys)} keys vs {len(_cap_from_type.buf)} records"
    for k, (lt, pl, gs) in zip(keys, _cap_from_type.buf):
        out[k] = (lt, pl, gs)
    return out


def download_shards(s3_prefix, world_size, local_dir, endpoint_url):
    """Fetch the ws model shards + huggingface/ config dir from S3 via s3fs.

    Credentials are read from ``CW_AKID``/``CW_SECRET`` when set, otherwise the
    standard AWS environment variables. The endpoint defaults to ``cwobject.com``.
    """
    # Env is authoritative: these creds live on the EXTERNAL cwobject.com store
    # (NOT the cluster-local cwlota.com). Coerce region 'auto' -> concrete us-east-1
    # (aiobotocore SigV4 needs a real region against cwobject.com).
    ep = endpoint_url or "https://cwobject.com"
    region = os.environ.get("AWS_REGION") or "us-east-1"
    if region == "auto":
        region = "us-east-1"
    client_kwargs = {"region_name": region}
    config_kwargs = {"s3": {"addressing_style": "virtual"}}
    # Prefer creds passed under non-standard names (CW_AKID/CW_SECRET) so the
    # cluster's auto-injected iris-task-env AWS_* (a different object store) cannot
    # shadow them.
    key = os.environ.get("CW_AKID") or os.environ["AWS_ACCESS_KEY_ID"]
    secret = os.environ.get("CW_SECRET") or os.environ["AWS_SECRET_ACCESS_KEY"]
    print(f"[reshard] s3fs endpoint={ep} region={region}", flush=True)
    fs = s3fs.S3FileSystem(
        key=key,
        secret=secret,
        endpoint_url=ep,
        client_kwargs=client_kwargs,
        config_kwargs=config_kwargs,
    )
    if not s3_prefix.startswith("s3://"):
        raise ValueError(f"--s3-prefix must start with s3://, got {s3_prefix!r}")
    prefix = s3_prefix.removeprefix("s3://").rstrip("/")
    os.makedirs(os.path.join(local_dir, "huggingface"), exist_ok=True)
    hf_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for fn in hf_files:
        fs.get_file(f"{prefix}/huggingface/{fn}", os.path.join(local_dir, "huggingface", fn))
        print(f"[reshard] downloaded huggingface/{fn}", flush=True)
    for r in range(world_size):
        fn = f"model_world_size_{world_size}_rank_{r}.pt"
        dst = os.path.join(local_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        fs.get_file(f"{prefix}/{fn}", dst)
        print(f"[reshard] downloaded {fn} ({os.path.getsize(dst) / 1e9:.2f} GB)", flush=True)


def consolidate(local_dir, world_size):
    """Stream the ws rank files into full CPU tensors via cat along dim-0."""
    _install_stubs()
    full = {}
    offsets = {}
    global_shapes = {}
    for r in range(world_size):
        path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{r}.pt")
        print(f"[reshard] extracting rank {r}: {path}", flush=True)
        shard = _extract_shard(path)
        for k, (lt, pl, gs) in shard.items():
            assert pl == [("Shard", 0)], f"{k}: unexpected placement {pl} (only even Shard(0) supported)"
            if r == 0:
                assert gs is not None, f"{k}: missing global shape"
                full[k] = torch.empty(gs, dtype=lt.dtype)
                offsets[k] = 0
                global_shapes[k] = tuple(gs)
                # even-sharding sanity (informational): dim0 divisible by ws
                if gs[0] % world_size != 0:
                    print(
                        f"[reshard] NOTE {k}: dim0 {gs[0]} not divisible by {world_size}; "
                        f"relying on contiguous rank order (still exact for Shard(0))",
                        flush=True,
                    )
            n = lt.shape[0]
            o = offsets[k]
            full[k][o : o + n].copy_(lt)
            offsets[k] = o + n
        del shard
    # Verify each tensor fully filled.
    for k in full:
        assert offsets[k] == global_shapes[k][0], f"{k}: filled {offsets[k]} rows != global {global_shapes[k][0]}"
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True, help="dir with model_world_size_*_rank_*.pt + huggingface/")
    ap.add_argument("--world-size", type=int, default=16)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--s3-prefix", default=None, help="if set, download shards from here first")
    ap.add_argument("--endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL", "https://cwobject.com"))
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    if args.s3_prefix:
        download_shards(args.s3_prefix, args.world_size, args.local_dir, args.endpoint_url)

    hf_src = os.path.join(args.local_dir, "huggingface")
    with open(os.path.join(hf_src, "config.json")) as config_file:
        config = json.load(config_file)
    print(
        f"[reshard] config: model_type={config['model_type']} heads={config['num_attention_heads']} "
        f"layers={config['num_hidden_layers']} vocab={config['vocab_size']}",
        flush=True,
    )

    full = consolidate(args.local_dir, args.world_size)
    print(f"[reshard] consolidated {len(full)} tensors", flush=True)
    total = sum(v.numel() for v in full.values())
    print(f"[reshard] total params = {total:,} ({total / 1e9:.2f}B)", flush=True)

    # finite spot-checks on a few load-bearing tensors
    for k in [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ]:
        t = full[k].float()
        assert torch.isfinite(t).all(), f"{k}: non-finite!"
        assert t.abs().sum() > 0, f"{k}: all-zero!"
        print(
            f"[reshard] {k}: shape={tuple(full[k].shape)} finite ok mean={t.mean():.4e} std={t.std():.4e}", flush=True
        )

    # ---- save HF model: meta model provides architecture, weights come from state_dict ----
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = AutoConfig.from_pretrained(hf_src)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    # load_state_dict(strict=True) is an explicit key+shape correctness gate;
    # assign=True swaps in our consolidated tensors without an extra copy.
    model.load_state_dict(full, strict=True, assign=True)
    print(f"[reshard] load_state_dict strict ok: key set matches Qwen3 exactly ({len(full)} tensors)", flush=True)
    del full
    gc.collect()
    model.save_pretrained(args.out_dir, safe_serialization=True)
    print(f"[reshard] wrote safetensors to {args.out_dir}", flush=True)

    # ---- copy tokenizer + config files from checkpoint huggingface/ dir ----
    for fn in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = os.path.join(hf_src, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out_dir, fn))
    # save_pretrained rewrote config.json from cfg; restore the checkpoint's canonical one
    shutil.copy2(os.path.join(hf_src, "config.json"), os.path.join(args.out_dir, "config.json"))

    # ---- verify + (if needed) patch the two Stage-B/C invariants ----
    tokenizer_config_path = os.path.join(args.out_dir, "tokenizer_config.json")
    with open(tokenizer_config_path) as tokenizer_config_file:
        tc = json.load(tokenizer_config_file)
    if tc.get("tokenizer_class") != "PreTrainedTokenizerFast":
        tc["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(tokenizer_config_path, "w") as tokenizer_config_file:
            json.dump(tc, tokenizer_config_file, ensure_ascii=False, indent=2)
        print("[reshard] PATCHED tokenizer_class -> PreTrainedTokenizerFast", flush=True)
    else:
        print("[reshard] tokenizer_class already PreTrainedTokenizerFast (ok)", flush=True)

    with open(os.path.join(args.out_dir, "config.json")) as output_config_file:
        outcfg = json.load(output_config_file)
    print(f"[reshard] rope: theta={outcfg.get('rope_theta')} scaling={outcfg.get('rope_scaling')}", flush=True)
    assert outcfg.get("rope_theta") == 500000.0, "rope_theta != 500000"
    assert (outcfg.get("rope_scaling") or {}).get("rope_type") == "llama3", "rope_type != llama3"

    print("[reshard] output file list:", sorted(os.listdir(args.out_dir)), flush=True)

    if args.no_upload:
        print("[reshard] --no-upload: skipping HF push", flush=True)
        return

    # ---- upload ----
    os.environ.pop("HF_HUB_OFFLINE", None)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.hf_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=args.out_dir,
        repo_id=args.hf_repo,
        repo_type="model",
        commit_message="Consolidate FSDP2 checkpoint as HF safetensors",
    )
    print(f"[reshard] UPLOADED to https://huggingface.co/{args.hf_repo}", flush=True)
    print("[reshard] HF repo files:", sorted(api.list_repo_files(args.hf_repo)), flush=True)

    # ---- verification: load the uploaded artifact (from local out_dir == uploaded bytes) ----
    del model  # free the consolidated model before loading a fresh copy
    gc.collect()
    vcfg = AutoConfig.from_pretrained(args.out_dir)
    assert vcfg.num_attention_heads == 42, vcfg.num_attention_heads
    tok = AutoTokenizer.from_pretrained(args.out_dir)
    assert tok.convert_tokens_to_ids("<|start_think|>") == 128002, tok.convert_tokens_to_ids("<|start_think|>")
    vmodel = AutoModelForCausalLM.from_pretrained(args.out_dir, torch_dtype=torch.bfloat16)
    nparam = sum(p.numel() for p in vmodel.parameters())
    print(
        f"[reshard][verify] loaded OK: params={nparam:,} ({nparam / 1e9:.2f}B) heads={vcfg.num_attention_heads}",
        flush=True,
    )
    assert 24e9 < nparam < 27e9, f"param count {nparam} not ~25B"
    for name in ["model.embed_tokens.weight", "lm_head.weight", "model.layers.25.mlp.down_proj.weight"]:
        t = dict(vmodel.named_parameters())[name].float()
        assert torch.isfinite(t).all() and t.abs().sum() > 0, name
        print(f"[reshard][verify] {name}: finite ok std={t.std():.4e}", flush=True)
    print("[reshard][verify] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
