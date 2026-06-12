"""Mixtral-8x7B EP=1 x FSDP distributed smoke (the OLMoE lesson).

Exercises the EXACT production code path for the Jupiter Mixtral RL config on the
REAL multi-rank EP=1 x FSDP topology — the path a single-process GPU smoke
CANNOT cover (the OLMoE EP=2 self-kill was a distributed state-dict-load bug):

  1. Build a structurally-REAL Mixtral (8 experts top-2, hidden 4096,
     intermediate 14336, GQA 32/8) but with a reduced layer count so it fits a
     4-GPU node quickly. The MoE block + fused expert nn.Parameters + router are
     byte-identical to the 47B model; only num_hidden_layers is shrunk.
  2. swap_moe_blocks_to_grouped  -> grouped-GEMM MoE (Mixtral bare-tensor shim).
  3. create_device_mesh(ep_size=1) + apply_fsdp2 (EP=1 => apply_ep NEVER called,
     no "ep" mesh dim, no 2-D expert-DTensor composition — the bug is sidestepped
     by construction).
  4. fsdp2_load_full_state_dict(ep_enabled=False) — the REAL distributed loader
     (naive non-EP branch), broadcasting rank-0 weights into the FSDP shards via
     load_state_dict(assign=True). THIS is the OLMoE-killer surface.
  5. One forward + backward microstep -> assert finite loss + finite grad-norm.

If the distributed load throws the EP-composition RuntimeError
("start+length exceeds dimension"), this STOPS with a clear marker — do NOT
blind-patch the shared EP loader (Qwen3-Coder 849580 depends on it).

Run (4 GPUs, from skyrl-train dir):
    torchrun --nproc_per_node=4 tests/gpu/mixtral_ep1_fsdp_smoke.py
"""

import os
import sys

import torch
import torch.distributed as dist


def _log(msg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[mixtral-smoke rank{rank}] {msg}", file=sys.stderr, flush=True)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    fsdp_size = world  # EP=1 x FSDP=world

    from transformers import AutoConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
    from skyrl_train.models.layers.moe_swap import swap_moe_blocks_to_grouped
    from skyrl_train.distributed.fsdp_utils import (
        create_device_mesh,
        apply_fsdp2,
        fsdp2_load_full_state_dict,
    )
    from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy

    MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    cfg = AutoConfig.from_pretrained(MODEL)
    # Structurally real, layer-reduced so it loads fast on 4 GPUs. Keep the FULL
    # MoE topology (8 experts, top_k 2, intermediate 14336, hidden 4096, GQA).
    cfg.num_hidden_layers = 2
    cfg.use_cache = False
    if rank == 0:
        _log(
            f"config: experts={cfg.num_local_experts} top_k={cfg.num_experts_per_tok} "
            f"hidden={cfg.hidden_size} inter={cfg.intermediate_size} "
            f"layers={cfg.num_hidden_layers} attn={cfg.num_attention_heads}/{cfg.num_key_value_heads}KV"
        )

    # ---- Mirror the EP=1 production seam EXACTLY (fsdp_strategy.py:332):
    # build the model with REAL weights, run swap_moe_blocks_to_grouped on it, and
    # snapshot full_state = module.state_dict() AFTER the swap (so full_state
    # already carries the grouped `mlp.moe.{router,experts}.*` keys that match the
    # swapped meta module's state_dict). The non-EP loader zips full_sd with the
    # meta sharded sd by POSITION, so the two must be the SAME post-swap module
    # structure. All ranks build+swap on meta; weights flow in via the loader.
    torch.manual_seed(0)
    if rank == 0:
        real = MixtralForCausalLM(cfg).to(torch.bfloat16)
        n_swapped_real = swap_moe_blocks_to_grouped(real)
        assert n_swapped_real == cfg.num_hidden_layers
        full_state = {k: v.detach().to("cpu", copy=True) for k, v in real.state_dict().items()}
        del real
        torch.cuda.empty_cache()
    else:
        full_state = {}

    with torch.device("meta"):
        model = MixtralForCausalLM(cfg).to(torch.bfloat16)

    # ---- Stage 3b grouped-GEMM swap (Mixtral bare-tensor shim) on the meta module.
    n_swapped = swap_moe_blocks_to_grouped(model)
    if rank == 0:
        _log(f"swap_moe_blocks_to_grouped: swapped {n_swapped} blocks (expect {cfg.num_hidden_layers})")
    assert n_swapped == cfg.num_hidden_layers, f"expected {cfg.num_hidden_layers} swaps, got {n_swapped}"

    # ---- EP=1 mesh + FSDP2 wrap (apply_ep is NOT called for ep_size==1).
    device_mesh = create_device_mesh(world_size=world, fsdp_size=fsdp_size, ep_size=1)
    if rank == 0:
        _log(f"device_mesh: {device_mesh} (ep_size=1 => no 'ep' dim => apply_ep skipped)")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
    )
    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": CPUOffloadPolicy(pin_memory=True),
        "reshard_after_forward": True,
    }
    apply_fsdp2(model, fsdp_kwargs, {"cpu_offload": True})
    if rank == 0:
        _log("apply_fsdp2 done")

    # ---- THE OLMoE-KILLER SURFACE: real distributed state-dict load.
    try:
        fsdp2_load_full_state_dict(model, full_state, cpu_offload=None, ep_enabled=False)
    except RuntimeError as e:
        if "exceeds dimension" in str(e) or "start" in str(e).lower():
            _log(f"!!! EP-COMPOSITION LOAD FAILURE (STOP, do NOT patch): {e}")
        else:
            _log(f"!!! distributed load RuntimeError: {e}")
        raise
    dist.barrier()
    if rank == 0:
        _log("fsdp2_load_full_state_dict OK (distributed load + assign=True)")

    # ---- One forward + backward microstep -> finite loss + grad.
    torch.manual_seed(100 + rank)
    seq = 64
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq), device="cuda")
    labels = input_ids.clone()
    model.train()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    _log(f"forward loss: {loss.item():.4f} finite={torch.isfinite(loss).item()}")
    assert torch.isfinite(loss).item(), "non-finite loss"
    loss.backward()

    # grad-norm across FSDP2 (DTensor-aware).
    from skyrl_train.distributed.fsdp_utils import fsdp2_clip_grad_norm_

    gn = fsdp2_clip_grad_norm_(model.parameters(), max_norm=0.9)
    gn_val = float(gn.item() if torch.is_tensor(gn) else gn)
    if rank == 0:
        _log(f"grad_norm: {gn_val:.4f} finite={torch.isfinite(torch.tensor(gn_val)).item()}")
    assert torch.isfinite(torch.tensor(gn_val)).item(), "non-finite grad_norm"

    dist.barrier()
    if rank == 0:
        _log("==== MIXTRAL EP=1 x FSDP DISTRIBUTED SMOKE PASSED ====")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
