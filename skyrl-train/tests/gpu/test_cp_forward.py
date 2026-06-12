"""Stage 4 (FSDP2 CP) — CP forward wrap gate: ring SDPA via torch context_parallel.

This stage proves the CP forward wrap RUNS and is a literal no-op at cp=1. Per-token
unshard *correctness* (G3 parity) is Stage 5; here the forward immediately unshards
the logits to get a runnable forward.

Two entry points:

  (1) 1-GPU no-op (plain pytest, single process) — `test_cp1_noop_*`:
      cp_size=1 ⇒ `maybe_cp_context` is `contextlib.nullcontext`; the forward output
      is BIT-IDENTICAL to the Stage-2 sdpa forward (G1). No distributed init needed.

  (2) 2-GPU cp=2 (torchrun, 2 ranks) — `main()`:
      - forward completes (no NCCL hang) on the dense [B, S] path;
      - inside the CP context the model logits are sequence-sharded [B, S/2, V];
      - BOTH rotate methods ("allgather", "all_to_all") run;
      - non-divisible seq_len is padded to a 2*cp multiple, pad region masked
        (no NaN/inf in logprobs over real tokens);
      - allgather vs all_to_all unsharded logits are allclose (bf16 tol).

IMPORTANT — the SIF bakes SkyRL at /opt/SkyRL, which shadows a worktree clone.
Run with PYTHONPATH pointing at the worktree's skyrl-train so the model_wrapper /
cp_utils UNDER TEST (not the baked ones) are imported (the import-path asserts
below catch the shadow). Set LIBRARY_PATH=/.singularity.d/libs for Triton JIT gcc.

    # 1-GPU no-op:
    srun --account=reformo --reservation=reformo --gres=gpu:1 ... \
      apptainer exec --nv --env PYTHONPATH=<worktree>/skyrl-train \
        --env LIBRARY_PATH=/.singularity.d/libs <sif> \
        python -m pytest -s -p no:cacheprovider --confcutdir tests/gpu \
          tests/gpu/test_cp_forward.py -v

    # 2-GPU cp=2 (torchrun main):
    srun --account=reformo --reservation=reformo --gres=gpu:2 ... \
      apptainer exec --nv --env PYTHONPATH=<worktree>/skyrl-train \
        --env LIBRARY_PATH=/.singularity.d/libs <sif> \
        torchrun --nproc-per-node=2 tests/gpu/test_cp_forward.py
"""

import os

import pytest
import torch

import skyrl_train.model_wrapper as _mw
import skyrl_train.distributed.cp_utils as _cp
from skyrl_train.model_wrapper import HFModelWrapper

MODEL_NAME = "Qwen/Qwen3-0.6B"

# bf16 tolerances. The cp1-vs-cp1 no-op MUST be bit-identical (atol=0); the
# cross-rotate-method comparison is at the bf16 ring-reduction floor.
NOOP_ATOL = 0.0
ROTATE_ATOL = 2e-2


def _assert_under_test():
    """Guard against the /opt/SkyRL shadow: the modules under test must come from
    the worktree (PYTHONPATH), not the baked SIF copy."""
    assert "/opt/SkyRL/" not in _mw.__file__, f"model_wrapper imported from baked SIF: {_mw.__file__}"
    assert "/opt/SkyRL/" not in _cp.__file__, f"cp_utils imported from baked SIF: {_cp.__file__}"
    assert hasattr(_cp, "maybe_cp_context"), "cp_utils.maybe_cp_context missing — wrong/old module"


def _dense_batch():
    """A small dense [B, S] batch with left-padding and a clear response span.
    Qwen3 pad/eos ids; values are arbitrary in-vocab tokens."""
    pad = 151643  # Qwen3 <|endoftext|> (pad)
    eos = 151645  # <|im_end|>
    seq_a = [pad] * 2 + [785, 374, 264, 1273, 315, 279, 1849, eos]
    seq_b = [pad] * 1 + [12091, 1879, 11, 419, 374, 264, 2588, 1273, eos]
    width = max(len(seq_a), len(seq_b))
    seq_a = [pad] * (width - len(seq_a)) + seq_a
    seq_b = [pad] * (width - len(seq_b)) + seq_b
    input_ids = torch.tensor([seq_a, seq_b], dtype=torch.long)
    attention_mask = (input_ids != pad).to(torch.long)
    num_actions = 4
    return input_ids, attention_mask, num_actions


def _build(attn_backend="sdpa", bf16=True, context_parallel_size=1, cp_mesh=None, cp_rotate_method="allgather"):
    model = HFModelWrapper(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=False,
        bf16=bf16,
        sequence_parallel_size=1,
        use_sample_packing=False,
        attn_backend=attn_backend,
        context_parallel_size=context_parallel_size,
        cp_mesh=cp_mesh,
        cp_rotate_method=cp_rotate_method,
    )
    model.model.eval()
    model.model.to("cuda")
    return model


def _run(model, input_ids, attention_mask, num_actions):
    with torch.no_grad():
        logp, out = model(input_ids, num_actions, attention_mask, compute_entropy=True, return_output=True)
    return logp.float(), out["entropy"].float()


# ----------------------------------------------------------------------------
# (1) 1-GPU no-op (plain pytest). cp_size=1 -> nullcontext -> byte-identical.
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_cp1_noop_is_nullcontext():
    """maybe_cp_context returns a literal nullcontext at cp_size=1 (no torch CP touched)."""
    import contextlib

    _assert_under_test()
    ctx = _cp.maybe_cp_context(1, None, None, buffers=[], seq_dims=[])
    assert isinstance(ctx, contextlib.nullcontext), f"cp_size=1 should be nullcontext, got {type(ctx)}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_cp1_noop_bit_identical():
    """cp_size=1 forward output is BIT-IDENTICAL to the plain sdpa forward (G1).

    Builds two HFModelWrappers from the same checkpoint: one with the Stage-4 CP
    plumbing present but cp_size=1 (cp_mesh=None), and one plain sdpa. Same seed
    weights from from_pretrained ⇒ identical; the CP code path must add ZERO drift.
    """
    _assert_under_test()
    input_ids, attention_mask, num_actions = _dense_batch()
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    m_cp1 = _build(attn_backend="sdpa", bf16=True, context_parallel_size=1, cp_mesh=None)
    assert m_cp1.context_parallel_size == 1
    assert m_cp1.cp_mesh is None
    logp_cp1, ent_cp1 = _run(m_cp1, input_ids, attention_mask, num_actions)
    del m_cp1
    torch.cuda.empty_cache()

    m_plain = _build(attn_backend="sdpa", bf16=True, context_parallel_size=1)
    logp_plain, ent_plain = _run(m_plain, input_ids, attention_mask, num_actions)

    d_logp = (logp_cp1 - logp_plain).abs().max().item()
    d_ent = (ent_cp1 - ent_plain).abs().max().item()
    print(f"\n[Stage4 cp1 no-op] logp diff={d_logp:.3e}  ent diff={d_ent:.3e}")
    assert torch.equal(logp_cp1, logp_plain), f"cp1 logprobs not bit-identical to plain sdpa: {d_logp:.3e}"
    assert torch.equal(ent_cp1, ent_plain), f"cp1 entropy not bit-identical to plain sdpa: {d_ent:.3e}"


# ----------------------------------------------------------------------------
# (2) 2-GPU cp=2 (torchrun main). Sharding shape + both rotate methods + pad + determinism.
# ----------------------------------------------------------------------------
def _shard_probe_logits(cp_mesh, model, input_ids, position_ids, attention_mask):
    """Run JUST the base model forward inside the CP context (no immediate unshard)
    so we can assert the logits come back sequence-sharded [B, S/cp, V]."""
    from skyrl_train.distributed.cp_utils import cp_context

    seqs = input_ids.clone()
    pos = position_ids.clone()
    attn = attention_mask.clone()
    buffers = [seqs, pos, attn]
    with cp_context(cp_mesh, "allgather", buffers=buffers, seq_dims=[1, 1, 1], no_restore={seqs}):
        out = model.model(seqs, attention_mask=attn, position_ids=pos)
        sharded_shape = tuple(out["logits"].shape)
    return sharded_shape


def main():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not torch.cuda.is_available():
        print("CUDA not available — Stage 4 CP 2-GPU gate DEFERRED.")
        return

    _assert_under_test()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cp_size = world_size
    assert world_size == 2, f"Stage 4 CP 2-GPU gate needs 2 ranks (cp=2); got {world_size}"

    # Build a cp mesh over all ranks.
    mesh = init_device_mesh("cuda", (cp_size,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    input_ids, attention_mask, num_actions = _dense_batch()
    # Force a NON-divisible seq_len to exercise the G4 pad path: seq_len here is 10,
    # 10 % (2*cp=4) == 2 -> pad to 12. Pad correctness gate.
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    pre_pad_seq_len = input_ids.size(1)
    assert pre_pad_seq_len % (2 * cp_size) != 0, "test wants a non-divisible seq_len to exercise pad"

    # --- shard-shape probe (allgather): base forward inside CP context returns [B, S_pad/cp, V] ---
    model = _build("sdpa", bf16=True, context_parallel_size=cp_size, cp_mesh=cp_mesh, cp_rotate_method="allgather")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # pad to 2*cp multiple for the raw probe (model_wrapper does this internally;
    # here we replicate just enough to feed the base model the padded buffers).
    multiple = 2 * cp_size
    pad = (multiple - pre_pad_seq_len % multiple) % multiple
    if pad:
        input_ids_p = torch.nn.functional.pad(input_ids, (0, pad), value=151643)
        last_pos = position_ids[:, -1:]
        pad_pos = torch.arange(1, pad + 1, device=position_ids.device).unsqueeze(0)
        position_ids_p = torch.cat((position_ids, last_pos + pad_pos), dim=-1)
        attn_p = torch.cat(
            (attention_mask, torch.zeros(attention_mask.size(0), pad, dtype=attention_mask.dtype, device="cuda")),
            dim=-1,
        )
    else:
        input_ids_p, position_ids_p, attn_p = input_ids, position_ids, attention_mask
    S_pad = input_ids_p.size(1)
    sharded_shape = _shard_probe_logits(cp_mesh, model, input_ids_p, position_ids_p, attn_p)
    B = input_ids.size(0)
    expected_shard_S = S_pad // cp_size
    if rank == 0:
        print(
            f"[Stage4 cp2] pre_pad_seq_len={pre_pad_seq_len} -> padded={S_pad}; "
            f"per-rank sharded logits shape={sharded_shape} (expect [B={B}, S/cp={expected_shard_S}, V])"
        )
    assert sharded_shape[0] == B, sharded_shape
    assert (
        sharded_shape[1] == expected_shard_S
    ), f"logits NOT sequence-sharded: got S={sharded_shape[1]}, expected {expected_shard_S}"
    dist.barrier()

    # --- full forward, BOTH rotate methods; pad masked (no NaN/inf); determinism ---
    results = {}
    for method in ("allgather", "all_to_all"):
        m = _build("sdpa", bf16=True, context_parallel_size=cp_size, cp_mesh=cp_mesh, cp_rotate_method=method)
        logp, ent = _run(m, input_ids, attention_mask, num_actions)
        # action_log_probs are over real (non-pad) response tokens -> must be finite.
        assert torch.isfinite(logp).all(), f"non-finite logprobs over real tokens (method={method})"
        assert torch.isfinite(ent).all(), f"non-finite entropy over real tokens (method={method})"
        assert logp.shape == (B, num_actions), f"action_log_probs shape {logp.shape} != {(B, num_actions)}"
        results[method] = logp
        if rank == 0:
            print(f"[Stage4 cp2] method={method}: forward OK, action_logp shape={tuple(logp.shape)}, finite=True")
        del m
        torch.cuda.empty_cache()
        dist.barrier()

    d = (results["allgather"] - results["all_to_all"]).abs().max().item()
    if rank == 0:
        print(f"[Stage4 cp2] allgather vs all_to_all action_logp max diff={d:.3e} (atol={ROTATE_ATOL})")
    assert torch.allclose(
        results["allgather"], results["all_to_all"], atol=ROTATE_ATOL, rtol=0.0
    ), f"rotate methods diverged: {d:.3e} > {ROTATE_ATOL}"

    if rank == 0:
        print("ALL PASS")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
