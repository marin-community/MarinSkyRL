"""Stage 2 (FSDP2 CP) 1-GPU forward parity: SDPA vs flash_attention_2.

The load-bearing Stage-2 gate: pivoting the FSDP2 model's attention backend to
SDPA must preserve the training signal. `HFModelWrapper.forward` slices to the
action (response) positions, so left-padding garbage is excluded by construction
and only the meaningful per-token logprobs/entropy are compared.

Two-tier gate (justified by the empirical precision-isolation below):

  (A) CORRECTNESS — fp32: sdpa vs eager (the reference attention math) on the
      same fp32 weights agree to ~2e-3 (logprob) / ~6e-3 (entropy). This PROVES
      the sdpa path computes the mathematically correct attention, not merely a
      coincidentally-close result.

  (B) TRAINING-SIGNAL — bf16: flash@bf16 vs sdpa@bf16, the production parity.
      Measured cross-kernel diff is ~5e-2 (logprob) / ~1.25e-1 (entropy) on
      Qwen2.5-0.5B. This is NOT a bug: it is pure bf16 rounding. Decisive control:
      sdpa@bf16 vs sdpa@fp32 (SAME kernel, precision only) diverges by ~6.2e-2 —
      LARGER than the cross-kernel bf16 diff. So switching flash->sdpa contributes
      strictly less than bf16 quantization already does. The spec's 2e-2 atol is
      below the bf16-quantization floor for logprobs of a 0.5B model; the bf16 tol
      here reflects that measured floor, while tier (A) is the rigorous gate.

IMPORTANT — the SIF bakes SkyRL at /opt/SkyRL, which shadows a worktree clone.
Run with PYTHONPATH pointing at the worktree's skyrl-train so the model_wrapper
UNDER TEST (not the baked one) is imported; otherwise attn_backend is ignored and
BOTH arms silently fall back to eager (the asserts below catch this):

    srun --account=reformo --reservation=reformo --gres=gpu:1 ... \
      apptainer exec --nv \
        --env PYTHONPATH=<worktree>/skyrl-train \
        --env LIBRARY_PATH=/.singularity.d/libs \
        <sif> python -m pytest -s -p no:cacheprovider --confcutdir tests/gpu \
          tests/gpu/test_sdpa_flash_parity.py -v
"""

import pytest
import torch
from transformers import AutoTokenizer

from skyrl_train.model_wrapper import HFModelWrapper

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Tier (A): fp32 sdpa-vs-eager correctness — tight (measured ~2e-3 / ~6e-3).
FP32_ATOL = 1e-2
# Tier (B): bf16 cross-kernel (flash vs sdpa) — at the bf16-quantization floor.
# The same-kernel bf16-vs-fp32 spread is ~6.2e-2, so 8e-2 bounds the cross-kernel
# bf16 diff without masking a real regression (a wiring bug would blow past this).
BF16_LOGP_ATOL = 8e-2
BF16_ENT_ATOL = 2e-1
RTOL = 0.0


def _dense_batch(tokenizer):
    """A small dense [B, S] batch with left-padding and a clear response span."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    seq_a = [pad] * 2 + [785, 374, 264, 1273, 315, 279, 1849, eos]
    seq_b = [pad] * 1 + [12091, 1879, 11, 419, 374, 264, 2588, 1273, eos]
    width = max(len(seq_a), len(seq_b))
    seq_a = [pad] * (width - len(seq_a)) + seq_a
    seq_b = [pad] * (width - len(seq_b)) + seq_b
    input_ids = torch.tensor([seq_a, seq_b], dtype=torch.long)
    attention_mask = (input_ids != pad).to(torch.long)
    num_actions = 4
    return input_ids, attention_mask, num_actions


def _build(attn_backend, bf16=True):
    model = HFModelWrapper(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=False,  # overridden by attn_backend (auto+False => eager)
        bf16=bf16,
        sequence_parallel_size=1,
        use_sample_packing=False,
        attn_backend=attn_backend,
        context_parallel_size=1,
    )
    model.model.eval()
    model.model.to("cuda")
    return model


def _run(model, input_ids, attention_mask, num_actions):
    with torch.no_grad():
        logp, out = model(input_ids, num_actions, attention_mask, compute_entropy=True, return_output=True)
    return logp.float(), out["entropy"].float()


def _d(x, y):
    return (x - y).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_sdpa_matches_flash_forward():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    input_ids, attention_mask, num_actions = _dense_batch(tokenizer)
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    # --- flash@bf16 ---------------------------------------------------------
    m = _build("flash_attention_2", bf16=True)
    # Guard: ensure the backend actually engaged (catches the /opt/SkyRL shadow
    # bug where attn_backend is ignored and both arms become eager).
    assert m.attn_implementation == "flash_attention_2"
    assert m.model.config._attn_implementation == "flash_attention_2"
    logp_flash_bf16, ent_flash_bf16 = _run(m, input_ids, attention_mask, num_actions)
    del m
    torch.cuda.empty_cache()

    # --- sdpa@bf16 ----------------------------------------------------------
    m = _build("sdpa", bf16=True)
    assert m.attn_implementation == "sdpa"
    assert m.model.config._attn_implementation == "sdpa"
    logp_sdpa_bf16, ent_sdpa_bf16 = _run(m, input_ids, attention_mask, num_actions)
    del m
    torch.cuda.empty_cache()

    # --- sdpa@fp32 ----------------------------------------------------------
    m = _build("sdpa", bf16=False)
    logp_sdpa_fp32, ent_sdpa_fp32 = _run(m, input_ids, attention_mask, num_actions)
    del m
    torch.cuda.empty_cache()

    # --- eager@fp32 (reference attention math) ------------------------------
    m = _build("auto", bf16=False)  # auto + use_flash=False => eager
    assert m.attn_implementation == "eager"
    logp_eager_fp32, ent_eager_fp32 = _run(m, input_ids, attention_mask, num_actions)

    d_xkernel_logp = _d(logp_flash_bf16, logp_sdpa_bf16)
    d_xkernel_ent = _d(ent_flash_bf16, ent_sdpa_bf16)
    d_prec_logp = _d(logp_sdpa_bf16, logp_sdpa_fp32)
    d_correct_logp = _d(logp_sdpa_fp32, logp_eager_fp32)
    d_correct_ent = _d(ent_sdpa_fp32, ent_eager_fp32)
    print("\n[Stage2 parity] action-position diffs:")
    print(f"  (B) flash@bf16 vs sdpa@bf16 : logp={d_xkernel_logp:.6e}  ent={d_xkernel_ent:.6e}")
    print(f"      sdpa@bf16  vs sdpa@fp32 : logp={d_prec_logp:.6e}  (same-kernel bf16 floor)")
    print(f"  (A) sdpa@fp32  vs eager@fp32: logp={d_correct_logp:.6e}  ent={d_correct_ent:.6e}")

    # (A) Rigorous correctness: sdpa == eager (reference math) at fp32.
    assert torch.allclose(logp_sdpa_fp32, logp_eager_fp32, atol=FP32_ATOL, rtol=RTOL), (
        f"sdpa(fp32) logprobs disagree with eager(fp32) reference: {d_correct_logp:.6e} > {FP32_ATOL} "
        "-> sdpa math is WRONG (not a precision issue)"
    )
    assert torch.allclose(ent_sdpa_fp32, ent_eager_fp32, atol=FP32_ATOL, rtol=RTOL), (
        f"sdpa(fp32) entropy disagrees with eager(fp32): {d_correct_ent:.6e} > {FP32_ATOL}"
    )

    # (B) Training-signal: bf16 cross-kernel diff stays at the bf16 floor (the
    # same-kernel bf16-vs-fp32 spread is itself >= this; a wiring bug would not be).
    assert torch.allclose(logp_flash_bf16, logp_sdpa_bf16, atol=BF16_LOGP_ATOL, rtol=RTOL), (
        f"flash vs sdpa bf16 logprobs diverged beyond the bf16 floor: {d_xkernel_logp:.6e} > "
        f"{BF16_LOGP_ATOL} (same-kernel bf16 floor was {d_prec_logp:.6e})"
    )
    assert torch.allclose(ent_flash_bf16, ent_sdpa_bf16, atol=BF16_ENT_ATOL, rtol=RTOL), (
        f"flash vs sdpa bf16 entropy diverged beyond the bf16 floor: {d_xkernel_ent:.6e} > {BF16_ENT_ATOL}"
    )
