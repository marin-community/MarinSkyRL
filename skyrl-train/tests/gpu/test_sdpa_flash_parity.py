"""Stage 2 (FSDP2 CP) 1-GPU forward parity: SDPA vs flash_attention_2.

The load-bearing Stage-2 gate: pivoting the FSDP2 model's attention backend to
SDPA must preserve the training signal. We run `HFModelWrapper.forward` over the
SAME dense [B, S] batch under flash_attention_2 and under sdpa, and assert that
per-token logprobs and entropy match within bf16 tolerance.

Run on a 1-GPU srun inside the torch-2.11 SIF (has flash-attn 2.6.3, so BOTH
arms work):
    srun --reservation reformo --gres=gpu:1 ... \
        apptainer exec --nv <sif> python -m pytest -s \
            -p no:cacheprovider tests/gpu/test_sdpa_flash_parity.py
"""

import pytest
import torch
from transformers import AutoTokenizer

from skyrl_train.model_wrapper import HFModelWrapper

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# bf16 numeric tolerance (matches the tol used by other parity tests in this repo).
ATOL = 2e-2
RTOL = 0.0


def _dense_batch(tokenizer):
    """A small dense [B, S] batch with left-padding and a clear response span."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    # Two sequences, left-padded to the same length; num_actions response tokens.
    seq_a = [pad] * 2 + [785, 374, 264, 1273, 315, 279, 1849, eos]
    seq_b = [pad] * 1 + [12091, 1879, 11, 419, 374, 264, 2588, 1273, eos]
    width = max(len(seq_a), len(seq_b))
    seq_a = [pad] * (width - len(seq_a)) + seq_a
    seq_b = [pad] * (width - len(seq_b)) + seq_b
    input_ids = torch.tensor([seq_a, seq_b], dtype=torch.long)
    attention_mask = (input_ids != pad).to(torch.long)
    # Guard: at least one real token at the left edge stays masked even if pad==eos.
    num_actions = 4
    return input_ids, attention_mask, num_actions


def _build(attn_backend):
    model = HFModelWrapper(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=False,  # overridden by attn_backend
        bf16=True,
        sequence_parallel_size=1,
        use_sample_packing=False,
        attn_backend=attn_backend,
        context_parallel_size=1,
    )
    model.model.eval()
    model.model.to("cuda")
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_sdpa_matches_flash_forward():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    input_ids, attention_mask, num_actions = _dense_batch(tokenizer)
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    model_flash = _build("flash_attention_2")
    assert model_flash.attn_implementation == "flash_attention_2"
    with torch.no_grad():
        logp_flash, out_flash = model_flash(
            input_ids, num_actions, attention_mask, compute_entropy=True, return_output=True
        )
    del model_flash
    torch.cuda.empty_cache()

    model_sdpa = _build("sdpa")
    assert model_sdpa.attn_implementation == "sdpa"
    with torch.no_grad():
        logp_sdpa, out_sdpa = model_sdpa(
            input_ids, num_actions, attention_mask, compute_entropy=True, return_output=True
        )

    logp_flash = logp_flash.float()
    logp_sdpa = logp_sdpa.float()
    ent_flash = out_flash["entropy"].float()
    ent_sdpa = out_sdpa["entropy"].float()

    max_logp_diff = (logp_flash - logp_sdpa).abs().max().item()
    max_ent_diff = (ent_flash - ent_sdpa).abs().max().item()
    print(f"\n[Stage2 parity] max|logprob diff| = {max_logp_diff:.6e}  (atol={ATOL})")
    print(f"[Stage2 parity] max|entropy  diff| = {max_ent_diff:.6e}  (atol={ATOL})")

    assert torch.allclose(logp_flash, logp_sdpa, atol=ATOL, rtol=RTOL), (
        f"logprobs diverged: max|diff|={max_logp_diff:.6e} > atol={ATOL}"
    )
    assert torch.allclose(ent_flash, ent_sdpa, atol=ATOL, rtol=RTOL), (
        f"entropy diverged: max|diff|={max_ent_diff:.6e} > atol={ATOL}"
    )
