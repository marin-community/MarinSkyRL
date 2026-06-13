"""Stage 3 (vLLM DCP) — THE rollout correctness gate (G2): DCP=N == DCP=1.

Prove that turning vLLM Decode Context Parallel (DCP) ON does NOT change what the
rollout samples or returns. DCP shards *where the KV cache lives* (token dim,
internally), not *what is sampled or returned*. SkyRL's TIS path consumes
`resp.token_ids` + per-token `resp.logprobs` straight from the vLLM sampler output
(`inference_engines/vllm/vllm_engine.py:_postprocess_outputs`, ~:683-704). If DCP
perturbed token order/count/logprob-alignment, TIS importance ratios would silently
corrupt. So parity here is THE gate: no parity ⇒ the feature does not ship.

This test builds TWO vllm.LLM engines on the SAME model, SAME seed, enforce_eager:
  Engine A: (tp=TP, dcp=1)   Engine B: (tp=TP, dcp=2)
and extracts token_ids + per-token logprobs with the IDENTICAL extraction logic
SkyRL's `_postprocess_outputs` uses (the exact object the TIS path consumes), then:
  #1 GREEDY token-id identity (G2, make-or-break): B.token_ids == A.token_ids EXACTLY.
  #2 LOGPROB allclose (G2): per-token logprobs B ~= A within fp tol (DCP's
     AllGather+ReduceScatter changes the fp reduction order, so logprobs match
     within tol, not bit-exactly; argmax is robust to sub-ulp jitter so #1 stays
     exact).
  #3 SAMPLED (seeded) token-id parity: fixed per-request seed, temperature>0.

CRITICAL GEOMETRY (vLLM hard asserts, GQA/non-MLA — vllm/config/model.py:1161-1185):
  dcp>1 requires: tp % dcp == 0; tp > num_kv_heads; dcp <= tp // num_kv_heads;
  AND (num_attention_heads // num_kv_heads) % dcp == 0.
  Qwen2.5-1.5B-Instruct: heads=12, kv_heads=2 -> q_per_kv=6.
    tp=4: 4%2==0; 4>2; dcp 2 <= 4//2=2; 6%2==0  => dcp=2 VALID at tp=4 (1 node, 4 GPU).
  (Qwen3-0.6B has kv_heads=8 -> needs tp>8; Qwen2.5-0.5B has heads=14/kv=2 ->
   q_per_kv=7 ODD -> 7%2!=0 -> dcp=2 INVALID. 1.5B is the smallest valid pick.)

Run (4-GPU single node, NOT torchrun — vLLM owns its own TP workers):
    apptainer exec --nv <sif> python tests/gpu/test_dcp_rollout_parity.py
"""

import os
import sys

# DCP's AllGather+ReduceScatter reorders the fp reduction vs dcp=1; logprobs match
# within tol, NOT bit-exactly. Greedy argmax (#1) must still be EXACT (argmax robust
# to sub-ulp logit jitter). atol 1e-2 absorbs the bf16 reduction-order residual on
# the post-softmax log-probabilities; do NOT loosen to pass — if #1 fails or #2
# exceeds this, that is a real correctness concern (NO-GO), not a tolerance to widen.
LOGPROB_ATOL = 1e-2

MODEL_NAME = os.environ.get("DCP_PARITY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TP = int(os.environ.get("DCP_PARITY_TP", "4"))
DCP = int(os.environ.get("DCP_PARITY_DCP", "2"))
MAX_TOKENS = int(os.environ.get("DCP_PARITY_MAX_TOKENS", "48"))
SEED = 1234

# Fixed prompts (token-id space; SkyRL's vLLM path only accepts prompt_token_ids).
# Built from the tokenizer below so they are valid for the chosen model.
PROMPT_TEXTS = [
    "The capital of France is",
    "In a galaxy far, far away, there lived a",
    "To compute the factorial of a number in Python, you can write",
    "The three primary colors are red, blue, and",
    "Once upon a time, in a small village by the sea,",
    "The chemical symbol for water is",
]


def _extract(outputs):
    """Replicate skyrl_train.inference_engines.vllm.vllm_engine._postprocess_outputs
    token-id + per-token-logprob extraction EXACTLY (the object the TIS path reads).
    Returns (response_ids: List[List[int]], response_logprobs: List[List[float]])."""
    response_ids = []
    response_logprobs = []
    for output in outputs:
        assert len(output.outputs) == 1, "expected n=1 per prompt"
        resp = output.outputs[0]
        response_ids.append(list(resp.token_ids))
        _logprobs = None
        if resp.logprobs:
            _logprobs = []
            for i, token_logprobs in enumerate(resp.logprobs):
                token_id = resp.token_ids[i]
                _logprobs.append(token_logprobs[token_id].logprob)
        response_logprobs.append(_logprobs)
    return response_ids, response_logprobs


def _build_engine(dcp):
    import vllm

    kwargs = dict(
        model=MODEL_NAME,
        tensor_parallel_size=TP,
        enforce_eager=True,         # determinism: no cudagraph capture differences
        seed=SEED,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,  # two engines may coexist; keep each modest
        max_model_len=2048,
        disable_log_stats=True,
        enable_prefix_caching=False,  # avoid cross-request KV reuse perturbing parity
    )
    # G1 contract: dcp kwarg ABSENT when ==1 (byte-identical to today); present when >1.
    if dcp > 1:
        kwargs["decode_context_parallel_size"] = dcp
    return vllm.LLM(**kwargs)


def _run_engine(llm, prompt_token_ids, temperature):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        logprobs=0,        # request the chosen-token logprob (the TIS contract field)
        seed=SEED,         # fixed sampler RNG for the seeded-sampled case
    )
    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
        sampling_params=sp,
    )
    # vllm.generate may reorder by request id; sort back to input order via request_id.
    outputs = sorted(outputs, key=lambda o: int(o.request_id))
    return _extract(outputs)


def main():
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available — Stage 3 DCP rollout parity gate DEFERRED.")
        return 0
    ngpu = torch.cuda.device_count()
    assert ngpu >= TP, f"Stage 3 DCP parity needs >= {TP} GPUs (tp={TP}); got {ngpu}"

    import vllm
    from transformers import AutoTokenizer

    print(f"[Stage3-DCP] vllm={vllm.__version__} model={MODEL_NAME} tp={TP} dcp={DCP} "
          f"max_tokens={MAX_TOKENS} seed={SEED} gpus={ngpu}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_token_ids = [tok(t, add_special_tokens=True)["input_ids"] for t in PROMPT_TEXTS]
    print(f"[Stage3-DCP] {len(prompt_token_ids)} prompts; lens={[len(p) for p in prompt_token_ids]}")

    all_ok = True

    # ---- Engine A: dcp=1 (the reference; G1 kwarg-absent path) ----
    print("\n[Stage3-DCP] building Engine A (dcp=1) ...")
    llm_a = _build_engine(dcp=1)
    ids_a_greedy, lp_a_greedy = _run_engine(llm_a, prompt_token_ids, temperature=0.0)
    ids_a_samp, _ = _run_engine(llm_a, prompt_token_ids, temperature=0.8)
    # free engine A before building B (each at 0.45 util; sequential is safest).
    del llm_a
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import ray  # vLLM may have spun a ray-internal executor; ignore if absent
    except Exception:
        ray = None

    # ---- Engine B: dcp=2 ----
    print(f"\n[Stage3-DCP] building Engine B (dcp={DCP}) ...")
    llm_b = _build_engine(dcp=DCP)
    ids_b_greedy, lp_b_greedy = _run_engine(llm_b, prompt_token_ids, temperature=0.0)
    ids_b_samp, _ = _run_engine(llm_b, prompt_token_ids, temperature=0.8)
    del llm_b

    # =====================================================================
    # #1 GREEDY token-id identity (G2 — make-or-break, must be EXACT)
    # =====================================================================
    n = len(prompt_token_ids)
    greedy_match = 0
    greedy_tokens_total = 0
    greedy_tokens_match = 0
    first_divergence = None
    for i in range(n):
        a, b = ids_a_greedy[i], ids_b_greedy[i]
        exact = (a == b)
        greedy_match += int(exact)
        # token-level match % over the common prefix length
        m = min(len(a), len(b))
        greedy_tokens_total += max(len(a), len(b))
        for j in range(m):
            if a[j] == b[j]:
                greedy_tokens_match += 1
            elif first_divergence is None:
                first_divergence = (i, j, a[j], b[j])
        if not exact and first_divergence is None and len(a) != len(b):
            first_divergence = (i, m, len(a), len(b))
    greedy_seq_pct = 100.0 * greedy_match / n
    greedy_tok_pct = 100.0 * greedy_tokens_match / max(greedy_tokens_total, 1)
    print(f"\n[Stage3-DCP] #1 GREEDY token-id identity: "
          f"{greedy_match}/{n} sequences exact ({greedy_seq_pct:.1f}%); "
          f"token-level {greedy_tokens_match}/{greedy_tokens_total} ({greedy_tok_pct:.2f}%)")
    if first_divergence is not None:
        print(f"[Stage3-DCP] #1 FIRST DIVERGENCE: prompt={first_divergence[0]} pos={first_divergence[1]} "
              f"A={first_divergence[2]} B={first_divergence[3]}")
    ok_greedy = (greedy_match == n)
    all_ok &= ok_greedy

    # =====================================================================
    # #2 LOGPROB allclose (G2 — within fp tol)
    # =====================================================================
    max_abs_dlp = 0.0
    mean_abs_dlp = 0.0
    cnt = 0
    worst = None
    for i in range(n):
        la, lb = lp_a_greedy[i], lp_b_greedy[i]
        assert la is not None and lb is not None, f"prompt {i}: missing logprobs (TIS contract field empty)"
        # only compare positions where greedy token_ids agree (else logprobs index
        # different tokens — a #1 failure already flagged above).
        m = min(len(la), len(lb))
        for j in range(m):
            if ids_a_greedy[i][j] != ids_b_greedy[i][j]:
                break
            d = abs(la[j] - lb[j])
            mean_abs_dlp += d
            cnt += 1
            if d > max_abs_dlp:
                max_abs_dlp = d
                worst = (i, j, la[j], lb[j])
    mean_abs_dlp = mean_abs_dlp / max(cnt, 1)
    print(f"\n[Stage3-DCP] #2 LOGPROB parity (over greedy-agreeing tokens, n={cnt}): "
          f"max|Δ|={max_abs_dlp:.3e}  mean|Δ|={mean_abs_dlp:.3e}  (atol={LOGPROB_ATOL})")
    if worst is not None:
        print(f"[Stage3-DCP] #2 worst Δ at prompt={worst[0]} pos={worst[1]} A={worst[2]:.5f} B={worst[3]:.5f}")
    ok_logprob = (max_abs_dlp <= LOGPROB_ATOL)
    all_ok &= ok_logprob

    # =====================================================================
    # #3 SAMPLED (seeded) token-id parity — diagnostic (greedy #1 is the hard gate)
    # =====================================================================
    samp_match = sum(int(ids_a_samp[i] == ids_b_samp[i]) for i in range(n))
    samp_pct = 100.0 * samp_match / n
    print(f"\n[Stage3-DCP] #3 SAMPLED (seed={SEED}, temp=0.8) token-id seq match: "
          f"{samp_match}/{n} ({samp_pct:.1f}%)  [diagnostic; greedy #1 is the hard gate]")

    # =====================================================================
    # Verdict
    # =====================================================================
    print("\n" + "=" * 72)
    print(f"[Stage3-DCP] VERDICT: greedy_exact={ok_greedy} ({greedy_seq_pct:.1f}% seq) | "
          f"logprob_allclose={ok_logprob} (max|Δ|={max_abs_dlp:.3e} <= {LOGPROB_ATOL}) | "
          f"sampled_match={samp_pct:.1f}%")
    print(f"[Stage3-DCP] G2 GATE (greedy token-id identity + logprob allclose): "
          f"{'PASS' if all_ok else 'NO-GO'}")
    print("=" * 72)
    assert all_ok, (
        "Stage-3 DCP rollout parity G2 gate FAILED — DCP changed the rollout output "
        "(greedy token_ids diverged or logprobs exceeded tol). This is a CORRECTNESS "
        "concern, not a tolerance to widen. See per-test output above."
    )
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
