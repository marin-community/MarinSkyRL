"""Stage 5 (FSDP2 CP) — THE correctness gate: per-CP-rank token-offset / unshard.

This stage replaces Stage-4's temporary immediate logit-unshard with the
loss-aligned per-token unshard: the per-token logprobs / entropy are computed on
the sequence-sharded `[B, S/cp, V]` logits (against the co-sharded
`sequences_rolled` labels) and then `context_parallel_unshard`-ed back to
natural-order `[B, S]` BEFORE the response slice / PPO loss. The make-or-break
test is **G3**: cp=2 action_log_probs / entropy / ref-KL == cp=1 within bf16 tol,
in identical token order.

Tests (all run under torchrun, 2 ranks, cp=2):
  1. G3 round-trip parity   — cp=2 == cp=1 (action_log_probs, entropy, ref-KL).
  2. Zigzag oracle          — context_parallel_unshard order == slime oracle (CPU).
  3. loss_mask alignment    — action_log_probs.shape == loss_mask.shape, same tokens.
  4. Loss-value parity      — full ppo_policy_loss, ALL FOUR reduce_loss modes.
  5. Pad-edge               — seq_len not divisible by 2*cp -> padded, parity holds.

IMPORTANT — the SIF bakes SkyRL at /opt/SkyRL, which shadows a worktree clone.
Run with PYTHONPATH pointing at the worktree's skyrl-train so the model_wrapper /
cp_utils UNDER TEST (not the baked ones) are imported. Set
LIBRARY_PATH=/.singularity.d/libs for Triton JIT gcc.

    # 2-GPU cp=2 (torchrun main):
    srun --account=reformo --reservation=reformo --gres=gpu:2 ... \
      apptainer exec --nv --env PYTHONPATH=<worktree>/skyrl-train \
        --env LIBRARY_PATH=/.singularity.d/libs <sif> \
        torchrun --nproc-per-node=2 tests/gpu/test_cp_logprob_parity.py
"""

import os

import torch

import skyrl_train.model_wrapper as _mw
import skyrl_train.distributed.cp_utils as _cp
from skyrl_train.model_wrapper import HFModelWrapper

# Import the oracle by file (tests/gpu is not a package on the import path under
# the torchrun launch; add the test dir to sys.path defensively).
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _zigzag_oracle  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-0.6B"

# bf16 ring-attention reduction floor. The G3 gate is atol=2e-2 per the spec — do
# NOT loosen this to pass; if it fails the feature is not built.
G3_ATOL = 2e-2


def _assert_under_test():
    assert "/opt/SkyRL/" not in _mw.__file__, f"model_wrapper imported from baked SIF: {_mw.__file__}"
    assert "/opt/SkyRL/" not in _cp.__file__, f"cp_utils imported from baked SIF: {_cp.__file__}"
    assert hasattr(_cp, "context_parallel_unshard"), "cp_utils.context_parallel_unshard missing — wrong/old module"


def _dense_batch(divisible: bool):
    """A small dense [B, S] batch with left-padding and a clear response span.

    `divisible=True`  -> seq_len chosen divisible by 2*cp (=4), no G4 pad.
    `divisible=False` -> seq_len NOT divisible (pad-edge test #5).
    """
    pad = 151643  # Qwen3 <|endoftext|> (pad)
    eos = 151645  # <|im_end|>
    if divisible:
        # width 12 -> 12 % 4 == 0 (no pad)
        seq_a = [pad] * 3 + [785, 374, 264, 1273, 315, 279, 1849, 11, eos]
        seq_b = [pad] * 2 + [12091, 1879, 11, 419, 374, 264, 2588, 1273, 13, eos]
    else:
        # width 10 -> 10 % 4 == 2 (pad to 12)
        seq_a = [pad] * 2 + [785, 374, 264, 1273, 315, 279, 1849, eos]
        seq_b = [pad] * 1 + [12091, 1879, 11, 419, 374, 264, 2588, 1273, eos]
    width = max(len(seq_a), len(seq_b))
    seq_a = [pad] * (width - len(seq_a)) + seq_a
    seq_b = [pad] * (width - len(seq_b)) + seq_b
    input_ids = torch.tensor([seq_a, seq_b], dtype=torch.long)
    attention_mask = (input_ids != pad).to(torch.long)
    num_actions = 4
    return input_ids, attention_mask, num_actions


def _build(context_parallel_size=1, cp_mesh=None, cp_rotate_method="allgather"):
    model = HFModelWrapper(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=False,
        bf16=True,
        sequence_parallel_size=1,
        use_sample_packing=False,
        attn_backend="sdpa",
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
# Test #2 — zigzag oracle (pure CPU, no torch-CP). Validates the oracle math and,
# below in test #1, the live context_parallel_unshard token order against it.
# ----------------------------------------------------------------------------
def _test_zigzag_oracle_selfcheck(rank):
    ok = True
    for cp_size in (2, 4):
        for seq_len in (8, 12, 16, 24, 2 * cp_size * 5):
            if seq_len % (2 * cp_size) != 0:
                continue
            natural, expected = _zigzag_oracle.build_sharded_then_unshard_check(cp_size, seq_len)
            if not torch.equal(natural, expected):
                ok = False
                if rank == 0:
                    print(f"[Stage5 #2] oracle self-check FAIL cp={cp_size} S={seq_len}")
    if rank == 0:
        print(f"[Stage5 #2] zigzag oracle self-check: {'PASS' if ok else 'FAIL'}")
    return ok


def _test_unshard_matches_oracle(cp_mesh, cp_size, rank):
    """Drive context_parallel_unshard on a tagged identity tensor and assert it
    returns natural order, matching the ported slime zigzag oracle."""
    from skyrl_train.distributed.cp_utils import cp_context, context_parallel_unshard

    ok = True
    for seq_len in (8, 12, 16):
        if seq_len % (2 * cp_size) != 0:
            continue
        # Build a [1, S] tensor whose value at natural index i is i. Shard it via
        # context_parallel (so torch's load balancer slices it), then unshard.
        full = torch.arange(seq_len, dtype=torch.float32, device="cuda").unsqueeze(0)
        buf = full.clone()
        with cp_context(cp_mesh, "allgather", buffers=[buf], seq_dims=[1], no_restore=set()):
            # inside the context `buf` is now this rank's local shard [1, S/cp]
            local = buf.clone()
            unsharded = context_parallel_unshard(cp_mesh, [local], [1])[0]
        # After unshard the values must be 0,1,2,...,S-1 (natural order).
        expected = torch.arange(seq_len, dtype=torch.float32, device="cuda").unsqueeze(0)
        match = torch.equal(unsharded.round().long(), expected.long())
        # Cross-check the per-rank local shard matches the oracle's index list.
        oracle_idx = _zigzag_oracle.cp_shard_indices(rank, cp_size, seq_len)
        oracle_vals = torch.tensor(oracle_idx, dtype=torch.float32, device="cuda")
        shard_match = torch.equal(local.squeeze(0).round().long(), oracle_vals.long())
        if not (match and shard_match):
            ok = False
            if rank == 0:
                print(
                    f"[Stage5 #2] live unshard vs oracle FAIL S={seq_len}: "
                    f"unshard_natural={match} shard_match={shard_match}\n"
                    f"  local(rank{rank})={local.squeeze(0).round().long().tolist()}\n"
                    f"  oracle(rank{rank})={oracle_idx}\n"
                    f"  unsharded={unsharded.round().long().tolist()}"
                )
    if rank == 0:
        print(f"[Stage5 #2] live context_parallel_unshard vs oracle: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# Test #4 — loss-value parity helper.
# ----------------------------------------------------------------------------
def _ppo_loss_all_modes(action_log_probs, loss_mask):
    """Run ppo_policy_loss for all four reduce_loss modes with fixed synthetic
    old_log_probs / advantages so cp=1 and cp=2 are compared apples-to-apples."""
    from omegaconf import OmegaConf
    from skyrl_train.utils.ppo_utils import ppo_policy_loss

    B, A = action_log_probs.shape
    torch.manual_seed(0)
    # deterministic synthetic inputs (same on every rank / both cp settings)
    old_log_probs = action_log_probs.detach() + 0.01
    advantages = torch.linspace(-1.0, 1.0, B * A, device=action_log_probs.device).reshape(B, A)
    max_seq_len = 64
    results = {}
    for mode in ("token_mean", "sequence_mean", "seq_mean_token_sum_norm", "seq_mean_token_sum_norm_global"):
        cfg = OmegaConf.create(
            {
                "policy_loss_type": "regular",
                "loss_reduction": mode,
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "clip_ratio_c": 3.0,
                "use_tis": False,
                "tis_imp_ratio_cap": 2.0,
                "max_seq_len": max_seq_len,
                "global_loss_denom": float(B * max_seq_len),
            }
        )
        loss, _ = ppo_policy_loss(action_log_probs, old_log_probs, advantages, cfg, loss_mask=loss_mask)
        results[mode] = loss.float().item()
    return results


def _ref_kl(policy_logp, ref_logp, loss_mask):
    from skyrl_train.utils.ppo_utils import compute_approx_kl

    return compute_approx_kl(policy_logp, ref_logp, loss_mask=loss_mask, kl_estimator_type="k3")


def main():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not torch.cuda.is_available():
        print("CUDA not available — Stage 5 CP parity gate DEFERRED.")
        return

    _assert_under_test()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cp_size = world_size
    assert world_size == 2, f"Stage 5 CP parity gate needs 2 ranks (cp=2); got {world_size}"

    mesh = init_device_mesh("cuda", (cp_size,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    all_ok = True

    # --- #2 zigzag oracle (CPU self-check + live unshard vs oracle) ---
    all_ok &= _test_zigzag_oracle_selfcheck(rank)
    all_ok &= _test_unshard_matches_oracle(cp_mesh, cp_size, rank)
    dist.barrier()

    # Build the cp=1 reference ONCE (deterministic, identical on every rank).
    m_cp1 = _build(context_parallel_size=1, cp_mesh=None)
    # Build the cp=2 model (same checkpoint weights).
    m_cp2 = _build(context_parallel_size=cp_size, cp_mesh=cp_mesh, cp_rotate_method="allgather")

    for tag, divisible in (("divisible", True), ("pad-edge", False)):
        input_ids, attention_mask, num_actions = _dense_batch(divisible)
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        S = input_ids.size(1)
        if rank == 0:
            print(f"\n[Stage5] === case={tag} seq_len={S} (S % 2cp = {S % (2 * cp_size)}) ===")

        logp_cp1, ent_cp1 = _run(m_cp1, input_ids, attention_mask, num_actions)
        logp_cp2, ent_cp2 = _run(m_cp2, input_ids, attention_mask, num_actions)

        # --- #1 G3 round-trip parity: action_log_probs + entropy ---
        d_logp = (logp_cp1 - logp_cp2).abs().max().item()
        # entropy returned is over the FULL [B, S]; slice to the response span to
        # compare apples-to-apples (action span = last num_actions).
        ent_cp1_a = ent_cp1[:, -num_actions:]
        ent_cp2_a = ent_cp2[:, -num_actions:]
        d_ent = (ent_cp1_a - ent_cp2_a).abs().max().item()

        # --- ref-KL parity: use cp1 logp as ref vs cp2 logp as policy. The KL of
        # cp2-vs-cp1 should also match the (trivially zero) cp1-vs-cp1 KL within
        # tol; more usefully we compare KL(cp2_policy, fixed_ref) to
        # KL(cp1_policy, fixed_ref) where fixed_ref is a deterministic shift. ---
        loss_mask = torch.ones_like(logp_cp1)
        fixed_ref = logp_cp1.detach() - 0.05
        kl_cp1 = _ref_kl(logp_cp1, fixed_ref, loss_mask)
        kl_cp2 = _ref_kl(logp_cp2, fixed_ref, loss_mask)
        d_kl = (kl_cp1 - kl_cp2).abs().max().item()

        if rank == 0:
            print(f"[Stage5 #1] G3 parity: d_action_log_probs={d_logp:.3e}  d_entropy={d_ent:.3e}  d_refKL={d_kl:.3e}")
            print(f"[Stage5 #1] (atol={G3_ATOL})")
        ok1 = (d_logp <= G3_ATOL) and (d_ent <= G3_ATOL) and (d_kl <= G3_ATOL)
        # identical token order: shapes must match exactly (no off-by-one)
        ok_order = logp_cp1.shape == logp_cp2.shape == (input_ids.size(0), num_actions)
        if rank == 0:
            print(f"[Stage5 #1] within tol: {ok1}  shape/order match: {ok_order}  shape={tuple(logp_cp2.shape)}")
        all_ok &= ok1 and ok_order

        # --- #3 loss_mask alignment ---
        ok3 = logp_cp2.shape == loss_mask.shape == logp_cp1.shape
        if rank == 0:
            print(f"[Stage5 #3] loss_mask alignment (shape match): {ok3}")
        all_ok &= ok3

        # --- #4 loss-value parity across all four reduce_loss modes ---
        losses_cp1 = _ppo_loss_all_modes(logp_cp1, loss_mask)
        losses_cp2 = _ppo_loss_all_modes(logp_cp2, loss_mask)
        ok4 = True
        for mode in losses_cp1:
            dloss = abs(losses_cp1[mode] - losses_cp2[mode])
            within = dloss <= G3_ATOL
            ok4 &= within
            if rank == 0:
                print(
                    f"[Stage5 #4] reduce_loss={mode:32s} cp1={losses_cp1[mode]:+.5f} "
                    f"cp2={losses_cp2[mode]:+.5f} |d|={dloss:.3e} {'OK' if within else 'FAIL'}"
                )
        all_ok &= ok4

        dist.barrier()

    if rank == 0:
        print(f"\n[Stage5] OVERALL: {'ALL PASS' if all_ok else 'FAIL'}")
    # make the gate fail loudly on any rank
    assert all_ok, "Stage-5 CP parity gate FAILED (see per-test output)"
    if rank == 0:
        print("ALL PASS")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
