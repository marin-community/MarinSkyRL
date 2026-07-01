from typing import List, Tuple, Optional
import torch
from loguru import logger
from transformers import AutoTokenizer
from jaxtyping import Float, Integer


def _routed_experts_dtype_for_num_experts(num_experts: Optional[int]) -> Optional[torch.dtype]:
    """Pick the narrowest integer dtype that can hold ANY valid expert id for a
    model with ``num_experts`` experts, DETERMINISTICALLY (max possible id =
    ``num_experts - 1``), independent of the per-batch observed max.

    This is load-bearing: the dtype must NOT depend on the data, or two batches
    / ranks whose observed max straddles a dtype boundary (e.g. Qwen3-Next with
    512 experts: one batch max=200 -> uint8, another max=300 -> int16) would
    pick DIFFERENT dtypes for the same field, size-mismatching a later cross-rank
    collective on this tensor -> NCCL hang. Keying on ``num_experts`` makes every
    rank/batch agree.

      * num_experts <= 256      -> uint8  (max id <= 255; Qwen3-Coder 128 -> uint8, identical to the prior per-batch pick)
      * num_experts <= 32768    -> int16  (max id <= 32767; Qwen3-Next 512 -> int16, deterministic)
      * otherwise               -> int64  (defensive; no shipped MoE model exceeds int16 range)

    Returns None when ``num_experts`` is None/unknown, signalling the caller to
    fall back to the (non-deterministic) per-batch-max pick.
    """
    if num_experts is None or num_experts <= 0:
        return None
    if num_experts <= (torch.iinfo(torch.uint8).max + 1):
        return torch.uint8
    if num_experts <= (torch.iinfo(torch.int16).max + 1):
        return torch.int16
    return torch.int64


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: Optional[List[torch.Tensor]],
    loss_masks: List[List[int]],
):
    assert (
        len(prompts) == len(responses) and len(prompts) > 0
    ), "prompts and responses must have the same length and length must be greater than 0, got {} and {}".format(
        len(prompts), len(responses)
    )

    if rewards is not None:
        assert len(rewards) == len(prompts), "rewards must have the same length as prompts, got {} and {}".format(
            len(rewards), len(prompts)
        )
    assert len(loss_masks) == len(prompts), "loss_masks must have the same length as prompt, got {} and {}".format(
        len(loss_masks), len(prompts)
    )

    # Element-type validation. torch.tensor(sequences) raises the cryptic
    # `ValueError: too many dimensions 'str'` if any prompt/response token-id
    # list contains a non-int (e.g. a stringified token leaking from a
    # malformed rollout trajectory). Surface exactly which sample + field +
    # offending element is corrupt instead, so the bad trajectory is
    # actionable rather than a bare ValueError at the tensor build. Valid
    # int-only inputs (the normal path, incl. a3) pass through unchanged.
    def _first_bad_token(seq):
        for tok in seq:
            if not isinstance(tok, (int, bool)):
                return tok
        return None

    for field_name, seqs in (("prompt", prompts), ("response", responses)):
        for idx, seq in enumerate(seqs):
            bad = _first_bad_token(seq)
            if bad is not None:
                raise ValueError(
                    "{field} token-id list at sample index {idx} contains a non-int element "
                    "{bad!r} (type {tname}); expected a flat list of token ids. This corrupts "
                    "torch.tensor() collation (the bare 'too many dimensions \\'str\\'' error). "
                    "The offending trajectory's {field}_ids must be tokenized ints.".format(
                        field=field_name, idx=idx, bad=bad, tname=type(bad).__name__
                    )
                )


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: List[List[float]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[List[float]]] = None,
    routed_experts: Optional[List[List[List[List[int]]]]] = None,
    token_level_shaping: Optional[List[List[float]]] = None,
    response_span_tags: Optional[List[List[int]]] = None,
    num_experts: Optional[int] = None,
) -> Tuple[
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Optional[Float[torch.Tensor, "batch response_len"]],
    Optional["torch.Tensor"],
    Optional[Float[torch.Tensor, "batch response_len"]],
    Optional[Integer[torch.Tensor, "batch response_len"]],
]:
    """
    Convert prompts and responses to batch tensors for training.

    This function concatenates all prompts and responses to the following format:

    | [PAD] [PAD] token token token | token token [PAD] [PAD] |
    | token token token token token | token token [PAD] [PAD] |
    | [PAD] [PAD] [PAD] token token | token token token [PAD] |
    |<---------- prompt ----------->|<-------- answer ------->|

    Assumes that the responses already contain an eos token at index -1.

    Args:
        tokenizer: Model tokenizer
        prompts: List of tokenized prompts
        responses: List of tokenized responses
        rewards: List of rewards for each response
        loss_masks: List of loss masks for each response
        logprobs: List of rollout log probs for each response

    Returns:
        sequences: Full trajectories (padded and concatenated prompts and responses). Size: (batch, seq_len).
        attention_mask: Attention mask for the model. Size: (batch, seq_len)
        action_mask: Response mask for the model. Size: (batch, response_len)
        rewards: Rewards for each output. Size: (batch, response_len)
        loss_masks: Loss masks for each output. Size: (batch, response_len)
    """
    _verify_inputs(prompts, responses, rewards, loss_masks)

    max_input_len, max_output_len = 0, 0
    prompt_token_lens, response_token_lens = [], []
    inputs_token_ids, outputs_token_ids = [], []
    for prompt, response in zip(prompts, responses):

        inputs_token_ids.append(prompt)
        outputs_token_ids.append(response)

        prompt_token_len = len(prompt)
        response_token_len = len(response)
        prompt_token_lens.append(prompt_token_len)
        response_token_lens.append(response_token_len)

        max_input_len = max(max_input_len, prompt_token_len)
        max_output_len = max(max_output_len, response_token_len)

    pad_token_id = tokenizer.pad_token_id
    sequences = []
    attention_masks = []
    action_masks = []
    for i, prompt in enumerate(prompts):
        # left padding input
        input_len = prompt_token_lens[i]
        input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])
        input_attention_mask = [0] * (max_input_len - input_len) + [1] * input_len

        # right padding output
        output_len = response_token_lens[i]
        output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)
        output_attention_mask = [1] * output_len + [0] * (max_output_len - output_len)

        # concat input and output
        sequences.append(input_ids + output_ids)
        attention_masks.append(input_attention_mask + output_attention_mask)
        action_masks.append(output_attention_mask)

    sequences = torch.tensor(sequences)
    attention_mask = torch.tensor(attention_masks, dtype=torch.int64)
    action_mask = torch.tensor(action_masks, dtype=torch.int64)

    # initialize ret loss masks to be the same as action mask
    ret_loss_masks = torch.zeros_like(action_mask, dtype=torch.float)
    for i, loss_mask in enumerate(loss_masks):
        ret_loss_masks[i, : len(loss_mask)] = torch.tensor(loss_mask)

    # do the same for custom rewards
    ret_rewards = torch.zeros_like(action_mask, dtype=torch.float)
    for i, custom_reward in enumerate(rewards):
        if isinstance(custom_reward, list):
            custom_reward = torch.tensor(custom_reward)
        ret_rewards[i, : len(custom_reward)] = custom_reward

    logprobs_tensor = None
    if logprobs:
        max_output_len = action_mask.size(1)
        padded_logprobs = [
            sample_logprobs + [0.0] * (max_output_len - len(sample_logprobs)) for sample_logprobs in logprobs
        ]
        logprobs_tensor = torch.tensor(padded_logprobs, dtype=torch.float)

    # MoE router-replay capture rail (Stage 1): right-pad routed_experts on the
    # response axis exactly like rollout_logprobs, but each per-token element is a
    # [L, K] expert-index vector. Result: [batch, response_len, L, K] int. Padding
    # rows are sentinel [L, K] (all zeros). 4-D is accepted by TensorBatch since
    # _check_consistency only validates dim-0.
    routed_experts_tensor = None
    if routed_experts:
        max_output_len = action_mask.size(1)
        # Infer the true [L, K] per-token row shape by scanning ALL samples for the
        # WIDEST real row, not just routed_experts[0][0]. Samples/rows that lack the
        # full routing (cross-sample sentinels emitted as a degenerate [1, 1], or
        # preempted/quant paths) would otherwise leave the L axis ragged ([48, K] vs
        # [1, 1]) and crash the dense torch.tensor() collation with
        # "expected sequence of length 1 at dim 2 (got 48)". We then NORMALIZE every
        # row to [L, K] so the tensorize is shape-uniform regardless of upstream raggedness.
        L, K = 1, 1
        for sample_re in routed_experts:
            for row in sample_re:
                if isinstance(row, (list, tuple)) and len(row) > L:
                    L = len(row)
                    inner = row[0] if L > 0 else None
                    K = len(inner) if isinstance(inner, (list, tuple)) and len(inner) > K else K
        sentinel_row = [[0] * K for _ in range(L)]

        def _normalize_row(row):
            # Coerce a per-token routed-experts row to exactly [L, K].
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                return [list(layer) for layer in sentinel_row]
            out = []
            for layer in row[:L]:
                if isinstance(layer, (list, tuple)):
                    layer = list(layer[:K]) + [0] * (K - len(layer))
                else:
                    layer = [int(layer)] + [0] * (K - 1)
                out.append(layer)
            # pad missing layers
            for _ in range(L - len(out)):
                out.append([0] * K)
            return out

        # Fast path: rows already uniform [L, K] (the production MoE path) skip
        # per-row normalization entirely so the byte-identical batch is preserved.
        def _row_is_canonical(row):
            return (
                isinstance(row, (list, tuple))
                and len(row) == L
                and all(isinstance(l, (list, tuple)) and len(l) == K for l in row)
            )

        padded_re = []
        for sample_re in routed_experts:
            sample_re = list(sample_re)
            normalized = [r if _row_is_canonical(r) else _normalize_row(r) for r in sample_re]
            pad_n = max_output_len - len(normalized)
            if pad_n > 0:
                normalized = normalized + [sentinel_row for _ in range(pad_n)]
            elif pad_n < 0:
                normalized = normalized[:max_output_len]
            padded_re.append(normalized)
        # Width-minimize the expert-id tensor (the R3 by-value forward-arg spill
        # fix, part 1). This tensor is [B, response_len, L, K] (48*8=384 ids/token
        # at Qwen3-Coder-30B-A3B) and at 131k it is multiple GB at int64 — the bulk
        # that, shipped by-value through every per-forward Ray task arg, spilled the
        # object store and wedged the 32-rank forward. Expert ids are small
        # non-negative ints, so pick the NARROWEST integer dtype that can hold ANY
        # valid id for this model. The choice is keyed on the model's num_experts
        # (max possible id = num_experts - 1), NOT the per-batch observed max, so it
        # is DETERMINISTIC across every rank and batch: a data-dependent pick lets
        # two batches whose observed max straddles a dtype boundary (e.g. Qwen3-Next
        # 512 experts: one batch max=200 -> uint8, another max=300 -> int16) diverge,
        # which size-mismatches a later cross-rank collective on this tensor -> NCCL
        # hang. Qwen3-Coder (128 experts) -> uint8 (max id 127 <= 255, identical to
        # the earlier per-batch pick); Qwen3-Next (512) -> int16 (deterministic). The
        # sentinel id (0) and every real id fit by construction. The training-side
        # consumer upcasts back to int64 (`model_wrapper._build_router_replay_targets`:
        # `.to(dtype=torch.long)`), so this is downstream-transparent — a pure
        # transport-size optimization. No torch.uint16 storage support across the
        # saved/pinned/transport path, so int16 (not uint16) is the mid tier.
        routed_experts_tensor = torch.tensor(padded_re, dtype=torch.long)
        _re_dtype = _routed_experts_dtype_for_num_experts(num_experts)
        if _re_dtype is None:
            # Fallback: num_experts unknown (non-MoE / unresolved config). Preserve
            # the ORIGINAL per-batch-max behavior so those cases are unaffected, but
            # warn once — this branch is NON-DETERMINISTIC across ranks/batches and
            # must not be hit on a real MoE-RL run.
            _max_expert_id = int(routed_experts_tensor.max().item()) if routed_experts_tensor.numel() else 0
            if _max_expert_id <= torch.iinfo(torch.uint8).max:
                _re_dtype = torch.uint8
            elif _max_expert_id <= torch.iinfo(torch.int16).max:
                _re_dtype = torch.int16
            else:
                _re_dtype = torch.int64
            logger.warning(
                "convert_prompts_responses_to_batch_tensors: num_experts is None; "
                "using the NON-DETERMINISTIC per-batch-max dtype pick for "
                "rollout_routed_experts (chose {}). This is safe for non-MoE / "
                "unknown-config cases but must NOT be hit on a MoE-RL run — thread "
                "the model's num_experts through to make the dtype rank-invariant.".format(_re_dtype)
            )
        routed_experts_tensor = routed_experts_tensor.to(_re_dtype)

    # Loop-behavior reward shaping (Stage B / F5 + F4): right-pad the per-token
    # shaping channel and span tags on the response axis exactly like rewards /
    # loss_mask. Both are gated upstream (only passed when
    # enable_token_reward_channel is on), so when off they stay None and the
    # returned tuple's last two slots are None — the caller attaches the batch keys
    # only when non-None, keeping the flag-off batch byte-identical.
    token_level_shaping_tensor = None
    if token_level_shaping is not None:
        token_level_shaping_tensor = torch.zeros_like(action_mask, dtype=torch.float)
        for i, sample_shaping in enumerate(token_level_shaping):
            if isinstance(sample_shaping, list):
                sample_shaping = torch.tensor(sample_shaping, dtype=torch.float)
            n = min(len(sample_shaping), token_level_shaping_tensor.size(1))
            token_level_shaping_tensor[i, :n] = sample_shaping[:n]

    response_span_tags_tensor = None
    if response_span_tags is not None:
        response_span_tags_tensor = torch.zeros_like(action_mask, dtype=torch.long)
        for i, sample_tags in enumerate(response_span_tags):
            if isinstance(sample_tags, list):
                sample_tags = torch.tensor(sample_tags, dtype=torch.long)
            n = min(len(sample_tags), response_span_tags_tensor.size(1))
            response_span_tags_tensor[i, :n] = sample_tags[:n]

    return (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        logprobs_tensor,
        routed_experts_tensor,
        token_level_shaping_tensor,
        response_span_tags_tensor,
    )
