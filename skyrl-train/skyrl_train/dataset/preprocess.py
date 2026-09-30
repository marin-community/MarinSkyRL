from typing import List, Tuple, Optional, Sequence
import numpy as np
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


def _collate_routed_experts_from_arrays(
    routed_experts: List[np.ndarray],
    max_output_len: int,
    num_experts: Optional[int],
    *,
    left_pad: bool = False,
) -> "torch.Tensor":
    """Build a dense ``[B, tokens, layer, top_k]`` routed-expert tensor.

    Rows are right-padded with zeroes in the final tensor dtype, or left-padded
    when ``left_pad`` is set (the prompt axis of the training sequences).
    """
    if any(rows.ndim != 3 for rows in routed_experts):
        raise ValueError("routed_experts must contain [token, layer, top_k] arrays")
    layers = max(rows.shape[1] for rows in routed_experts)
    top_k = max(rows.shape[2] for rows in routed_experts)
    _re_dtype = _routed_experts_dtype_for_num_experts(num_experts)
    if _re_dtype is None:
        _max_expert_id = max((int(rows.max()) for rows in routed_experts if rows.size), default=0)
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
    numpy_dtype = {torch.uint8: np.uint8, torch.int16: np.int16, torch.int64: np.int64}[_re_dtype]
    out = np.zeros((len(routed_experts), max_output_len, layers, top_k), dtype=numpy_dtype)
    for index, rows in enumerate(routed_experts):
        count = min(len(rows), max_output_len)
        start = max_output_len - count if left_pad else 0
        out[index, start : start + count, : rows.shape[1], : rows.shape[2]] = rows[:count]
    return torch.from_numpy(out)


def collate_prompt_routed_experts(
    prompt_routed_experts: List[np.ndarray],
    prompts: List[List[int]],
    num_experts: Optional[int],
) -> "torch.Tensor":
    """Build ``[B, prompt_len, layer, top_k]`` prompt routes aligned with the left-padded prompts.

    Each row holds the experts vLLM selected for that prompt token; padding rows are
    all zeroes, which router replay treats as native routing.
    """
    if len(prompt_routed_experts) != len(prompts):
        raise ValueError("prompt_routed_experts must have one entry per prompt")
    for rows, prompt in zip(prompt_routed_experts, prompts, strict=True):
        if len(rows) != len(prompt):
            raise ValueError("prompt_routed_experts must align with prompt token IDs")
    max_input_len = max(len(prompt) for prompt in prompts)
    return _collate_routed_experts_from_arrays(prompt_routed_experts, max_input_len, num_experts, left_pad=True)


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: Optional[List[torch.Tensor]],
    loss_masks: List[List[int]],
):
    assert len(prompts) == len(responses) and len(prompts) > 0, (
        "prompts and responses must have the same length and length must be greater than 0, got {} and {}".format(
            len(prompts), len(responses)
        )
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


def collate_response_token_channel(
    rows: Optional[Sequence[Sequence[float]]],
    response_template: torch.Tensor,
    *,
    dtype: torch.dtype,
    expected_lengths: Sequence[int],
) -> Optional[torch.Tensor]:
    """Validate and right-pad a scalar channel aligned one-to-one with response tokens."""
    if rows is None:
        return None
    if len(rows) != len(expected_lengths):
        raise ValueError("response token channel must have one row per response")
    if any(len(row) != expected for row, expected in zip(rows, expected_lengths)):
        raise ValueError("response token channel must align one-to-one with response tokens")
    result = torch.zeros_like(response_template, dtype=dtype)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row, dtype=dtype)
        result[index, : len(values)] = values
    return result


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: List[List[float]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[np.ndarray]] = None,
    routed_experts: Optional[List[np.ndarray]] = None,
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
    for prompt, response in zip(prompts, responses):
        prompt_token_len = len(prompt)
        response_token_len = len(response)
        prompt_token_lens.append(prompt_token_len)
        response_token_lens.append(response_token_len)

        max_input_len = max(max_input_len, prompt_token_len)
        max_output_len = max(max_output_len, response_token_len)

    # Copy each row's tokens into preallocated tensors. Building the padded batch from nested Python lists would
    # walk every padding element under the GIL, and padding dominates a batch with a long response window.
    batch_size = len(prompts)
    sequences = torch.full((batch_size, max_input_len + max_output_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_input_len + max_output_len), dtype=torch.int64)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # Left-pad the prompt and right-pad the response.
        prompt_start = max_input_len - prompt_token_lens[i]
        response_end = max_input_len + response_token_lens[i]
        sequences[i, prompt_start:max_input_len] = torch.as_tensor(prompt, dtype=torch.long)
        sequences[i, max_input_len:response_end] = torch.as_tensor(response, dtype=torch.long)
        attention_mask[i, prompt_start:response_end] = 1
    action_mask = attention_mask[:, max_input_len:].clone()

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
        logprobs_tensor = torch.zeros_like(action_mask, dtype=torch.float)
        for i, sample_logprobs in enumerate(logprobs):
            logprobs_tensor[i, : len(sample_logprobs)] = torch.as_tensor(sample_logprobs, dtype=torch.float)

    # MoE router-replay capture rail (Stage 1): right-pad routed_experts on the
    # response axis exactly like rollout_logprobs, but each per-token element is a
    # [L, K] expert-index vector. Result: [batch, response_len, L, K] int. Padding
    # rows are sentinel [L, K] (all zeros). 4-D is accepted by TensorBatch since
    # _check_consistency only validates dim-0.
    routed_experts_tensor = None
    if routed_experts:
        # Routes arrive as compact per-sample arrays. Slice assignment avoids
        # rebuilding the nested Python integer graph during collation.
        routed_experts_tensor = _collate_routed_experts_from_arrays(routed_experts, action_mask.size(1), num_experts)

    # Loop-behavior reward shaping (Stage B / F5 + F4): right-pad the per-token
    # shaping channel and span tags on the response axis exactly like rewards /
    # loss_mask. Both are gated upstream (only passed when
    # enable_token_reward_channel is on), so when off they stay None and the
    # returned tuple's last two slots are None — the caller attaches the batch keys
    # only when non-None, keeping the flag-off batch byte-identical.
    token_level_shaping_tensor = collate_response_token_channel(
        token_level_shaping,
        action_mask,
        dtype=torch.float,
        expected_lengths=response_token_lens,
    )
    response_span_tags_tensor = collate_response_token_channel(
        response_span_tags,
        action_mask,
        dtype=torch.long,
        expected_lengths=response_token_lens,
    )

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
