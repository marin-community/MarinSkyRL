import os
import torch
from difflib import SequenceMatcher
from typing import List, Tuple, Union, Optional, Dict, Any
from collections import defaultdict
import numpy as np
from skyrl_train.generators.base import GeneratorOutput, GeneratorInput, TrajectoryID, BatchMetadata, TrainingPhase
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from loguru import logger
from skyrl_gym.metrics import aggregate_for_environment


# Sentinel used to mark logprob positions that the alignment layer could NOT
# recover (no vLLM logprob available for that training token). Distinct from a
# legitimate 0.0 logprob so the metrics layer can count "holes" exactly. The
# training tensor still consumes a float, so callers replace UNALIGNED_LOGPROB
# with 0.0 right before emitting; the metrics are computed BEFORE that step.
UNALIGNED_LOGPROB = float("nan")


def _lcs_alert_threshold() -> float:
    """Threshold on ``tis/lcs_fallback_fraction`` above which the LCS guard ALERTS.

    Under full TITO the served-id splice / prompt-id assembly make tier-1
    exact-by-id alignment exact by construction, so ``align_logprobs_with_lcs`` is
    a DEFENSIVE GUARD that should fire ~0×. When it fires more than this fraction it
    signals a real serving↔training tokenizer/template regression and must surface
    loudly (metric ``tis/lcs_fallback_alert`` = 1.0 + an error log). Override via
    ``SKYRL_TIS_LCS_ALERT_THRESHOLD`` (default 0.005 = 0.5% of training tokens).
    """
    val = os.environ.get("SKYRL_TIS_LCS_ALERT_THRESHOLD")
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return 0.005


class AlignmentStats:
    """Per-assistant-message alignment bookkeeping for TIS logprob mapping.

    Carries enough signal to emit ``tis/exact_match_fraction``,
    ``tis/lcs_fallback_fraction`` and ``tis/alignment_fail_count`` at the batch
    level so an LCS fallback or a token-count divergence can NEVER silently
    degrade TIS — it is always observable on the dashboard.

    Fields (all token-counted, summed across messages in a trajectory):
        n_tokens:          total training (generated) tokens seen
        n_exact:           tokens mapped via the exact (token_id-zip) path
        n_lcs:             tokens mapped via the LCS string fallback
        n_unaligned:       training tokens that received NO vLLM logprob (holes)
        n_messages:        assistant messages processed
        n_lcs_messages:    assistant messages that took the LCS fallback path
        n_failed_messages: assistant messages where alignment fully failed
                           (entire message zeroed)
    """

    __slots__ = (
        "n_tokens",
        "n_exact",
        "n_lcs",
        "n_unaligned",
        "n_messages",
        "n_lcs_messages",
        "n_failed_messages",
    )

    def __init__(self):
        self.n_tokens = 0
        self.n_exact = 0
        self.n_lcs = 0
        self.n_unaligned = 0
        self.n_messages = 0
        self.n_lcs_messages = 0
        self.n_failed_messages = 0

    def merge(self, other: "AlignmentStats") -> None:
        self.n_tokens += other.n_tokens
        self.n_exact += other.n_exact
        self.n_lcs += other.n_lcs
        self.n_unaligned += other.n_unaligned
        self.n_messages += other.n_messages
        self.n_lcs_messages += other.n_lcs_messages
        self.n_failed_messages += other.n_failed_messages

    def as_metrics(self, prefix: str = "tis/") -> Dict[str, float]:
        n = max(self.n_tokens, 1)
        lcs_frac = self.n_lcs / n
        return {
            f"{prefix}aligned_tokens": float(self.n_tokens),
            f"{prefix}exact_match_fraction": self.n_exact / n,
            f"{prefix}lcs_fallback_fraction": lcs_frac,
            f"{prefix}unaligned_fraction": self.n_unaligned / n,
            f"{prefix}alignment_fail_count": float(self.n_failed_messages),
            f"{prefix}lcs_fallback_messages": float(self.n_lcs_messages),
            # Metered LCS guard: 1.0 when the LCS defensive fallback fired on more than
            # the alert threshold of training tokens (a serving↔training tokenizer/
            # template regression). ALWAYS emitted (keyset-stable across ranks — the
            # NumelIn=1 all_reduce-deadlock trap). Under full TITO this should stay 0.
            f"{prefix}lcs_fallback_alert": 1.0 if lcs_frac > _lcs_alert_threshold() else 0.0,
        }


def align_logprobs_by_token_ids(
    generated_token_ids: List[int],
    vllm_token_ids: List[int],
    vllm_logprobs: List[float],
    stats: Optional["AlignmentStats"] = None,
) -> Optional[List[float]]:
    """EXACT alignment: zip vLLM logprobs onto the training tokens by token id.

    This is the ROBUST path. vLLM emits, per generated token, both the token id
    (``completion_token_ids``) and its logprob (``logprobs``), index-aligned by
    construction. When the training-side ``generated_token_ids`` (sliced out of
    the re-tokenized assistant message) are IDENTICAL to ``vllm_token_ids``, the
    logprobs map 1:1 with NO guessing — positions are exact.

    Returns the per-token logprob list (len == len(generated_token_ids)) on an
    exact id match, or ``None`` if the ids diverge (caller should fall back to
    LCS and record the fallback in ``stats``). A divergence is expected to be
    rare; when it happens it means the served chat template / tokenizer and the
    training re-tokenization produced different ids for the same assistant turn.
    """
    if vllm_token_ids is None or vllm_logprobs is None:
        return None
    if len(vllm_token_ids) != len(vllm_logprobs):
        # vLLM contract violation (ids and logprobs must be parallel). Don't
        # trust either — force the caller down the LCS path.
        return None
    if list(generated_token_ids) != list(vllm_token_ids):
        return None
    if stats is not None:
        stats.n_exact += len(generated_token_ids)
    return list(vllm_logprobs)


def align_logprobs_with_lcs(
    retokenized_ids: List[int],
    vllm_token_logprobs: List[Dict[str, Any]],
    tokenizer,
    stats: Optional["AlignmentStats"] = None,
) -> List[float]:
    """Align vLLM logprobs to re-tokenized IDs using LCS on token strings.

    RETAINED DEFENSIVE GUARD (metered, not deleted). Prefer
    :func:`align_logprobs_by_token_ids`, which is exact. Under the served-id splice
    and full-TITO assembly, tier-1 exact-by-id alignment is exact by construction and
    this LCS path fires ~0× on real traces (empirically 0/25 across sampled MoE
    Qwen3-Coder trials — see agent_logs/2026-07-13_tito_rollout_migration.md). It is
    kept reachable ONLY as a last-resort safety net for a future tokenizer/template
    regression, and every invocation is recorded in ``stats`` (``n_lcs`` /
    ``n_lcs_messages`` / ``n_unaligned``) and surfaced as ``tis/lcs_fallback_fraction``
    + the ``tis/lcs_fallback_alert`` threshold metric (see :func:`_lcs_alert_threshold`)
    — retiring the silent repair to a METERED GUARD, never removing the signal. It is
    inherently a guess and CAN misalign, so its firing must always be observable.

    Args:
        retokenized_ids: Token IDs from re-tokenizing the response text
        vllm_token_logprobs: List of dicts with "token" (str) and "logprob" (float)
            from Harbor's rollout_details
        tokenizer: HuggingFace tokenizer used for re-tokenization
        stats: Optional AlignmentStats to record fallback counts into.

    Returns:
        List of aligned logprobs, one per retokenized_id. Unmatched tokens get 0.0.
    """
    if stats is not None:
        stats.n_lcs_messages += 1
    if not vllm_token_logprobs:
        if stats is not None:
            stats.n_unaligned += len(retokenized_ids)
            stats.n_failed_messages += 1
        return [0.0] * len(retokenized_ids)

    if not retokenized_ids:
        return []

    # Convert re-tokenized IDs to token strings for matching
    retok_strings = tokenizer.convert_ids_to_tokens(retokenized_ids)

    # Extract token strings and logprobs from vLLM output
    vllm_strings = [tl["token"] for tl in vllm_token_logprobs]
    vllm_logprobs = [tl["logprob"] for tl in vllm_token_logprobs]

    # Use SequenceMatcher to find LCS alignment
    matcher = SequenceMatcher(None, retok_strings, vllm_strings)
    aligned = [0.0] * len(retokenized_ids)
    matched_mask = [False] * len(retokenized_ids)

    # Get all matching blocks and assign logprobs
    for a_start, b_start, size in matcher.get_matching_blocks():
        for i in range(size):
            aligned[a_start + i] = vllm_logprobs[b_start + i]
            matched_mask[a_start + i] = True

    matched_count = sum(1 for m in matched_mask if m)
    if stats is not None:
        stats.n_lcs += matched_count
        stats.n_unaligned += len(retokenized_ids) - matched_count
        if matched_count == 0:
            stats.n_failed_messages += 1

    # Log alignment statistics for debugging
    if matched_count < len(retokenized_ids) * 0.9:  # Less than 90% matched
        logger.warning(
            f"TIS LCS fallback: matched only {matched_count}/{len(retokenized_ids)} tokens "
            f"(vLLM had {len(vllm_token_logprobs)} tokens). This indicates a "
            f"tokenizer/chat-template mismatch between serving and training. "
            f"First few retok: {retok_strings[:5]}, vLLM: {vllm_strings[:5]}"
        )

    return aligned


CUSTOM_CHAT_TEMPLATES = {
    # chat template for qwen3 that preserves thinking tokens
    "qwen3_with_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # chat template for qwen3 that strips non-last-turn thinking tokens (same as the official Qwen3 chat
    # template but we add `generation` and `endgeneration` tags)
    "qwen3_without_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{% set full_content = message['content'] %}"
        "{% set mycontent = message['content'] %}"
        "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
        "{% if '</think>' in full_content and not is_last_message %}"
        "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
        "{% endif %}"
        "{{mycontent + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # Qwen2.5 chat template but with `generation` and `endgeneration` tags, and simplified
    "qwen2_5_with_generation_tag_simplified": (
        "{% for message in messages %}"
        "{% if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
        "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}"
        "{% elif message.role == 'assistant' %}"
        "{{ '<|im_start|>' + message.role + '\n'}}"
        "{% generation %}"
        "{{ message.content + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{ '\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    ),
}


def get_custom_chat_template(chat_template_config: Optional[Union[dict, DictConfig]] = None) -> Optional[str]:
    """
    Get custom chat template based on the new config structure.

    Args:
        chat_template_config: Config dict with 'source' and 'name_or_path' fields.

    Returns:
        Chat template string or None
    """
    if chat_template_config is None:
        return None

    source = chat_template_config.get("source")
    if not source:
        raise ValueError("'source' is required in chat_template_config")

    name_or_path = chat_template_config.get("name_or_path")
    if not name_or_path:
        return None  # if name_or_path is not provided, use the default chat template from the tokenizer

    if source == "name":
        if name_or_path in CUSTOM_CHAT_TEMPLATES:
            return CUSTOM_CHAT_TEMPLATES[name_or_path]
        else:
            raise ValueError(
                f"Template name '{name_or_path}' not found. Available templates: {list(CUSTOM_CHAT_TEMPLATES.keys())}"
            )
    elif source == "file":
        try:
            with open(name_or_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(f"Template file '{name_or_path}' not found") from e
        except OSError as e:
            raise ValueError(f"Error reading template file '{name_or_path}': {e}") from e
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'name' or 'file'")


def normalize_token_ids(encoded) -> List[int]:
    """Coerce a ``tokenizer.apply_chat_template(..., tokenize=True)`` result into a
    flat ``List[int]`` of token ids.

    ``apply_chat_template`` is *supposed* to return a flat list of ints when
    ``return_dict=False`` (our call-site default). But several legitimate upstream
    shapes leak through and break ``len()``-based slicing and downstream batch
    collation:

    * ``BatchEncoding`` / ``dict`` / any mapping — when a tokenizer / chat-template
      path returns the full encoding instead of just the ids (the mapping's
      ``input_ids`` value is the real ids). NOTE that ``transformers.BatchEncoding``
      is a ``UserDict``, NOT a ``dict`` subclass (``isinstance(BatchEncoding(...),
      dict)`` is False on transformers 4.57+), so iterating it yields its KEYS and
      ``len()`` returns the *key count* (e.g. 2) — which is exactly how the Qwen3-
      Next-80B response-side ``len()``-slicing produced empty ``response_ids`` /
      ``loss_mask`` ("All outputs are loss masked" -> NaN advantages -> no-op
      step). We detect it by its mapping interface (``keys()`` / ``__getitem__``),
      not ``isinstance(dict)``.
    * tensor / ndarray — flattened via ``.tolist()``.
    * ``[[int, ...]]`` — a singleton-batched nested list. We unwrap the single row.
    * ``List[int]`` — the normal/correct path, returned unchanged (no ``.tolist()``,
      no unwrap), so the 8B/a3 flat-list-template path is byte-identical.
    """
    # BatchEncoding / dict / any Mapping carrying the ids under a key.
    #
    # NOTE: `transformers.BatchEncoding` is a `UserDict`, NOT a `dict` subclass
    # (`isinstance(BatchEncoding(...), dict)` is False on transformers 4.57+),
    # and both iterating it and `len()` operate on its KEYS. So we must detect it
    # by its mapping interface (`keys()` / `__getitem__`), not by `isinstance(dict)`.
    if hasattr(encoded, "keys") and hasattr(encoded, "__getitem__") and not isinstance(encoded, (list, tuple)):
        keys = list(encoded.keys())
        for key in ("input_ids", "token_ids", "ids"):
            if key in keys:
                encoded = encoded[key]
                break
        else:
            raise ValueError(
                "apply_chat_template returned a mapping without an "
                "'input_ids'/'token_ids'/'ids' key "
                f"(keys={keys}); cannot recover token ids."
            )

    # Tensor / ndarray (e.g. a BatchEncoding value or a return_tensors result).
    if hasattr(encoded, "tolist") and not isinstance(encoded, (list, tuple)):
        encoded = encoded.tolist()

    encoded = list(encoded)

    # Singleton-batched nesting: [[int, ...]] -> [int, ...]. Only unwrap a
    # length-1 outer list whose sole element is itself a list of ints.
    if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = list(encoded[0])

    return encoded


def get_generation_prompt_ids(tokenizer, custom_chat_template=None, chat_template_kwargs=None) -> List[int]:
    """
    Helper function to get the generation prompt ids for a given tokenizer.

    ``chat_template_kwargs`` (e.g. ``{"enable_thinking": True}``) is forwarded to
    ``apply_chat_template`` so the rendered generation prompt MATCHES the served
    vLLM stream. For Qwen3.5/3.6 thinking models this is LOAD-BEARING: with it
    UNSET the chat template auto-injects an EMPTY ``<think>\\n\\n</think>`` block
    (the ``enable_thinking is false`` default branch), which the served
    completion_token_ids never contain -> TIS prefix divergence (see
    detect_qwen3_5_empty_think_prefix). Passing ``enable_thinking=True`` renders
    the bare ``<think>`` open instead, matching the served stream. ``None`` ->
    byte-identical to the old call (no kwargs forwarded).
    """
    ctk = chat_template_kwargs or {}
    empty_user = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=True, chat_template=custom_chat_template, **ctk
        )
    )
    empty_user_with_generation_prompt = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
            chat_template=custom_chat_template,
            **ctk,
        )
    )

    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]
    return generation_prompt_ids


def detect_qwen3_5_empty_think_prefix(tokenizer, generation_prompt_ids: List[int]) -> Optional[List[int]]:
    """Detect the Qwen3.5/3.6 (``qwen3_5_moe``) empty-think generation prompt and,
    if present, return the prefix ids UP TO BUT NOT INCLUDING the injected empty
    ``<think> ... </think>`` block.

    The Qwen3.5/3.6 chat template renders the assistant generation prompt as
    ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` — it auto-injects an
    EMPTY reasoning block. At ROLLOUT time the served ``completion_token_ids`` do
    NOT contain that injected empty block: the model emits its OWN ``<think>`` with
    real reasoning content (interleaved-thinking protocol). So the re-tokenized
    assistant turn (``<|im_start|>assistant\\n<think>real…</think>…``) NEVER matches
    the full ``generation_prompt_ids`` prefix (which expects the EMPTY think block),
    the prefix-strip falls back to ``prefix_len=0``, and the re-tokenized generated
    region diverges from the served stream in both COUNT and CONTENT — collapsing
    TIS tier-1 exact alignment (observed: served 32/41 vs re-tok 35/49 for the same
    turn → ``serving↔training tokenizer divergence`` → logprobs zeroed).

    This is qwen3_5-SPECIFIC: it is detected purely from the tokenizer's own
    generation prompt (a ``<think>`` token immediately followed — modulo a single
    whitespace token — by ``</think>``). Dense Qwen3 (qwen3_with_thinking custom
    template) and Qwen2.5 do NOT inject an empty think block, so this returns None
    for them and every non-qwen3_5 path is byte-identical.

    Returns:
        The leading ``<|im_start|>assistant\\n`` ids (the prefix BEFORE the injected
        ``<think>``), to be used as the real generation-prompt prefix for splicing
        the served ids in. ``None`` when no empty-think block is detected.
    """
    think_open_id = tokenizer.convert_tokens_to_ids("<think>")
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    unk = getattr(tokenizer, "unk_token_id", None)
    # If the tokenizer has no <think>/</think> special tokens, this is not a
    # qwen3_5-style thinking model — bail (byte-identical to old behavior).
    if think_open_id is None or think_close_id is None:
        return None
    if think_open_id == unk or think_close_id == unk:
        return None
    if think_open_id not in generation_prompt_ids or think_close_id not in generation_prompt_ids:
        return None
    open_pos = generation_prompt_ids.index(think_open_id)
    close_pos = generation_prompt_ids.index(think_close_id)
    # Require the close to follow the open with at most one intervening token
    # (the ``\n\n`` whitespace token) — i.e. an EMPTY injected reasoning block.
    if not (open_pos < close_pos <= open_pos + 2):
        return None
    return list(generation_prompt_ids[:open_pos])


@torch.no_grad()
def get_metrics_from_generator_output(generator_output: GeneratorOutput, uids: List[str]) -> Tuple[float, float]:
    """
    Get `mean_raw_reward` (or avg_score), `pass_at_n` from generator output.

    The `n` in `pass_at_n` is the number of trajectories we generate for each example. It is
    calculated as `len(generator_output["rewards"]) / len(uids)`, where `len(uids)` is the number of
    unique examples.

    Rewards can be either per-trajectory or per-token, and metrics are computed correspondingly.
    """
    rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
    if not len(rewards):
        raise ValueError(f"`rewards` must be a non-empty list, got {rewards}")

    # TODO: We should make metrics customizable by the environment.
    # Map from the example's uid to each trajectory's reward on that same example
    uid_to_trajectory_rewards = defaultdict(list)
    if isinstance(rewards[0], list):
        # Token-level rewards: rewards is List[List[float]]
        # For each trajectory, we sum over the token rewards for `mean_raw_reward` computation
        mean_raw_reward = float(np.mean([sum(trajectory_rewards) for trajectory_rewards in rewards]))
        # Assume the last token's reward signifies the trajectory's reward for `pass_at_n` computation
        for i, cur_trajectory_rewards in enumerate(rewards):
            # A prompt rejected before inference has no terminal token and contributes zero reward.
            terminal_reward = cur_trajectory_rewards[-1] if cur_trajectory_rewards else 0.0
            uid_to_trajectory_rewards[uids[i]].append(terminal_reward)
    else:
        mean_raw_reward = float(np.mean(rewards))
        for i, reward in enumerate(rewards):
            uid_to_trajectory_rewards[uids[i]].append(reward)

    # For each example, pass@n = 1 if any trajectory achieves a positive reward.
    # With binary rewards, this means any success. With shaped rewards (e.g. pass_ratio),
    # this means any partial progress. Using > 0.0 rather than >= 1.0 because shaped
    # rewards may never reach 1.0 (e.g. 9/10 tests = 0.9).
    pass_at_n = sum(1 for v in uid_to_trajectory_rewards.values() if any(r > 0.0 for r in v)) / len(
        uid_to_trajectory_rewards
    )

    return mean_raw_reward, pass_at_n


def concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """
    Concatenate the generator outputs of multiple batches.

    We only aggregate rollout metrics the can deduced by responses and rewards, but not
    those that use `env_metrics` or `env_classes`.
    """
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [output.get("rollout_logprobs") is not None for output in generator_outputs]
    any_has_logprobs = any(has_rollout_logprobs)

    # Handle mixed rollout_logprobs: if some batches have logprobs and others don't,
    # fill in placeholder [0.0] values for the batches that don't have them.
    # This can happen when all trials in a batch fail (returns None) while other batches succeed.
    rollout_logprobs_concat = None
    if any_has_logprobs:
        rollout_logprobs_concat = []
        for output in generator_outputs:
            if output.get("rollout_logprobs") is not None:
                rollout_logprobs_concat.extend(output["rollout_logprobs"])
            else:
                # Fill in placeholder logprobs for batches that don't have them
                # Each trajectory needs logprobs matching its response_ids length
                for response_ids in output["response_ids"]:
                    rollout_logprobs_concat.append([0.0] * len(response_ids))

    # Handle mixed routed_experts (Stage 1 MoE router-replay capture rail) the same
    # way as rollout_logprobs: if any batch carries routed_experts but others don't,
    # sentinel-fill the missing batches with a per-token [1, 1] sentinel row so the
    # concatenated list stays 1:1 with response_ids. When the flag is off, NO batch
    # carries the key (the generator omits it), so this stays None and the result
    # dict is byte-identical to today.
    has_routed_experts = [
        "rollout_routed_experts" in output and output.get("rollout_routed_experts") is not None
        for output in generator_outputs
    ]
    rollout_routed_experts_concat = None
    if any(has_routed_experts):
        # Learn the real [L, K] per-token row shape from the FIRST sample that
        # actually carries routing. Samples lacking routing (preempted requests,
        # quant paths) must be sentinel-filled with the SAME [L, K] width — a
        # degenerate [1, 1] sentinel here makes the L axis ragged across the batch
        # ([48, K] real rows vs [1, 1] sentinels) and crashes the dense
        # torch.tensor() collation in convert_prompts_responses_to_batch_tensors
        # ("expected sequence of length 1 at dim 2"). See _sentinel_routed_experts_row.
        _concat_sentinel_row = None
        for output in generator_outputs:
            re_out = output.get("rollout_routed_experts")
            if re_out is not None and len(re_out) > 0:
                for sample_re in re_out:
                    if sample_re is not None and len(sample_re) > 0:
                        _concat_sentinel_row = _sentinel_routed_experts_row(sample_re[0])
                        break
            if _concat_sentinel_row is not None:
                break
        rollout_routed_experts_concat = []
        for output in generator_outputs:
            if "rollout_routed_experts" in output and output.get("rollout_routed_experts") is not None:
                rollout_routed_experts_concat.extend(output["rollout_routed_experts"])
            else:
                for response_ids in output["response_ids"]:
                    rollout_routed_experts_concat.append(_re_sentinel_rows(len(response_ids), _concat_sentinel_row))

    # Loop-behavior reward shaping (Stage B / F5 + F4): mix the per-token shaping
    # channel + span tags the same way as routed_experts — sentinel-fill (zeros)
    # any batch that lacks them so the concatenated list stays 1:1 with
    # response_ids. When the channel is off NO batch carries the keys (the
    # generator omits them), so these stay None and the result is byte-identical.
    has_token_shaping = [
        "token_level_shaping" in output and output.get("token_level_shaping") is not None
        for output in generator_outputs
    ]
    token_level_shaping_concat = None
    if any(has_token_shaping):
        token_level_shaping_concat = []
        for output in generator_outputs:
            if "token_level_shaping" in output and output.get("token_level_shaping") is not None:
                token_level_shaping_concat.extend(output["token_level_shaping"])
            else:
                for response_ids in output["response_ids"]:
                    token_level_shaping_concat.append([0.0] * len(response_ids))

    has_span_tags = [
        "response_span_tags" in output and output.get("response_span_tags") is not None for output in generator_outputs
    ]
    response_span_tags_concat = None
    if any(has_span_tags):
        response_span_tags_concat = []
        for output in generator_outputs:
            if "response_span_tags" in output and output.get("response_span_tags") is not None:
                response_span_tags_concat.extend(output["response_span_tags"])
            else:
                for response_ids in output["response_ids"]:
                    response_span_tags_concat.append([0] * len(response_ids))

    result: GeneratorOutput = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in generator_outputs], []),
        "response_ids": sum([output["response_ids"] for output in generator_outputs], []),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
        "stop_reasons": (
            sum([output["stop_reasons"] for output in generator_outputs], [])
            if "stop_reasons" in generator_outputs[0] and generator_outputs[0]["stop_reasons"] is not None
            else None
        ),
        "rollout_logprobs": rollout_logprobs_concat,
    }
    if rollout_routed_experts_concat is not None:
        result["rollout_routed_experts"] = rollout_routed_experts_concat
    if token_level_shaping_concat is not None:
        result["token_level_shaping"] = token_level_shaping_concat
    if response_span_tags_concat is not None:
        result["response_span_tags"] = response_span_tags_concat

    # propagate additional keys with list values as-is
    additional_keys = [
        key for key in generator_outputs[0] if key not in result and isinstance(generator_outputs[0][key], list)
    ]
    if len(additional_keys):
        logger.info(f"Attempting to concatenate values for additional keys {additional_keys}")
    for key in additional_keys:
        result[key] = sum([generator_output[key] for generator_output in generator_outputs], [])

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])

    # Preserve the TIS alignment metrics (generate/tis/*). get_rollout_metrics
    # only derives reward/length stats, so without this the per-batch TIS
    # alignment health (exact_match_fraction / lcs_fallback_fraction /
    # alignment_fail_count) would be SILENTLY DROPPED on the fully-async path
    # (concatenate happens per training step there) and never reach wandb. Merge
    # them back by recombining the token-weighted fractions across batches.
    total_aligned = 0.0
    sum_exact = sum_lcs = sum_unaligned = 0.0
    sum_fail = sum_lcs_msgs = 0.0
    saw_tis = False
    for output in generator_outputs:
        rm = output.get("rollout_metrics") or {}
        n = rm.get("generate/tis/aligned_tokens")
        if n is None:
            continue
        saw_tis = True
        total_aligned += n
        sum_exact += rm.get("generate/tis/exact_match_fraction", 0.0) * n
        sum_lcs += rm.get("generate/tis/lcs_fallback_fraction", 0.0) * n
        sum_unaligned += rm.get("generate/tis/unaligned_fraction", 0.0) * n
        sum_fail += rm.get("generate/tis/alignment_fail_count", 0.0)
        sum_lcs_msgs += rm.get("generate/tis/lcs_fallback_messages", 0.0)
    if saw_tis:
        denom = max(total_aligned, 1.0)
        rollout_metrics["generate/tis/aligned_tokens"] = total_aligned
        rollout_metrics["generate/tis/exact_match_fraction"] = sum_exact / denom
        rollout_metrics["generate/tis/lcs_fallback_fraction"] = sum_lcs / denom
        rollout_metrics["generate/tis/unaligned_fraction"] = sum_unaligned / denom
        rollout_metrics["generate/tis/alignment_fail_count"] = sum_fail
        rollout_metrics["generate/tis/lcs_fallback_messages"] = sum_lcs_msgs
        # Recompute the metered LCS-guard alert from the recombined fraction so it
        # stays keyset-stable and consistent with the per-trajectory emission.
        rollout_metrics["generate/tis/lcs_fallback_alert"] = 1.0 if (sum_lcs / denom) > _lcs_alert_threshold() else 0.0

    # Preserve the batch failure counts for the same reason: get_rollout_metrics
    # derives nothing from them, and the fully-async trainer reads rollout metrics
    # off this result only. Sum the counts and recompute the fraction from those
    # totals — averaging the per-group fractions would weight unequal groups equally.
    failure_counts = [
        rm
        for rm in (output.get("rollout_metrics") or {} for output in generator_outputs)
        if "generate/num_trials" in rm
    ]
    if failure_counts:
        totals = {key: sum(rm.get(key, 0) for rm in failure_counts) for key in _BATCH_FAILURE_COUNT_KEYS}
        rollout_metrics.update(
            get_batch_failure_metrics(
                totals["generate/num_trials"],
                num_failed_trajectories=totals["generate/num_failed_trajectories"],
                num_failed_instances=totals["generate/num_failed_instances"],
                num_masked_trajectories=totals["generate/num_masked_trajectories"],
            )
        )

    result["rollout_metrics"] = rollout_metrics

    # Validate the generator output using the number of prompts
    # Import here to avoid circular dependency.
    from skyrl_train.utils.trainer_utils import validate_generator_output

    num_prompts = len(result["prompt_token_ids"])
    validate_generator_output(num_prompts, result)

    return result


def apply_overlong_filtering(
    loss_masks: List[List[int]],
    response_ids: List[List[int]],
    eos_token_id: int,
) -> List[List[int]]:
    """
    Implements DAPO Overlong Filtering: zero-out every token's mask whenever
    the response does not end with the eos token id (i.e. truncated).

    Returns:
        - The loss masks with tokens zeroed out for truncated responses
    """
    assert len(loss_masks) == len(response_ids), "loss_masks and response_ids must have the same length"
    return [
        [0] * len(mask) if not response or response[-1] != eos_token_id else mask
        for mask, response in zip(loss_masks, response_ids)
    ]


def get_rollout_metrics(
    responses: List[List[int]],
    rewards: Union[List[float], List[List[float]]],
    env_metrics: Optional[List[Dict[str, Any]]] = None,
    env_classes: Optional[List[str]] = None,
):
    """
    Computes rollout metrics including token statistics and optional environment-specific metrics.

    Args:
        responses: List of token ID sequences for each response
        rewards: List of rewards (either per-trajectory or per-token)
        env_metrics: Optional list of environment-specific metrics for each trajectory
        env_classes: Optional list of environment class names for each trajectory

    Returns:
        Dictionary of aggregated metrics
    """
    num_tokens_arr = np.array([len(response) for response in responses])
    # Support both response-level and token-level rewards
    flat_rewards = []
    for r in rewards:
        if isinstance(r, list):
            flat_rewards.append(float(np.sum(r)))
        else:
            flat_rewards.append(float(r))
    flat_rewards_arr = np.array(flat_rewards)
    non_zero_rewards_arr = flat_rewards_arr > 0.0
    zero_rewards_arr = flat_rewards_arr == 0.0
    # average tokens for non zero rewards
    avg_tokens_non_zero_rewards = (
        np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
    )
    # average tokens for zero rewards
    avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)

    rollout_metrics = {
        "generate/min_num_tokens": np.min(num_tokens_arr).item(),
        "generate/max_num_tokens": np.max(num_tokens_arr).item(),
        "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
        "generate/std_num_tokens": np.std(num_tokens_arr).item(),
        "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
        "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
    }

    if env_metrics is not None and env_classes is not None:
        env_to_metrics = defaultdict(list)
        for i, metrics in enumerate(env_metrics):
            env_to_metrics[env_classes[i]].append(metrics)
        for env_name, metrics in env_to_metrics.items():
            # Aggregate metrics across all trajectories for the same environment
            agg = aggregate_for_environment(env_name, metrics)
            for key, value in agg.items():
                rollout_metrics[f"environment/{key}"] = value

    return rollout_metrics


_BATCH_FAILURE_COUNT_KEYS = (
    "generate/num_trials",
    "generate/num_failed_instances",
    "generate/num_failed_trajectories",
    "generate/num_masked_trajectories",
)


def get_batch_failure_metrics(
    num_trials: int,
    num_failed_trajectories: int,
    num_failed_instances: int,
    num_masked_trajectories: int,
) -> Dict[str, float]:
    """Failure counts for one generated batch, plus the fraction of it they cost.

    The denominator is one ``generate`` call, which is the whole training batch on
    the synchronous trainer and a single rollout group on the fully asynchronous
    one. It is emitted alongside the counts so ``concatenate_generator_outputs``
    can re-derive the fraction over a merged batch.

    Args:
        num_trials: Trajectories the batch asked for; the denominator.
        num_failed_instances: Distinct instances with at least one failed trajectory.
    """
    return {
        "generate/num_trials": num_trials,
        "generate/num_failed_instances": num_failed_instances,
        "generate/num_failed_trajectories": num_failed_trajectories,
        "generate/num_masked_trajectories": num_masked_trajectories,
        "generate/failed_trajectory_fraction": num_failed_trajectories / num_trials if num_trials else 0.0,
    }


def prepare_generator_input(
    prompts: List[Any],
    n_samples_per_prompt: int,
    sampling_params: Dict[str, Any],
    default_env_class: str,
    training_phase: TrainingPhase,
    global_step: int,
) -> Tuple[GeneratorInput, List[str]]:
    """Prepares the generator input for training and eval

    Args:
        prompts (List[Any]): list of prompts
        n_samples_per_prompt (int): how many samples to create per prompt
        sampling_params (Dict[str, Any]): sampling parameters
        default_env_class (str): env class to use if env class missing from prompts
        training_phase (TrainingPhase): training or eval
        global_step (int): current global step

    Returns:
        Tuple[GeneratorInput, List[str]]: generator input and list of uuids
    """

    all_prompts = [prompt["prompt"] for prompt in prompts for _ in range(n_samples_per_prompt)]

    all_envs = [
        prompt["env_class"] if prompt["env_class"] is not None else default_env_class
        for prompt in prompts
        for _ in range(n_samples_per_prompt)
    ]

    # all the other columns are env_extras
    env_extras = [prompt["env_extras"] for prompt in prompts for _ in range(n_samples_per_prompt)]

    # Create TrajectoryID objects - one UID per row, repetition_id for multiple samples
    trajectory_ids = []
    uids = []
    for _, prompt in enumerate(prompts):
        uid: str = prompt["uid"]

        # Create TrajectoryID for each repetition
        for repetition_id in range(n_samples_per_prompt):
            trajectory_ids.append(TrajectoryID(instance_id=uid, repetition_id=repetition_id))
            uids.append(uid)

    generator_input: GeneratorInput = {
        "prompts": all_prompts,
        "env_classes": all_envs,
        "env_extras": env_extras,
        "sampling_params": sampling_params,
        "trajectory_ids": trajectory_ids,
        "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
    }

    return generator_input, uids


def encode_messages_subset(messages: ConversationType, tokenizer, custom_chat_template=None, chat_template_kwargs=None):
    """Encodes a subset of messages from a multi-turn conversation using the fixed base approach.

    This function tokenizes messages as if they are part of a larger conversation, ensuring
    no additional default system messages are prepended by the tokenizer's chat template

    The "fixed base approach" works by:
    - Creating a dummy base conversation to establish context
    - Appending the target messages to this base
    - Tokenizing the full conversation and extracting only the tokens for the target messages

    For simple chat templates without complex token splitting behavior, this produces the same
    result as directly tokenizing the messages. For templates like Qwen's ChatML format where
    a default system prompt can be appended, this ensures correct tokenization.

    In addition, for Qwen3, this function will keep all the thinking tokens from the messages.

    Reference: https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach

    Args:
        messages: List of message dicts with 'role' and 'content' keys. Must contain at least
                 one message. These are assumed to be a subset from a larger conversation.
        tokenizer: HuggingFace tokenizer with chat_template support and eos_token_id defined.
        custom_chat_template: Optional custom chat template string to use instead of tokenizer's default.
        chat_template_kwargs: Optional dict forwarded to apply_chat_template (e.g.
            ``{"enable_thinking": True}``). Threaded for consistency with the served
            stream / get_generation_prompt_ids; ``None`` -> byte-identical old behavior.
            (These calls use ``add_generation_prompt=False``, so enable_thinking does
            not change the encoded message ids under the current Qwen3.5 templates; it
            is forwarded defensively so any template that gates message-body rendering
            on it stays consistent between serve and re-tokenize.)

    Returns:
        List[int]: Token IDs for the given messages, with proper multi-turn context handling.
    """
    assert len(messages), "messages list cannot be empty"
    ctk = chat_template_kwargs or {}
    # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
    base_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
    ]
    base_conversation_token_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            base_conversation,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=custom_chat_template,
            **ctk,
        )
    )

    full_conversation = base_conversation + messages
    full_conversation_token_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            full_conversation,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=custom_chat_template,
            **ctk,
        )
    )
    conversation_token_ids = full_conversation_token_ids[len(base_conversation_token_ids) :]
    return conversation_token_ids


def extract_logprobs_from_rollout_details(
    rollout_details: Optional[List[Dict[str, Any]]],
) -> Optional[List[List[Dict[str, Any]]]]:
    """
    Extract per-turn logprobs (with token strings) from Harbor's rollout_details structure.

    Harbor stores rollout details as a list of RolloutDetail dicts. Each RolloutDetail
    contains per-turn data for a conversation trajectory:
        - prompt_token_ids: list[list[int]] - prompt tokens per turn
        - completion_token_ids: list[list[int]] - completion tokens per turn
        - logprobs: list[list[dict]] - logprobs per turn, where each dict has
            {"token": str, "logprob": float} for LCS alignment

    For agents with subagents or summarization, multiple RolloutDetail objects may exist.
    By convention, the first RolloutDetail contains the main agent's conversation.

    Args:
        rollout_details: List of RolloutDetail dicts from Harbor's AgentContext.
            Can be None or empty if rollout details weren't collected.

    Returns:
        Per-turn logprobs, or None if rollout_details is empty/missing or doesn't
        contain logprobs. Two inner formats are accepted and both are now
        supported (the float format is the canonical Harbor format and pairs
        index-for-index with ``completion_token_ids``):
            - float format: [[float, float, ...]_turn1, ...]
            - dict  format: [[{"token": str, "logprob": float}, ...]_turn1, ...]
        The float format is preferred: combined with the per-turn
        ``completion_token_ids`` (see :func:`extract_token_ids_from_rollout_details`)
        it enables the EXACT token-id alignment path, which needs no LCS guessing.

    Example:
        >>> rollout_details = result.agent_result.rollout_details
        >>> assistant_logprobs = extract_logprobs_from_rollout_details(rollout_details)
        >>> assistant_token_ids = extract_token_ids_from_rollout_details(rollout_details)
        >>> response_ids, loss_mask, rollout_logprobs, _stats = get_response_ids_and_loss_mask_from_messages(
        ...     messages, tokenizer, assistant_logprobs, assistant_token_ids=assistant_token_ids
        ... )
    """
    if not rollout_details or len(rollout_details) == 0:
        return None

    # First rollout_detail contains the main agent's conversation
    main_rollout = rollout_details[0]

    # Handle both dict and object-like access patterns
    if isinstance(main_rollout, dict):
        logprobs = main_rollout.get("logprobs")
    else:
        logprobs = getattr(main_rollout, "logprobs", None)

    if not logprobs:
        return None

    # Validate structure: should be list of lists
    if not isinstance(logprobs, list):
        logger.warning(f"Unexpected logprobs type: {type(logprobs)}, expected list")
        return None

    if len(logprobs) > 0 and not isinstance(logprobs[0], list):
        logger.warning(
            f"Unexpected logprobs[0] type: {type(logprobs[0])}, expected list. "
            f"rollout_details may have unexpected structure."
        )
        return None

    # Float format ([[float, ...], ...]) is now FULLY supported via the exact
    # token-id alignment path (it rides index-for-index with completion_token_ids),
    # so we no longer disable TIS for it. Dict format ([[{token, logprob}], ...])
    # is still accepted for the LCS fallback path. We pass either format through
    # untouched; the downstream alignment layer detects which it received.
    logger.debug(f"Extracted logprobs from rollout_details: {len(logprobs)} turns")
    return logprobs


def extract_token_ids_from_rollout_details(
    rollout_details: Optional[List[Dict[str, Any]]],
) -> Optional[List[List[int]]]:
    """Extract per-turn ``completion_token_ids`` from Harbor's rollout_details.

    These are the EXACT token ids vLLM generated for each assistant turn,
    index-aligned with the per-turn ``logprobs`` floats. Carrying them into
    :func:`get_response_ids_and_loss_mask_from_messages` lets TIS map logprobs to
    training tokens by token id (exact, no re-tokenization guess) instead of
    string-LCS. Mirrors :func:`extract_logprobs_from_rollout_details`.

    Returns per-turn ids ``[[id, ...]_turn1, ...]`` or None when absent.
    """
    if not rollout_details or len(rollout_details) == 0:
        return None

    main_rollout = rollout_details[0]
    if isinstance(main_rollout, dict):
        token_ids = main_rollout.get("completion_token_ids")
    else:
        token_ids = getattr(main_rollout, "completion_token_ids", None)

    if not token_ids:
        return None
    if not isinstance(token_ids, list):
        logger.warning(f"Unexpected completion_token_ids type: {type(token_ids)}, expected list")
        return None
    if len(token_ids) > 0 and not isinstance(token_ids[0], list):
        logger.warning(f"Unexpected completion_token_ids[0] type: {type(token_ids[0])}, expected list.")
        return None
    return token_ids


def extract_prompt_token_ids_from_rollout_details(
    rollout_details: Optional[List[Dict[str, Any]]],
) -> Optional[List[List[int]]]:
    """Extract per-turn ``prompt_token_ids`` from Harbor's rollout_details.

    Sibling of :func:`extract_token_ids_from_rollout_details` (which reads the
    per-turn *completion* ids). ``prompt_token_ids[t]`` is the EXACT token id
    sequence the inference engine tokenized as the prompt for turn ``t`` — the
    full growing context (system + user + every prior assistant completion + every
    tool observation, plus the assistant generation prompt). Harbor accumulates it
    per turn in ``chat.py`` (``_prompt_token_ids_list``). By construction it
    satisfies the prefix invariant

        ``prompt_token_ids[t] == prompt_token_ids[t-1] + completion_token_ids[t-1] + observation[t-1]``

    so consuming it lets the trainer assemble the MASKED context from the exact
    served ids instead of re-tokenizing it (full TITO — see
    :func:`_tito_full_enabled`). Returns per-turn ids ``[[id, ...]_turn0, ...]`` or
    None when absent (None-safe, mirrors the completion extractor).
    """
    if not rollout_details or len(rollout_details) == 0:
        return None

    main_rollout = rollout_details[0]
    if isinstance(main_rollout, dict):
        token_ids = main_rollout.get("prompt_token_ids")
    else:
        token_ids = getattr(main_rollout, "prompt_token_ids", None)

    if not token_ids:
        return None
    if not isinstance(token_ids, list):
        logger.warning(f"Unexpected prompt_token_ids type: {type(token_ids)}, expected list")
        return None
    if len(token_ids) > 0 and not isinstance(token_ids[0], list):
        logger.warning(f"Unexpected prompt_token_ids[0] type: {type(token_ids[0])}, expected list.")
        return None
    return token_ids


def extract_routed_experts_from_rollout_details(
    rollout_details: Optional[List[Dict[str, Any]]],
) -> Optional[List[Any]]:
    """Extract per-turn MoE ``routed_experts`` from Harbor's rollout_details.

    Sibling of :func:`extract_logprobs_from_rollout_details`. The vLLM fork emits
    per-token expert-selection indices ``[gen_len, L, K]`` (L = MoE layers,
    K = top-k experts) over ``/v1`` non-streaming via ``provider_specific_fields``.
    Harbor's ``_extract_provider_extra`` lands this in
    ``RolloutDetail.extra["routed_experts"]`` as a per-turn list (one ``[gen_len, L, K]``
    entry per assistant turn, aligned with ``completion_token_ids``).

    This is Stage 1 of the FSDP2 EP/router-replay port (R3 capture rail). No MoE
    math here — pure data-plane extraction. Returns None when absent so the field
    is treated as a sentinel-filled sample downstream (preempted requests, quant
    paths, and disabled-capture modes silently drop routing — see
    notes/skyrl/stage1_capture_rail_scope.md Q1).

    Args:
        rollout_details: List of RolloutDetail dicts from Harbor's AgentContext.

    Returns:
        Per-turn routed_experts ``[[gen_len, L, K]_turn1, ...]``, or None if
        rollout_details is empty/missing or doesn't carry routed_experts.
    """
    if not rollout_details or len(rollout_details) == 0:
        return None

    # First rollout_detail contains the main agent's conversation.
    main_rollout = rollout_details[0]

    if isinstance(main_rollout, dict):
        extra = main_rollout.get("extra")
    else:
        extra = getattr(main_rollout, "extra", None)

    if not extra or not isinstance(extra, dict):
        return None

    routed_experts = extra.get("routed_experts")
    if not routed_experts:
        return None

    if not isinstance(routed_experts, list):
        logger.warning(f"Unexpected routed_experts type: {type(routed_experts)}, expected list")
        return None

    logger.debug(f"Extracted routed_experts from rollout_details: {len(routed_experts)} turns")
    if _r3_tensor_capture_enabled():
        # Fix B capture repackage: tensorize each turn's [gen_len, L, K] to a
        # contiguous np.int16 array HERE, at the earliest capture point, so every
        # downstream Ray crossing ships it out-of-band (no nested-list msgpack
        # deserialize). Empty/degenerate turns pass through untouched so the
        # sentinel-shape learning downstream is unchanged.
        out = []
        for turn_re in routed_experts:
            if turn_re is not None and len(turn_re) > 0:
                out.append(_as_routed_experts_array(turn_re))
            else:
                out.append(turn_re)
        return out
    return routed_experts


SENTINEL_EXPERT_ID = 0  # sentinel for unmatched / non-generated token rows in routed_experts

# Fix B: int16 array dtype for the routed-experts carrier (512 experts -> max id 511
# needs 9 bits, so uint8 overflows; int16 is near-minimal and matches the collator's
# deterministic narrow). See _r3_tensor_capture_enabled.
_ROUTED_EXPERTS_ARRAY_DTYPE = np.int16


def _r3_tensor_capture_enabled() -> bool:
    """Fix B (SKYRL_R3_TENSOR_CAPTURE, default OFF) — carry ``routed_experts`` as
    ``np.int16`` arrays end-to-end instead of nested ``List[List[List[List[int]]]]``.

    When OFF (default) every routed-experts code path below is BYTE-IDENTICAL to the
    historical nested-list behavior (the array branches are never taken). When ON,
    each per-token ``[L, K]`` row is an ``np.int16`` array from the capture boundary
    onward, so Ray ships it OUT-OF-BAND (zero-copy, GIL-releasing memcpy) — killing
    the driver-side ``_deserialize_msgpack_data`` GIL-hold that both slows the 80B
    forward and root-causes the #6936 gs1 watchdog stall. The collated
    ``[B, resp_len, L, K]`` int16 tensor is bit-for-bit identical either way (only the
    CONTAINER TYPE of the ids on the wire changes). Mirrors the existing
    ``SKYRL_R3_RESIDENT`` / ``SKYRL_R3_DECENTRAL`` env-flag discipline.
    """
    return os.environ.get("SKYRL_R3_TENSOR_CAPTURE", "0") == "1"


def _as_routed_experts_array(turn_re: Any) -> "np.ndarray":
    """Coerce ONE turn's nested ``[gen_len, L, K]`` routed-experts list to a
    contiguous ``np.int16`` array (Fix B capture repackage). Idempotent on arrays."""
    arr = np.asarray(turn_re, dtype=_ROUTED_EXPERTS_ARRAY_DTYPE)
    return np.ascontiguousarray(arr)


def align_routed_experts_with_lcs(
    retokenized_ids: List[int],
    vllm_routed_experts: List[Any],
    tokenizer,
    vllm_token_strings: Optional[List[str]] = None,
) -> List[List[List[int]]]:
    """Align vLLM per-token ``routed_experts`` rows to re-tokenized IDs via LCS.

    Mirror of :func:`align_logprobs_with_lcs`, but each per-token element is a
    ``[L, K]`` VECTOR (MoE-layer x top-k expert indices) rather than a scalar
    logprob. ``routed_experts`` is 1:1 with the vLLM response tokens — exactly the
    same index space as the per-token logprobs — so when the vLLM token strings are
    available (``vllm_token_strings``, from the parallel logprob dicts) we run the
    IDENTICAL ``SequenceMatcher.get_matching_blocks()`` LCS used by
    ``align_logprobs_with_lcs`` and copy the whole ``[L, K]`` row for each matched
    position. Unmatched positions get a sentinel ``[L, K]`` row (all
    ``SENTINEL_EXPERT_ID``).

    When token strings are unavailable, the exact 1:1 count case (same tokenizer —
    the production / smoke path) is a direct copy; differing counts fall back to a
    positional-index LCS proxy.

    Args:
        retokenized_ids: Token IDs from re-tokenizing the response text.
        vllm_routed_experts: Per-token routed-experts rows from vLLM, each a
            ``[L, K]`` nested list (length == number of vLLM tokens).
        tokenizer: HuggingFace tokenizer used for re-tokenization.
        vllm_token_strings: Optional per-token vLLM token strings (same order as
            ``vllm_routed_experts``) used to share the logprob LCS map.

    Returns:
        List of ``[L, K]`` rows aligned to ``retokenized_ids`` (one per token).
        Unmatched tokens get a sentinel ``[L, K]`` row.
    """
    if _r3_tensor_capture_enabled() and isinstance(vllm_routed_experts, np.ndarray):
        # Fix B array path: identical LCS/positional alignment + sentinel placement,
        # expressed as np.int16 array-row slice-assign instead of list-row copy. The
        # returned [n_retok, L, K] array's .tolist() is bit-identical to the list
        # branch's output (see test_align_flag_parity).
        return _align_routed_experts_with_lcs_array(retokenized_ids, vllm_routed_experts, tokenizer, vllm_token_strings)

    if not vllm_routed_experts:
        # No routed_experts to align — caller sentinel-pads; return [] so the
        # per-turn extend uses a sentinel block sized to the generated tokens.
        return []

    if not retokenized_ids:
        return []

    # Infer the [L, K] shape from the first vLLM row so the sentinel matches.
    sentinel_row = _sentinel_routed_experts_row(vllm_routed_experts[0])
    aligned = [list(sentinel_row) for _ in range(len(retokenized_ids))]

    n_vllm = len(vllm_routed_experts)
    n_retok = len(retokenized_ids)

    if vllm_token_strings is not None and len(vllm_token_strings) == n_vllm:
        # Faithful mirror of align_logprobs_with_lcs: LCS over token strings,
        # copy the [L, K] row instead of a scalar.
        retok_strings = tokenizer.convert_ids_to_tokens(retokenized_ids)
        matcher = SequenceMatcher(None, retok_strings, vllm_token_strings)
        for a_start, b_start, size in matcher.get_matching_blocks():
            for i in range(size):
                aligned[a_start + i] = vllm_routed_experts[b_start + i]
        return aligned

    if n_vllm == n_retok:
        # Exact 1:1 — common case (same tokenizer). Direct copy.
        for i in range(n_retok):
            aligned[i] = vllm_routed_experts[i]
        return aligned

    # No token strings and counts differ: positional-index LCS proxy (routed_experts
    # shares the vLLM response-token index space).
    matcher = SequenceMatcher(None, list(range(n_retok)), list(range(n_vllm)))
    matched_any = False
    for a_start, b_start, size in matcher.get_matching_blocks():
        for i in range(size):
            aligned[a_start + i] = vllm_routed_experts[b_start + i]
            matched_any = True
    if not matched_any:
        logger.debug(f"routed_experts LCS: no positional match (retok={n_retok}, vLLM={n_vllm}); all rows sentinel.")
    return aligned


def _align_routed_experts_with_lcs_array(
    retokenized_ids: List[int],
    vllm_routed_experts: "np.ndarray",
    tokenizer,
    vllm_token_strings: Optional[List[str]] = None,
) -> Any:
    """Fix B array twin of :func:`align_routed_experts_with_lcs` — SAME alignment
    semantics (LCS-over-token-strings / exact 1:1 direct copy / positional-index LCS
    proxy) and SAME sentinel placement, on an ``np.int16`` ``[n_vllm, L, K]`` input,
    returning an ``np.int16`` ``[n_retok, L, K]`` array (row-for-row identical to the
    list branch). Empty inputs return ``[]`` exactly like the list branch so the
    caller's sentinel-fallback fires identically."""
    n_vllm = int(vllm_routed_experts.shape[0]) if vllm_routed_experts.ndim >= 1 else 0
    if n_vllm == 0:
        return []
    if not retokenized_ids:
        return []

    # Infer [L, K] from the first vLLM row; the sentinel canvas is zeros of that shape.
    sentinel_row = _sentinel_routed_experts_row(vllm_routed_experts[0])  # np.int16 [L, K]
    L, K = int(sentinel_row.shape[0]), int(sentinel_row.shape[1])
    n_retok = len(retokenized_ids)
    aligned = np.zeros((n_retok, L, K), dtype=_ROUTED_EXPERTS_ARRAY_DTYPE)

    if vllm_token_strings is not None and len(vllm_token_strings) == n_vllm:
        retok_strings = tokenizer.convert_ids_to_tokens(retokenized_ids)
        matcher = SequenceMatcher(None, retok_strings, vllm_token_strings)
        for a_start, b_start, size in matcher.get_matching_blocks():
            if size:
                aligned[a_start : a_start + size] = vllm_routed_experts[b_start : b_start + size]
        return aligned

    if n_vllm == n_retok:
        # Exact 1:1 — direct copy (same values/rows as the list branch's per-index copy).
        return np.ascontiguousarray(vllm_routed_experts.astype(_ROUTED_EXPERTS_ARRAY_DTYPE, copy=True))

    matcher = SequenceMatcher(None, list(range(n_retok)), list(range(n_vllm)))
    matched_any = False
    for a_start, b_start, size in matcher.get_matching_blocks():
        if size:
            aligned[a_start : a_start + size] = vllm_routed_experts[b_start : b_start + size]
            matched_any = True
    if not matched_any:
        logger.debug(f"routed_experts LCS: no positional match (retok={n_retok}, vLLM={n_vllm}); all rows sentinel.")
    return aligned


def _sentinel_routed_experts_row(template_row: Any) -> Any:
    """Build a sentinel ``[L, K]`` row matching the shape of ``template_row``.

    Fix B: when ``template_row`` is an ``np.ndarray`` (the array-carrier path), the
    sentinel is a zeros ``np.int16`` array of the SAME ``[L, K]`` shape, so every
    downstream row stays an array (out-of-band shippable). Values (all
    ``SENTINEL_EXPERT_ID``) and shape are identical to the nested-list sentinel."""
    if isinstance(template_row, np.ndarray):
        return np.zeros(template_row.shape, dtype=_ROUTED_EXPERTS_ARRAY_DTYPE)
    # template_row is a [L, K] nested list. Mirror its L x K shape with sentinels.
    if not isinstance(template_row, (list, tuple)) or len(template_row) == 0:
        # Degenerate / unknown shape — fall back to a single [1, 1] sentinel.
        return [[SENTINEL_EXPERT_ID]]
    sentinel = []
    for layer in template_row:
        if isinstance(layer, (list, tuple)):
            sentinel.append([SENTINEL_EXPERT_ID] * len(layer))
        else:
            sentinel.append([SENTINEL_EXPERT_ID])
    return sentinel


def _re_sentinel_rows(n: int, sentinel_row: Optional[List[List[int]]]) -> List[List[List[int]]]:
    """Return ``n`` copies of a sentinel ``[L, K]`` routed_experts row.

    If the ``[L, K]`` shape has not been learned yet (no real row seen), fall back
    to a degenerate ``[[SENTINEL_EXPERT_ID]]`` row; the collator infers the true
    ``[L, K]`` from whichever sample first carries real routing and pads the rest.
    """
    if n <= 0:
        return []
    if isinstance(sentinel_row, np.ndarray):
        # Fix B: contiguous [n, L, K] int16 sentinel block. Consumers .extend() /
        # slice-assign this, which iterates axis-0 into per-token [L, K] array rows
        # — identical rows/values to the nested-list list-of-copies below.
        return np.broadcast_to(sentinel_row, (n,) + sentinel_row.shape).astype(_ROUTED_EXPERTS_ARRAY_DTYPE, copy=True)
    if sentinel_row is None:
        sentinel_row = [[SENTINEL_EXPERT_ID]]
    return [list(sentinel_row) for _ in range(n)]


def _tis_splice_enabled() -> bool:
    """Unified TIS served-id splice policy (deslop stage 2). Default ON.

    Uses vLLM's raw served ``completion_token_ids`` as the generated (loss_mask==1)
    region so TIS tier-1 exact-by-id alignment holds, closing the residual think-block
    re-tokenization divergence. The GENERALIZED served-id splice supersedes the
    qwen3_5 empty-think special case; this single gate arms both blocks (which are
    mutually exclusive by construction — empty-think fires only when the template
    renders the empty ``<think></think>``, the generalized path handles the rest).

    Byte-identical for turns whose guards don't match (non-thinking models: the
    re-tokenized turn already equals the served stream, so splicing is a no-op). The
    canonical knob is ``SKYRL_TIS_SPLICE``; the two legacy names
    (``SKYRL_TIS_SERVED_ID_SPLICE``, ``SKYRL_QWEN3_5_TIS_SPLICE``) are still honored as
    overrides. Set any of them to a falsey value to disable.
    """
    for var in ("SKYRL_TIS_SPLICE", "SKYRL_TIS_SERVED_ID_SPLICE", "SKYRL_QWEN3_5_TIS_SPLICE"):
        val = os.environ.get(var)
        if val is not None:
            return val in ("1", "true", "True")
    return True


def _tito_full_enabled(use_tis: bool = False, tito_full: Optional[bool] = None) -> bool:
    """Full token-in-token-out assembly policy — AUTO-defaults to ``use_tis``.

    When ON *and* Harbor's per-turn ``prompt_token_ids`` are available, the
    trajectory ``response_ids`` / ``loss_mask`` / ``rollout_logprobs`` are assembled
    directly from the exact served id streams (``prompt_token_ids`` for the masked
    context, ``completion_token_ids`` for the generated region) with NO
    re-tokenization of the multi-turn context. This is an ADDITIVE SUPERSET of the
    served-id splice (``_tis_splice_enabled``): the splice already makes the
    *generated region* exact-by-construction; TITO-full additionally makes the
    *masked context* byte-exact to what the inference engine served, closing the
    residual BPE-boundary re-tokenization drift of prior assistant turns fed back as
    text (Stage 0 catalogue).

    Resolution precedence (params come from the caller's cfg — no globals):
      1. ``SKYRL_TITO_FULL`` env var — if set, WINS (quick override / testing escape
         hatch). Truthy ⇒ ON, else OFF.
      2. else the EXPLICIT config flag ``tito_full`` (``trainer.algorithm.tito_full``)
         — if not ``None`` (an explicit True/False), use it verbatim.
      3. else (auto / unset) — DEFAULT TO ``use_tis`` (``trainer.algorithm.use_tis``):
         TITO-full ON whenever TIS is on, OFF otherwise.

    Non-TIS byte-identical guarantee: ``use_tis=False`` with no explicit flag/env ⇒
    returns ``False`` ⇒ every existing code path untouched (``torch.equal``). Mirrors
    the EP/CP/splice flag-off byte-identical scaffold discipline.
    """
    val = os.environ.get("SKYRL_TITO_FULL")
    if val is not None:
        return val in ("1", "true", "True")
    if tito_full is not None:
        return bool(tito_full)
    return bool(use_tis)


def _normalize_candidate_logprobs(candidate_logprobs):
    """Split a per-turn logprob list into (token_strings_or_None, float_logprobs).

    Mirrors the normalization inside the main assembly loop: Harbor's canonical
    format is plain floats; the dict format (``{"token", "logprob"}``) only carries
    token strings for the LCS fallback.
    """
    token_strings = None
    if len(candidate_logprobs) > 0 and isinstance(candidate_logprobs[0], dict):
        if "token" in candidate_logprobs[0]:
            token_strings = [lp.get("token") for lp in candidate_logprobs]
        floats = [lp.get("logprob", 0.0) for lp in candidate_logprobs]
    else:
        floats = list(candidate_logprobs)
    return token_strings, floats


def _assemble_response_ids_tito_full(
    messages,
    tokenizer,
    generation_prompt_ids,
    assistant_logprobs,
    assistant_token_ids,
    assistant_prompt_token_ids,
    assistant_routed_experts,
    alignment_stats,
    custom_chat_template,
    chat_template_kwargs,
):
    """Full-TITO assembly of ``response_ids``/``loss_mask``/logprobs/routed_experts.

    Builds the ENTIRE trajectory from Harbor's exact per-turn served id streams — NO
    re-tokenization of the multi-turn context — so the MASKED context is byte-exact to
    what the inference engine served and the generated region is the exact sampled ids.

    Uses the served-stream prefix invariant
    ``prompt_token_ids[t] == prompt_token_ids[t-1] + completion_token_ids[t-1] + observation[t-1]``:
    the full served stream is ``prompt_token_ids[-1] + completion_token_ids[-1]``, and turn
    ``t``'s generated (loss_mask==1) region sits at offset ``len(prompt_token_ids[t])`` with
    length ``len(completion_token_ids[t])``. ``response_ids`` is that served stream minus the
    initial prompt (system+user0, no generation prompt), which by construction ends exactly at
    ``len(prompt_token_ids[0]) - len(generation_prompt_ids)``.

    FAIL-LOUD, NEVER-GUESS: returns ``None`` (caller falls back to the re-tok + splice path,
    recording nothing lost) whenever the streams are absent/short/inconsistent or the invariant
    does not hold — so a malformed capture degrades to today's behavior instead of assembling a
    wrong sequence. On success returns ``(response_ids, loss_mask, rollout_logprobs,
    rollout_routed_experts)`` (the last two ``None`` when their inputs were ``None``).
    """
    if assistant_prompt_token_ids is None or assistant_token_ids is None:
        return None
    n_turns = len(assistant_token_ids)
    if n_turns == 0 or len(assistant_prompt_token_ids) != n_turns:
        return None
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if len(assistant_msgs) != n_turns:
        return None
    # Every turn must carry non-empty prompt + completion id streams.
    for t in range(n_turns):
        p = assistant_prompt_token_ids[t]
        c = assistant_token_ids[t]
        if not p or not isinstance(p, list) or not c or not isinstance(c, list):
            return None
    # Prefix invariant across turns.
    for t in range(1, n_turns):
        prev = list(assistant_prompt_token_ids[t - 1]) + list(assistant_token_ids[t - 1])
        cur = list(assistant_prompt_token_ids[t])
        if cur[: len(prev)] != prev:
            return None
    gp = list(generation_prompt_ids)
    gp_len = len(gp)
    p0 = list(assistant_prompt_token_ids[0])
    initial_prompt_len = len(p0) - gp_len
    # Turn-0 prompt must end with the generation prompt (fixes the response/prompt boundary).
    if initial_prompt_len < 0 or p0[initial_prompt_len:] != gp:
        return None
    served_full = list(assistant_prompt_token_ids[-1]) + list(assistant_token_ids[-1])
    # Every completion region must sit at its expected offset in the served stream.
    for t in range(n_turns):
        off = len(assistant_prompt_token_ids[t])
        comp = list(assistant_token_ids[t])
        if served_full[off : off + len(comp)] != comp:
            return None

    response_ids = list(served_full[initial_prompt_len:])
    total_len = len(response_ids)
    loss_mask = [0] * total_len
    rollout_logprobs = None if assistant_logprobs is None else [0.0] * total_len

    # routed_experts sentinel [L, K] shape learned up-front.
    rollout_routed_experts = None
    _re_sentinel_row = None
    if assistant_routed_experts is not None:
        rollout_routed_experts = [None] * total_len  # placeholder; sentinel-filled below
        for _turn_re in assistant_routed_experts:
            if _turn_re is not None and len(_turn_re) > 0:
                _re_sentinel_row = _sentinel_routed_experts_row(_turn_re[0])
                break

    for t in range(n_turns):
        start = len(assistant_prompt_token_ids[t]) - initial_prompt_len
        comp = list(assistant_token_ids[t])
        end = start + len(comp)
        region_tokens = response_ids[start:end]
        for i in range(start, end):
            loss_mask[i] = 1

        if assistant_logprobs is not None:
            alignment_stats.n_messages += 1
            alignment_stats.n_tokens += len(comp)
            msg_logprobs = None
            if t < len(assistant_logprobs):
                _, floats = _normalize_candidate_logprobs(assistant_logprobs[t])
                # Exact by construction: region_tokens == comp (served ids).
                msg_logprobs = align_logprobs_by_token_ids(region_tokens, comp, floats, stats=alignment_stats)
            if msg_logprobs is None:
                # Should not happen given the invariant checks; record loudly.
                alignment_stats.n_unaligned += len(comp)
                alignment_stats.n_failed_messages += 1
                msg_logprobs = [0.0] * len(comp)
            rollout_logprobs[start:end] = msg_logprobs

        if assistant_routed_experts is not None:
            msg_re = None
            if t < len(assistant_routed_experts):
                candidate_re = assistant_routed_experts[t]
                if candidate_re is not None and len(candidate_re) > 0:
                    if _re_sentinel_row is None and len(candidate_re) > 0:
                        _re_sentinel_row = _sentinel_routed_experts_row(candidate_re[0])
                    vllm_token_strings = None
                    if assistant_logprobs and t < len(assistant_logprobs):
                        strs, _ = _normalize_candidate_logprobs(assistant_logprobs[t])
                        vllm_token_strings = strs
                    # Stage 3 keeps R3 on the existing LCS aligner (Stage 4 adds the
                    # exact-by-id path); routed_experts rides the same region.
                    msg_re = align_routed_experts_with_lcs(
                        region_tokens, candidate_re, tokenizer, vllm_token_strings=vllm_token_strings
                    )
            if msg_re is None or len(msg_re) != len(comp):
                msg_re = _re_sentinel_rows(len(comp), _re_sentinel_row)
            rollout_routed_experts[start:end] = msg_re

    # Fill any remaining (masked / non-generated) routed_experts positions with sentinels.
    if rollout_routed_experts is not None:
        for i in range(total_len):
            if rollout_routed_experts[i] is None:
                rollout_routed_experts[i] = _re_sentinel_row if _re_sentinel_row is not None else [[SENTINEL_EXPERT_ID]]

    # Byte-parity tail: the re-tok path appends the FINAL assistant turn's trailing
    # template tokens after its EOS (e.g. the ``\n`` after ``<|im_end|>``). The served
    # stream has no next prompt to contain them, so recover them from a single re-tok of
    # the last assistant message and append (masked) — makes the clean case byte-identical
    # to the re-tok path while the body stays exact-from-served-ids.
    last_msg = assistant_msgs[-1]
    cur = encode_messages_subset([last_msg], tokenizer, custom_chat_template, chat_template_kwargs=chat_template_kwargs)
    if tokenizer.eos_token_id in cur:
        _le = len(cur) - 1 - cur[::-1].index(tokenizer.eos_token_id)
        trailing = cur[_le + 1 :]
    else:
        trailing = []
    if trailing:
        response_ids.extend(trailing)
        loss_mask.extend([0] * len(trailing))
        if rollout_logprobs is not None:
            rollout_logprobs.extend([0.0] * len(trailing))
        if rollout_routed_experts is not None:
            rollout_routed_experts.extend(_re_sentinel_rows(len(trailing), _re_sentinel_row))

    assert len(loss_mask) == len(response_ids)
    assert rollout_logprobs is None or len(rollout_logprobs) == len(response_ids)
    assert rollout_routed_experts is None or len(rollout_routed_experts) == len(response_ids)
    return response_ids, loss_mask, rollout_logprobs, rollout_routed_experts


def get_response_ids_and_loss_mask_from_messages(
    messages: ConversationType,
    tokenizer,
    assistant_logprobs=None,
    custom_chat_template=None,
    assistant_routed_experts=None,
    assistant_token_ids=None,
    alignment_stats: Optional["AlignmentStats"] = None,
    chat_template_kwargs=None,
    assistant_prompt_token_ids=None,
    use_tis: bool = False,
    tito_full: Optional[bool] = None,
):
    """
    Get the response ids and loss mask from a list of messages.

    We encode each message one by one, using a fixed base approach, building response token IDs, loss mask,
    and rollout logprobs if provided. For Qwen3, this function will keep all the thinking tokens from the messages.

    TIS logprob alignment (robust, two-tier):
      1. EXACT path (preferred): when ``assistant_token_ids`` (Harbor's per-turn
         ``completion_token_ids``) is provided AND matches the re-tokenized
         generated tokens for that turn, the per-turn logprob floats are zipped
         on 1:1 — positions are exact by construction, NO guessing.
      2. LCS fallback (last resort): when the exact path is unavailable (no
         token ids) or the ids diverge, fall back to LCS string matching. Every
         fallback is RECORDED in ``alignment_stats`` (and logged at WARNING) so
         it surfaces as ``tis/lcs_fallback_fraction`` and never silently degrades.

    Args:
        messages: List of message dicts with 'role' and 'content' keys. Must contain at least
                 one message.
        tokenizer: HuggingFace tokenizer with chat_template support and eos_token_id defined.
        assistant_logprobs: Optional per-assistant-message logprobs. Two formats:
            - float format (canonical Harbor): [[float, ...], ...]
            - dict format (token strings): [[{"token": str, "logprob": float}, ...], ...]
        custom_chat_template: Optional custom chat template string to use instead of tokenizer's default.
        assistant_token_ids: Optional per-assistant-message exact vLLM token ids
            (Harbor ``completion_token_ids``), index-aligned with assistant_logprobs.
            Enables the EXACT alignment path.
        alignment_stats: Optional AlignmentStats accumulator the caller can read
            afterward to emit tis/* metrics. A local one is created if None.
        chat_template_kwargs: Optional dict forwarded to apply_chat_template when
            building the generation prompt / re-tokenizing turns (e.g.
            ``{"enable_thinking": True}`` for Qwen3.5/3.6 so the re-tokenized
            generation prompt matches the served stream). ``None`` -> byte-identical
            old behavior.

    Returns:
        Tuple of (response ids, loss mask, rollout logprobs[, rollout routed experts]).
        When ``assistant_routed_experts`` is None a 3-tuple is returned (back-compat);
        otherwise a 4-tuple. Per-turn alignment counts are written into
        ``alignment_stats`` (pass one in to read them).
    """
    assert len(messages), "messages list cannot be empty"
    if alignment_stats is None:
        alignment_stats = AlignmentStats()

    # Needed to correctly mask it zero for assistant messages.
    # chat_template_kwargs (e.g. {"enable_thinking": True}) is forwarded so the
    # re-tokenized generation prompt MATCHES the served vLLM stream. For Qwen3.5/3.6
    # this renders the bare <think> open instead of the empty-think block default
    # (see get_generation_prompt_ids / detect_qwen3_5_empty_think_prefix). None ->
    # byte-identical old behavior (empty-think default -> the splice backstop fires).
    generation_prompt_ids = get_generation_prompt_ids(
        tokenizer, custom_chat_template=custom_chat_template, chat_template_kwargs=chat_template_kwargs
    )

    # --- Full TITO assembly (default OFF, additive superset of the splice) ---
    # When SKYRL_TITO_FULL is on AND Harbor's per-turn prompt_token_ids are present,
    # assemble the WHOLE trajectory from the exact served id streams (no re-tok of the
    # multi-turn context), making the MASKED context byte-exact to what the engine
    # served. Fails loud → None → falls through to the re-tok + splice path below
    # (byte-identical), so a malformed capture never yields a wrong sequence. Default
    # OFF ⇒ this block is skipped entirely (byte-identical to prior behavior).
    if (
        _tito_full_enabled(use_tis=use_tis, tito_full=tito_full)
        and assistant_prompt_token_ids is not None
        and assistant_token_ids is not None
    ):
        _tito = _assemble_response_ids_tito_full(
            messages,
            tokenizer,
            generation_prompt_ids,
            assistant_logprobs,
            assistant_token_ids,
            assistant_prompt_token_ids,
            assistant_routed_experts,
            alignment_stats,
            custom_chat_template,
            chat_template_kwargs,
        )
        if _tito is not None:
            _rids, _lmask, _rlp, _rre = _tito
            if assistant_routed_experts is None:
                return _rids, _lmask, _rlp
            return _rids, _lmask, _rlp, _rre
        else:
            logger.warning(
                "SKYRL_TITO_FULL on but prompt-id assembly declined (missing/inconsistent "
                "served id streams or prefix invariant failed); falling back to re-tok + splice."
            )

    # ARCH-GATED (qwen3_5/3.6 only): the Qwen3.5/3.6 chat template injects an EMPTY
    # ``<think>\n\n</think>`` block into the assistant generation prompt, which the
    # served ``completion_token_ids`` do NOT contain (the model emits its own real
    # think block). That breaks the gen-prompt prefix-strip and makes the
    # re-tokenized generated region diverge from the served stream → TIS tier-1
    # collapse. When detected AND the served per-turn ids are available, we SPLICE
    # the served ids in as the generated region (exact by construction) instead of
    # re-tokenizing. ``qwen3_5_assistant_prefix_ids`` is the real
    # ``<|im_start|>assistant\n`` prefix (BEFORE the injected empty think block).
    # Returns None for every non-qwen3_5 tokenizer → byte-identical old behavior,
    # consistent with the aa11512 qwen3_5 arch-gating. Gated by the unified
    # ``_tis_splice_enabled()`` policy (default ON; disable via SKYRL_TIS_SPLICE=0).
    qwen3_5_assistant_prefix_ids = None
    if assistant_token_ids is not None and _tis_splice_enabled():
        qwen3_5_assistant_prefix_ids = detect_qwen3_5_empty_think_prefix(tokenizer, generation_prompt_ids)

    # 1. Initalize the things to accumulate
    response_ids = []
    loss_mask = []
    rollout_logprobs = None if assistant_logprobs is None else []
    # routed_experts rides the SAME per-token / per-turn index space as logprobs.
    # Each accumulated element is a [L, K] row; user/prefix/post-EOS rows are
    # sentinel-filled (see align_routed_experts_with_lcs / SENTINEL_EXPERT_ID).
    rollout_routed_experts = None if assistant_routed_experts is None else []
    # Sentinel [L, K] shape — learned UP-FRONT by scanning assistant_routed_experts
    # for the first real per-token row, so that sentinel rows emitted BEFORE the
    # first generated token (e.g. a leading user message) already have the correct
    # [L, K] width. Otherwise a single sample could mix [1, 1] and [L, K] rows and
    # break the dense torch.tensor() collation.
    _re_sentinel_row = None
    if assistant_routed_experts is not None:
        for _turn_re in assistant_routed_experts:
            if _turn_re is not None and len(_turn_re) > 0:
                _re_sentinel_row = _sentinel_routed_experts_row(_turn_re[0])
                break
    assistant_msg_idx = 0

    for i in range(len(messages)):
        # 2. Use fixed base approach to encode the message and accumulate
        cur_message = messages[i]
        cur_token_ids = encode_messages_subset(
            [cur_message], tokenizer, custom_chat_template, chat_template_kwargs=chat_template_kwargs
        )

        # 3. Set loss mask and rollout logprobs.
        # Regardless of the message role, each message is responsible for adding its own generation
        # prompt, and we apply the correct masking.
        if cur_message["role"] in ("user", "tool", "system"):
            # 3.1. For user / tool-result / system messages, the mask is simply
            # zeros: these are observations the policy CONDITIONS ON but is not
            # trained on. Agentic (opencode) rollouts interleave role="tool"
            # tool-result messages between assistant turns; the chat template
            # renders them, so they occupy real, masked positions in the training
            # sequence — exactly like user turns. Silently dropping them here
            # (the old `else: raise`) EXCLUDED every tool-using trajectory from the
            # batch, starving keep-1: v26 gs1 had avg_num_tokens=3.34, reward=0.0.
            response_ids.extend(cur_token_ids)
            loss_mask.extend([0] * len(cur_token_ids))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(cur_token_ids))
            if assistant_routed_experts is not None:
                rollout_routed_experts.extend(_re_sentinel_rows(len(cur_token_ids), _re_sentinel_row))
        elif cur_message["role"] == "assistant":
            # 3.2. For assistant messages, we need to separate out:
            # 1) generation prompt IDs -- mask is 0
            # 2) tokens actually generated by the assistant (including the EOS) -- mask is 1
            # 3) tokens after the EOS token (the `\n` in Qwen models) -- mask is 0
            prefix_len = len(generation_prompt_ids)
            prefix_matches = cur_token_ids[:prefix_len] == generation_prompt_ids

            # --- ARCH-GATED qwen3_5/3.6 served-id splice (TIS exact-by-construction) ---
            # When the qwen3_5 empty-think generation prompt is in play AND we have the
            # served completion_token_ids for this turn, REBUILD this assistant turn so
            # the generated (loss_mask==1) region IS the served stream verbatim — no
            # re-tokenization, no divergence. The prefix is the real
            # ``<|im_start|>assistant\n`` (before the template's injected empty
            # ``<think></think>``) and tokens_after_eos is the trailing ``\n`` from the
            # re-tokenized turn. This makes tier-1 align_logprobs_by_token_ids exact by
            # construction (generated_token_ids == served ids) and dodges the qwen3_5
            # prefix-mismatch fallback that was zeroing logprobs. Non-qwen3_5 paths
            # (qwen3_5_assistant_prefix_ids is None) skip this block entirely.
            spliced = False
            if (
                qwen3_5_assistant_prefix_ids is not None
                and assistant_token_ids is not None
                and assistant_msg_idx < len(assistant_token_ids)
            ):
                served_ids = assistant_token_ids[assistant_msg_idx]
                if served_ids and isinstance(served_ids, list):
                    real_prefix = list(qwen3_5_assistant_prefix_ids)
                    rp = len(real_prefix)
                    # The re-tokenized turn must start with the real assistant prefix.
                    if cur_token_ids[:rp] == real_prefix:
                        # Recover the trailing template tokens AFTER the served content
                        # (e.g. ``<|im_end|>\n``). The re-tokenized turn ends with the
                        # SAME trailing template, so take whatever follows the last EOS.
                        if tokenizer.eos_token_id in cur_token_ids:
                            _le = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(tokenizer.eos_token_id)
                            trailing = cur_token_ids[_le + 1 :]
                        else:
                            trailing = []
                        # The served stream already ends in EOS (vLLM emits it); if it
                        # somehow does not, the trailing template still masks correctly.
                        prefix_len = rp
                        generated_token_ids = list(served_ids)
                        tokens_after_eos = list(trailing)
                        cur_token_ids = real_prefix + generated_token_ids + tokens_after_eos
                        spliced = True

            # --- GENERALIZED served-id splice (Fix A, env-gated) — SUPERSET of the
            # empty-think splice above. Fires PER-TURN whenever the served
            # completion_token_ids are present AND the re-tokenized turn starts with
            # the real generation-prompt prefix (``prefix_matches``) — DECOUPLED from
            # empty-think detection. This closes the residual off-by-1..4 think-block
            # whitespace divergence that survives the enable_thinking=True root-fix:
            # under enable_thinking=True ``detect_qwen3_5_empty_think_prefix`` returns
            # None (so the block above never fires), yet the template still
            # canonicalizes think-block newlines at re-tok time → re-tok token count
            # drifts from the served stream → TIS tier-1 exact-by-id match fails →
            # logprobs zeroed. Here ``generation_prompt_ids`` already matches the
            # served-stream boundary (get_generation_prompt_ids forwards
            # chat_template_kwargs), so the served ids ARE exactly the tokens that
            # followed the generation prompt at rollout time. Using them VERBATIM as
            # the generated (loss_mask==1) region makes tier-1 exact by construction
            # (generated_token_ids == served ids) and trains on the exact sampled
            # tokens. Trailing template tokens (e.g. ``\n`` after the served EOS) are
            # recovered from the last EOS of the re-tokenized turn, mirroring the
            # empty-think splice. Fully gated behind SKYRL_TIS_SERVED_ID_SPLICE=1;
            # with the env unset this entire block is skipped → BYTE-IDENTICAL to the
            # prior behavior (the empty-think splice above is independently gated by
            # SKYRL_QWEN3_5_TIS_SPLICE and is untouched here).
            # NOTE (deslop stage 2): gated by the unified _tis_splice_enabled()
            # policy (default ON — this generalized served-id path SUPERSEDES the
            # empty-think special case above). Byte-identical for turns whose guards
            # (prefix_matches + served ids present) don't match, i.e. the common
            # non-thinking case where re-tok already equals the served stream.
            if (
                not spliced
                and _tis_splice_enabled()
                and prefix_matches
                and assistant_token_ids is not None
                and assistant_msg_idx < len(assistant_token_ids)
            ):
                served_ids = assistant_token_ids[assistant_msg_idx]
                if served_ids and isinstance(served_ids, list):
                    # Recover the trailing template tokens AFTER the served content
                    # (e.g. ``<|im_end|>\n`` → the ``\n``) from the re-tokenized turn,
                    # which ends with the SAME trailing template.
                    if tokenizer.eos_token_id in cur_token_ids:
                        _le = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(tokenizer.eos_token_id)
                        trailing = cur_token_ids[_le + 1 :]
                    else:
                        trailing = []
                    prefix_len = len(generation_prompt_ids)
                    generated_token_ids = list(served_ids)
                    tokens_after_eos = list(trailing)
                    cur_token_ids = list(generation_prompt_ids) + generated_token_ids + tokens_after_eos
                    spliced = True

            if not spliced:
                if not prefix_matches:
                    actual_prefix = cur_token_ids[:prefix_len]
                    logger.warning(
                        "Assistant message prefix mismatch (expected {}, got {}). "
                        "Falling back to treating the entire assistant message as generated tokens.",
                        generation_prompt_ids,
                        actual_prefix,
                    )
                    prefix_len = 0

                if tokenizer.eos_token_id in cur_token_ids:
                    last_eos_token_index = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(tokenizer.eos_token_id)
                    generated_token_ids = cur_token_ids[prefix_len : last_eos_token_index + 1]
                    tokens_after_eos = cur_token_ids[last_eos_token_index + 1 :]
                else:
                    generated_token_ids = cur_token_ids[prefix_len:]
                    tokens_after_eos = []

            # Now that cur_token_ids is finalized (re-tok or spliced), accumulate it.
            response_ids.extend(cur_token_ids)
            assert prefix_len + len(generated_token_ids) + len(tokens_after_eos) == len(cur_token_ids), (
                "The sum of the lengths of the generation prompt IDs, the generated tokens, and the tokens after the EOS token should equal the length of the current token IDs"
            )

            # 3.2.1. Add the generation prompt IDs.
            loss_mask.extend([0] * prefix_len)
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * prefix_len)
            if assistant_routed_experts is not None:
                rollout_routed_experts.extend(_re_sentinel_rows(prefix_len, _re_sentinel_row))

            # 3.2.2. Add what the assistant actually generated
            loss_mask.extend([1] * len(generated_token_ids))
            if assistant_logprobs:
                msg_logprobs = None
                alignment_stats.n_messages += 1
                alignment_stats.n_tokens += len(generated_token_ids)
                if assistant_msg_idx >= len(assistant_logprobs):
                    logger.warning(
                        "Missing logprobs for assistant message #{} (provided {} lists). "
                        "Proceeding with zeroed logprobs.",
                        assistant_msg_idx + 1,
                        len(assistant_logprobs),
                    )
                    alignment_stats.n_unaligned += len(generated_token_ids)
                    alignment_stats.n_failed_messages += 1
                else:
                    candidate_logprobs = assistant_logprobs[assistant_msg_idx]

                    # Normalize logprobs to (token_strings_or_None, float_logprobs).
                    # vLLM/Harbor canonical format is plain floats; the dict format
                    # (with token strings) only feeds the LCS fallback.
                    candidate_token_strings = None
                    if len(candidate_logprobs) > 0 and isinstance(candidate_logprobs[0], dict):
                        if "token" in candidate_logprobs[0]:
                            candidate_token_strings = [lp.get("token") for lp in candidate_logprobs]
                        candidate_float_logprobs = [lp.get("logprob", 0.0) for lp in candidate_logprobs]
                    else:
                        candidate_float_logprobs = list(candidate_logprobs)

                    # --- Tier 1: EXACT alignment by token id (preferred) ---
                    # Use Harbor's per-turn completion_token_ids when available.
                    candidate_ids = None
                    if assistant_token_ids is not None and assistant_msg_idx < len(assistant_token_ids):
                        candidate_ids = assistant_token_ids[assistant_msg_idx]
                    if candidate_ids is not None:
                        msg_logprobs = align_logprobs_by_token_ids(
                            generated_token_ids,
                            candidate_ids,
                            candidate_float_logprobs,
                            stats=alignment_stats,
                        )

                    # --- Tier 2: LCS fallback (last resort, always recorded) ---
                    if msg_logprobs is None:
                        if candidate_token_strings is not None:
                            # Reconstruct dict format for the LCS matcher.
                            dict_logprobs = [
                                {"token": t, "logprob": lp}
                                for t, lp in zip(candidate_token_strings, candidate_float_logprobs)
                            ]
                            msg_logprobs = align_logprobs_with_lcs(
                                generated_token_ids,
                                dict_logprobs,
                                tokenizer,
                                stats=alignment_stats,
                            )
                        elif len(candidate_float_logprobs) == len(generated_token_ids):
                            # No token strings, but counts match exactly: positional
                            # 1:1 (treat as exact — this is the float+no-ids case).
                            msg_logprobs = candidate_float_logprobs
                            alignment_stats.n_exact += len(generated_token_ids)
                        else:
                            # No token ids, no token strings, count mismatch: cannot
                            # align. Record the failure loudly instead of guessing.
                            logger.warning(
                                "TIS alignment FAILED for assistant message #{}: "
                                "logprob count ({}) != token count ({}), and no token ids/strings "
                                "available for fallback. Zeroing this message's logprobs.",
                                assistant_msg_idx + 1,
                                len(candidate_float_logprobs),
                                len(generated_token_ids),
                            )
                            alignment_stats.n_unaligned += len(generated_token_ids)
                            alignment_stats.n_failed_messages += 1

                rollout_logprobs.extend(msg_logprobs if msg_logprobs is not None else [0.0] * len(generated_token_ids))

            # 3.2.2b. Add the per-token routed_experts [L, K] rows for what the
            # assistant actually generated, aligned to the re-tokenized generated
            # tokens via LCS (mirrors the logprobs alignment above).
            if assistant_routed_experts is not None:
                msg_routed_experts = None
                if assistant_msg_idx < len(assistant_routed_experts):
                    candidate_re = assistant_routed_experts[assistant_msg_idx]
                    if candidate_re is not None and len(candidate_re) > 0:
                        # Lazily learn the [L, K] sentinel shape from the first real row.
                        if _re_sentinel_row is None and len(candidate_re) > 0:
                            _re_sentinel_row = _sentinel_routed_experts_row(candidate_re[0])
                        # Share the logprob LCS map: routed_experts rides the SAME
                        # vLLM response-token index space as the per-token logprobs,
                        # so reuse those token strings when present for an identical
                        # tokenizer-mismatch alignment.
                        vllm_token_strings = None
                        if assistant_logprobs and assistant_msg_idx < len(assistant_logprobs):
                            lp_candidate = assistant_logprobs[assistant_msg_idx]
                            if lp_candidate and isinstance(lp_candidate[0], dict) and "token" in lp_candidate[0]:
                                vllm_token_strings = [tl["token"] for tl in lp_candidate]
                        msg_routed_experts = align_routed_experts_with_lcs(
                            generated_token_ids,
                            candidate_re,
                            tokenizer,
                            vllm_token_strings=vllm_token_strings,
                        )
                else:
                    logger.warning(
                        "Missing routed_experts for assistant message #{} (provided {} lists). "
                        "Proceeding with sentinel rows.",
                        assistant_msg_idx + 1,
                        len(assistant_routed_experts),
                    )
                if msg_routed_experts is None or len(msg_routed_experts) != len(generated_token_ids):
                    msg_routed_experts = _re_sentinel_rows(len(generated_token_ids), _re_sentinel_row)
                rollout_routed_experts.extend(msg_routed_experts)

            # 3.2.3. Add the tokens after the EOS token.
            loss_mask.extend([0] * len(tokens_after_eos))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(tokens_after_eos))
            if assistant_routed_experts is not None:
                rollout_routed_experts.extend(_re_sentinel_rows(len(tokens_after_eos), _re_sentinel_row))

            assistant_msg_idx += 1
        else:
            raise ValueError(
                f"Expected message role to be 'user', 'assistant', 'tool', or 'system', got {cur_message['role']}"
            )

        assert len(loss_mask) == len(response_ids)
        assert len(rollout_logprobs) == len(response_ids) if rollout_logprobs is not None else True
        assert len(rollout_routed_experts) == len(response_ids) if rollout_routed_experts is not None else True

    if assistant_routed_experts is None:
        return response_ids, loss_mask, rollout_logprobs
    return response_ids, loss_mask, rollout_logprobs, rollout_routed_experts
