# This code is adapted from OpenRLHF and OpenReasonerZero
# https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/models.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

import os
from typing import Any, Dict, Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import numpy as np
from skyrl_train.distributed.ulysses.utils import ulysses_pad_and_slice_inputs, gather_outputs_and_unpad
from skyrl_train.distributed.cp_utils import maybe_cp_context, context_parallel_unshard, cp_unshard_grad_safe
from skyrl_train.utils.torch_utils import chunked_entropy_from_logits, logprobs_from_logits
from packaging.version import Version

# --- Stage 2 (FSDP2 CP): guarded flash-attn import ---------------------------
# The CP path runs through SDPA ring attention, not flash-attn varlen, so the
# environment that loads the model need NOT have flash-attn installed. Previously
# `from flash_attn.bert_padding import pad_input, unpad_input` was an
# unconditional module-level import that broke `import model_wrapper` in any
# env without flash-attn. We make it lazy: try the import; if it fails, bind
# `pad_input`/`unpad_input` to shims that raise ONLY if actually called (every
# call site is gated on `attn_implementation == "flash_attention_2"` or
# `use_sample_packing`, both of which are off on the sdpa/CP path). `_HAS_FLASH`
# records availability for tests / diagnostics.
try:
    from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401

    _HAS_FLASH = True
except ImportError:  # flash-attn not installed (e.g. the CP/sdpa-only env)
    _HAS_FLASH = False

    def _flash_missing(*args, **kwargs):
        raise ImportError(
            "flash_attn is not installed but a flash-attn-only code path "
            "(sample packing / pad_input / unpad_input) was invoked. Install "
            "flash-attn, or use attn_backend='sdpa'/'flex' with "
            "use_sample_packing=false (the CP path)."
        )

    def pad_input(*args, **kwargs):  # noqa: F811
        return _flash_missing(*args, **kwargs)

    def unpad_input(*args, **kwargs):  # noqa: F811
        return _flash_missing(*args, **kwargs)


def resolve_attn_implementation(
    attn_backend: str = "auto",
    use_flash_attention_2: bool = False,
    context_parallel_size: int = 1,
) -> str:
    """Resolve the HF `attn_implementation` string from the Stage-2 backend selector.

    `attn_backend` ∈ {"auto", "flash_attention_2", "sdpa", "flex"}:
      - "auto" (default) reproduces the pre-Stage-2 behavior EXACTLY:
        "flash_attention_2" if `use_flash_attention_2` else "eager" (G1 —
        every existing run stays byte-identical).
      - "flash_attention_2" / "sdpa" force that backend (overriding `flash_attn`).
      - "flex" maps to HF's "flex_attention".

    When CP is enabled (`context_parallel_size > 1`, the Stage-0 flag), the
    backend MUST be a ring-compatible non-varlen attention (sdpa/flex); flash
    attention varlen is rejected (G2). `auto`/`flash_attention_2` are rejected
    under CP; the caller must explicitly select sdpa/flex.
    """
    valid = {"auto", "flash_attention_2", "sdpa", "flex"}
    assert attn_backend in valid, f"attn_backend='{attn_backend}' is invalid; must be one of {sorted(valid)}"

    if attn_backend == "auto":
        impl = "flash_attention_2" if use_flash_attention_2 else "eager"
    elif attn_backend == "flex":
        impl = "flex_attention"
    else:
        impl = attn_backend  # "flash_attention_2" or "sdpa"

    if context_parallel_size > 1:
        assert impl in ("sdpa", "flex_attention"), (
            f"context_parallel_size={context_parallel_size} requires a ring-compatible "
            f"attention backend (attn_backend ∈ {{'sdpa','flex'}}); got attn_backend="
            f"'{attn_backend}' -> attn_implementation='{impl}'. Flash-attn varlen is not "
            "supported under context parallel (G2)."
        )
    return impl


def _cp_mask_dict_supported(model) -> bool:
    """Whether `model`'s forward accepts the per-layer-type mask DICT escape hatch.

    Under CP we must skip HF's 4D additive `create_causal_mask` (its kv axis gets
    sharded while q stays full → torch CP SDPA `aten.expand` failure). HF's dense
    Qwen3 path supports passing `attention_mask` ALREADY as a per-layer-type dict
    (modeling_qwen3.py:403 `if not isinstance(attention_mask, dict)`), which
    short-circuits `create_causal_mask`. But the MoE path (modeling_qwen3_moe.py:497)
    has NO dict handling: it feeds the arg straight into `create_causal_mask` →
    `attention_mask.ndim` → AttributeError on a dict. So MoE models must instead
    get `attention_mask=None` + monotonic position_ids (causality via SDPA
    `is_causal=True`; padding recovered post-hoc by the loss/entropy masks).

    Detection: route MoE architectures to the None path, everything else (dense
    Qwen3 etc.) to the proven dict path. Checked once at init and cached. We look
    at the underlying HF module class name and the config `model_type` (the model
    may be PEFT/FSDP-wrapped, so we probe both) — "moe" in either ⇒ not supported.
    """
    name = type(model).__name__.lower()
    cfg = getattr(model, "config", None)
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    return "moe" not in name and "moe" not in model_type


class HFModelWrapper(nn.Module):
    """
    Base class for wrapped HF models in reinforcement learning.

    This class serves as a foundation for implementing various model roles.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        exclude_modules (list, optional): List of modules to exclude from applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        # TODO(shu): combine all LoRA specific configs into one place?
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        exclude_modules=None,
        ds_config=None,
        device_map=None,
        temperature=1.0,
        use_liger_kernel=False,
        sequence_parallel_size=1,
        use_sample_packing: bool = False,
        use_torch_compile: bool = False,
        rope_scaling: Dict[str, Any] = {},
        rope_theta: float | None = None,
        moe_router_replay: bool = False,
        moe_grouped_gemm: bool = False,
        attn_backend: str = "auto",
        context_parallel_size: int = 1,
        cp_mesh=None,
        cp_rotate_method: str = "allgather",
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.sequence_parallel_size = sequence_parallel_size
        self.context_parallel_size = context_parallel_size
        # Stage 4 (FSDP2 CP): the cp submesh + rotate method for the ring-SDPA
        # forward wrap. cp_mesh is None when context_parallel_size == 1 (flag-off
        # → forward takes the literal no-op nullcontext, byte-identical to today).
        self.cp_mesh = cp_mesh
        self.cp_rotate_method = cp_rotate_method
        # Stage 2: resolve attention backend. attn_backend="auto" reproduces the
        # pre-Stage-2 logic byte-for-byte (G1); otherwise it overrides flash_attn.
        # Under CP (context_parallel_size > 1) flash-attn varlen is rejected (G2).
        self.attn_implementation = resolve_attn_implementation(
            attn_backend=attn_backend,
            use_flash_attention_2=use_flash_attention_2,
            context_parallel_size=context_parallel_size,
        )
        self.use_sample_packing = use_sample_packing
        # packing samples using Flash Attention 2
        if use_sample_packing:
            assert (
                self.attn_implementation == "flash_attention_2"
            ), "Flash attention 2 should be used for `use_sample_packing`"

        if isinstance(pretrain_or_model, str):
            # Qwen3-Next GatedDeltaNet kernel routing (Stage 7/8): when the fla
            # overlay is mounted, the broken fla-0.5.0 wheel would crash the
            # qwen3_next modeling import — mask fla off BEFORE from_pretrained so
            # transformers uses its pure-torch (or, opt-in, FlashQLA) GDN path.
            # Gated on SKYRL_GDN_MASK_FLA so non-Qwen3-Next runs are untouched.
            if os.environ.get("SKYRL_GDN_MASK_FLA", "0") in ("1", "true", "True"):
                from skyrl_train.models.qwen3_next_gdn import mask_fla

                mask_fla()
            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                if bf16 and ds_config["torch_autocast"]["enabled"]:
                    # The model’s dtype on initialization follows the config passed to `HfDeepSpeedConfig`,
                    # regardless of the `torch_dtype` specified in `from_pretrained`.
                    # To align with this behavior, we temporarily set `bf16` to True in a copied config.
                    # Note: this does NOT affect the config passed to `deepspeed.initialize()`.
                    ds_config = deepcopy(ds_config)
                    ds_config["bf16"] = {"enabled": True}
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None  # noqa: F841

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            rope_scaling_kwargs = {}
            if rope_scaling:
                rope_scaling_kwargs["rope_scaling"] = rope_scaling
            if rope_theta:
                rope_scaling_kwargs["rope_theta"] = rope_theta

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map=device_map,
                **rope_scaling_kwargs,
            )

            # gpt oss
            if Version(transformers.__version__) >= Version("4.56.2"):
                from transformers import GptOssConfig

                if isinstance(self.model.config, GptOssConfig):
                    # patch attention with Unsloth's flex attn
                    from skyrl_train.patches.gptoss.patch_transformers import (
                        custom_attention,
                        custom_attention_mask,
                        patch_GptOssAttention,
                    )
                    from transformers import AttentionInterface, AttentionMaskInterface

                    AttentionInterface.register("custom_flex", custom_attention)
                    AttentionMaskInterface.register("custom_flex", custom_attention_mask)
                    # set attention implementation to be `custom_flex`
                    self.model.set_attn_implementation("custom_flex")
                    self.attn_implementation = "custom_flex"
                    # NOTE: Even though we set a custom attn implementation, we
                    # also patch the full attention function for GPT OSS
                    patch_GptOssAttention()

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    exclude_modules=exclude_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                # On the grouped-GEMM path (Stage 3b+) the HF ``*SparseMoeBlock``
                # instances are swapped for ``GroupedMoEShim``s that deliberately
                # DROP the HF aux-loss and return ``router_logits=None``. Leaving
                # ``output_router_logits=True`` then makes the HF model forward
                # feed an empty/None ``all_router_logits`` tuple into
                # ``load_balancing_loss_func`` → ``gate_logits[0]`` IndexError
                # (Qwen2/Qwen3-MoE). The grouped path doesn't use the HF aux-loss,
                # so keep router-logit collection OFF there. Flag-off / eager
                # paths are unchanged.
                if moe_grouped_gemm:
                    logger.info(
                        "[MoE] grouped-GEMM swap active — leaving output_router_logits False (aux-loss dropped)"
                    )
                    self.model.config.output_router_logits = False
                else:
                    logger.info("[MoE] set output_router_logits as True")
                    self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # Qwen3-Next: opt-in FlashQLA fused GDN kernel (Stage 8). No-op unless
            # SKYRL_GDN_FLASHQLA=1 and the fla_tilelang overlay is mounted; rebinds
            # each Qwen3NextGatedDeltaNet.chunk_gated_delta_rule to the fused
            # tilelang kernel. Falls back to pure-torch (warning) if unavailable.
            if os.environ.get("SKYRL_GDN_MASK_FLA", "0") in ("1", "true", "True"):
                from skyrl_train.models.qwen3_next_gdn import engage_flashqla

                engage_flashqla(self.model)
        else:
            self.model = pretrain_or_model

        # CP mask contract probe (computed once): does this HF model's forward
        # accept the per-layer-type mask DICT escape hatch? Dense Qwen3 does;
        # Qwen3-MoE does NOT (its create_causal_mask path crashes on a dict).
        # Gates the CP forward below between the dict path and the
        # None+monotonic-position_ids path. See _cp_mask_dict_supported.
        self._cp_mask_dict_supported = _cp_mask_dict_supported(self.model)

        # TODO (sumanthrh): do the same for `logprobs_from_logits` and test.
        # Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/#efficient-solution
        self.chunked_entropy_from_logits_fn = (
            torch.compile(chunked_entropy_from_logits, dynamic=True)
            if use_torch_compile
            else chunked_entropy_from_logits
        )

        # MoE router replay (R3) — Stage 2. Detect MoE blocks, and only when the
        # flag is on AND the model actually has MoE blocks do we instantiate a
        # controller + monkeypatch the block class. Flag-off ⇒ self._router_replay
        # is None, no class is patched, and the forward is byte-identical to
        # stock HF (the patched forward, even if some other run installed it,
        # short-circuits to the original when no controller is active).
        self.moe_router_replay = moe_router_replay
        self.moe_grouped_gemm = moe_grouped_gemm
        self._router_replay = None

        # Stage 3b: grouped-GEMM MoE swap (EP=1, torch backend). Behind the
        # `moe_grouped_gemm` flag (default False → no swap, HF eager block class
        # untouched, stock forward — byte-identical to today). When on, each HF
        # `*SparseMoeBlock` instance is replaced (before FSDP2 wrap) by a thin
        # shim around the lifted grouped `MoE`; the shim reuses the Stage-2
        # RouterReplay singleton as the replay transport via the native router
        # `routed_experts` arg, so the forward replay-install seam is unchanged.
        num_moe_blocks = 0
        if moe_grouped_gemm:
            from skyrl_train.models.layers.moe_swap import swap_moe_blocks_to_grouped

            num_moe_blocks = swap_moe_blocks_to_grouped(self.model)

        if moe_router_replay:
            from skyrl_train.models.router_replay import (
                RouterReplay,
                install_router_replay_patch,
                count_moe_layers,
            )

            if not moe_grouped_gemm:
                # Eager-fallback (Stage 2/3a) path: monkeypatch the HF block class.
                # On the grouped path the HF blocks no longer exist (swapped), and
                # the shim drives replay through the native router instead.
                num_moe_blocks = install_router_replay_patch(self.model)
            if num_moe_blocks > 0:
                # Stage 3a: sample packing (use_sample_packing + FA2) is now
                # supported on the eager path. The packed [1, nnz] target is a
                # plain index_select of the dense [B, seq_len] target by the same
                # nnz_indices the forward's unpad_input used (both batch-major).
                # SP (sequence_parallel_size > 1) remains deferred to Stage 4.
                self._router_replay = RouterReplay()
                expected_layers = count_moe_layers(self.model.config)
                if num_moe_blocks != expected_layers:
                    raise AssertionError(
                        f"router_replay: discovered {num_moe_blocks} MoE blocks but "
                        f"config says {expected_layers} MoE layers."
                    )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "min_p": kwargs.get("min_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """
        Process generated sequences to create attention masks and action masks.

        Args:
            sequences (torch.Tensor): Generated sequence tensor
            input_len (int): Length of the input sequence
            eos_token_id (int): Token ID for the end-of-sequence token
            pad_token_id (int): Token ID for the padding token

        Returns:
            tuple: A tuple containing three elements:
                - sequences: Original sequence
                - attention_mask: Attention mask indicating valid token positions
                - action_mask: Action mask indicating valid action token positions
        """
        # Create initial attention mask by marking positions that are neither EOS nor padding tokens
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find the position of the last valid token in each sequence
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

        # Handle cases where EOS tokens might appear in the middle of the prompt (for Llama3 and Qwen2 models)
        # Find the position of the first valid token in each sequence
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # Create position mask
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # Generate final attention mask, keeping only positions between first and last valid tokens
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # In reinforcement learning, the state transition is represented as:
        # state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # Generate state sequence from input_len-1 to second-to-last token
        state_seq = sequences[:, input_len - 1 : -1]
        # Generate action mask indicating valid action token positions
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_output=False,
        compute_entropy=False,
        entropy_requires_grad=True,
        rollout_routed_experts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        sequences_fwd = sequences
        position_ids_fwd = position_ids
        attention_mask_fwd = attention_mask
        if self.use_sample_packing:
            with torch.no_grad():
                # Removes padding to get a packed tensor. `unpad_input` expects 3 dimensional tensor so we unsqueeze first
                # flash_attn < 2.7 returns a 4-tuple (hidden, indices, cu_seqlens,
                # max_seqlen); >= 2.7 adds a 5th `seqused`. We only consume the
                # first two, so star the tail to stay version-agnostic (the SIF
                # ships flash_attn 2.6.3 -> 4-tuple).
                sequences_fwd, nnz_indices, *_ = unpad_input(
                    sequences.unsqueeze(-1), attention_mask=attention_mask
                )
                # (nnz, 1) -> (1, nnz)
                sequences_fwd = sequences_fwd.transpose(0, 1)
                position_ids_fwd, *_ = unpad_input(position_ids.unsqueeze(-1), attention_mask)
                # (nnz, 1) -> (1, nnz)
                position_ids_fwd = position_ids_fwd.transpose(0, 1)
                attention_mask_fwd = None  # no attention mask with FA 2

        sequences_rolled = torch.roll(sequences_fwd, shifts=-1, dims=1)
        if self.sequence_parallel_size > 1:
            # NOTE: don't pass any attn mask with sample packing
            attention_mask_fwd = None if self.use_sample_packing else attention_mask_fwd

            # slice for sequence parallelism
            # (bsz, seqlen) -> (bsz, seqlen//sp_size)
            sequences_fwd, position_ids_fwd, attention_mask_fwd, pad_size = ulysses_pad_and_slice_inputs(
                sequences_fwd, position_ids_fwd, attention_mask_fwd, self.sequence_parallel_size
            )
            sequences_rolled, _, _, _ = ulysses_pad_and_slice_inputs(
                sequences_rolled, None, None, self.sequence_parallel_size
            )

        # Stage 4 (FSDP2 CP): pad the sequence buffers so seq_len % (2*cp) == 0
        # (G4 — torch's CP load balancer requirement) BEFORE entering the CP
        # context. CP and Ulysses are mutually exclusive (both shard the seq dim);
        # CP runs on the dense [B, S] path only (Stage 0 forbids packing under CP).
        # We pad on the RIGHT with pad tokens / position_ids continuing the cumsum
        # / attention_mask=0, and record `cp_pad_size` so Stage 5 can unpad after
        # the per-token unshard. NO slice here — torch's `context_parallel` does
        # the per-rank sharding (and zigzag offset) inside the context. cp_size==1
        # leaves every buffer untouched (the block is skipped, G1).
        cp_size = self.context_parallel_size
        cp_pad_size = 0
        cp_left_shifts = None
        if cp_size > 1:
            assert self.sequence_parallel_size == 1, "CP and Ulysses SP are mutually exclusive (G2)"
            assert not self.use_sample_packing, "CP requires the dense (unpacked) path (G2)"
            # Stage 6 (production fix): CP ring SDPA runs PURE-CAUSAL with
            # attention_mask=None inside the context (the 2D mask is not
            # CP-shardable — see the long comment below). Pure-causal attention
            # only masks pad tokens that come AFTER the real tokens (trailing /
            # right-pad — causality blocks real tokens from attending forward).
            # LEFT-padding (pads BEFORE the real tokens) is NOT masked: every
            # real token would attend back across the leading pads, diverging
            # grossly from the cp=1 (masked) path (~1.0 logprob error). So CP
            # REQUIRES right-aligned (left-flush) batches. Rather than crash on a
            # left-padded batch (SkyRL's collator left-pads prompts), we DETECT
            # the per-row leading-pad count and ROLL-LEFT every per-row [B,S]
            # buffer entering the CP forward so each row becomes left-flush (real
            # tokens first, pads trailing). The roll is recorded in
            # `cp_left_shifts` and INVERTED on the per-token outputs (Stage 5b)
            # so the returned logprobs/entropy are in the ORIGINAL column order —
            # byte-identical alignment to the cp=1 path. Gated by
            # SKYRL_CP_REQUIRE_RIGHT_ALIGN (default "1"); set "0" only if the
            # caller has independently guaranteed right-alignment and wants to
            # skip the per-step realignment. cp_size==1 never reaches here (G1).
            if attention_mask_fwd is not None and os.environ.get("SKYRL_CP_REQUIRE_RIGHT_ALIGN", "1") not in (
                "0",
                "false",
                "False",
            ):
                am = attention_mask_fwd.to(torch.bool)
                _, S = am.shape
                # `first_real`: index of the FIRST real (mask==1) token per row.
                # argmax over a bool->int row returns the first 1; an all-pad row
                # has no 1 → argmax gives 0 (we keep it 0 via `has_real` so the
                # all-pad row is a no-op roll). Right-aligned (left-flush) rows
                # already have first_real==0.
                has_real = am.any(dim=1)
                first_real = torch.argmax(am.int(), dim=1)
                first_real = torch.where(has_real, first_real, torch.zeros_like(first_real))
                if bool((first_real > 0).any()):
                    # Roll each row LEFT by first_real[i] columns (the leading pads
                    # wrap to the trailing end → right-pad). gather_idx[i, j] =
                    # (j + first_real[i]) % S selects, for output column j, the
                    # source column that lands there after the left-roll.
                    arange_S = torch.arange(S, device=am.device).unsqueeze(0)
                    gather_idx = (arange_S + first_real.unsqueeze(1)) % S
                    sequences_fwd = torch.gather(sequences_fwd, 1, gather_idx)
                    sequences_rolled = torch.gather(sequences_rolled, 1, gather_idx)
                    position_ids_fwd = torch.gather(position_ids_fwd, 1, gather_idx)
                    attention_mask_fwd = torch.gather(attention_mask_fwd, 1, gather_idx)
                    cp_left_shifts = first_real
            _, total_seq_len = sequences_fwd.shape
            multiple = 2 * cp_size
            cp_pad_size = (multiple - total_seq_len % multiple) % multiple
            if cp_pad_size > 0:
                pad_id = 0
                sequences_fwd = torch.nn.functional.pad(sequences_fwd, (0, cp_pad_size), value=pad_id)
                sequences_rolled = torch.nn.functional.pad(sequences_rolled, (0, cp_pad_size), value=pad_id)
                # position_ids: continue the per-row count past the last real token
                # so RoPE on the pad region is well-defined (it is masked out anyway).
                last_pos = position_ids_fwd[:, -1:]
                pad_pos = torch.arange(1, cp_pad_size + 1, device=position_ids_fwd.device).unsqueeze(0)
                position_ids_fwd = torch.cat((position_ids_fwd, last_pos + pad_pos), dim=-1)
                if attention_mask_fwd is not None:
                    pad_attn = torch.zeros(
                        attention_mask_fwd.size(0),
                        cp_pad_size,
                        dtype=attention_mask_fwd.dtype,
                        device=attention_mask_fwd.device,
                    )
                    attention_mask_fwd = torch.cat((attention_mask_fwd, pad_attn), dim=-1)
            # The CP context shards ONLY sequences + position_ids along dim=1 (the
            # sequence dim). The 2D attention_mask is deliberately NOT sharded /
            # passed into the model under CP: HF would expand it to a 4D additive
            # bias `[B, 1, S_q, S_kv]` whose key axis must stay FULL-length, but
            # the sharded 2D mask makes that expand fail inside the CP region
            # (`aten.expand` size mismatch S_kv/cp vs S_q). torch CP ring SDPA
            # instead runs PURE CAUSAL attention (is_causal inferred when
            # attention_mask=None), which it shards correctly. Left-padding masking
            # is recovered via position_ids + the post-hoc entropy/logprob masks.
            # `attention_mask_fwd` is kept (full, unsharded) for the entropy mask.
            #
            # Stage 5: we ALSO CP-shard `sequences_rolled` (the per-token labels)
            # with the SAME zigzag load balancer so the per-token logprobs computed
            # on the local sharded logits `[B, S/cp, V]` line up token-for-token
            # with the local logits BEFORE the unshard. (Stage 4 computed logprobs
            # against the FULL sequences_rolled after an immediate logit unshard;
            # Stage 5 moves the unshard seam to AFTER the per-token compute, which
            # is the memory-efficient gather — `[B,S]` logprobs not `[B,S,V]`
            # logits — and is the seam the loss/loss_mask/KL must align on.)
            _cp_buffers = [sequences_fwd, position_ids_fwd, sequences_rolled]
            _cp_seq_dims = [1, 1, 1]
            _cp_no_restore = {sequences_fwd, sequences_rolled}

        # MoE router replay (R3) — Stage 2/3a. When enabled and targets are
        # provided, install the per-layer forced top-k into the controller for
        # the duration of the model forward. Single-GPU; dense (unpacked) AND
        # packed (use_sample_packing + FA2) paths. SP (sequence_parallel_size >
        # 1) remains Stage 4.
        replay_installed = False
        if self._router_replay is not None and rollout_routed_experts is not None and self.sequence_parallel_size == 1:
            from skyrl_train.models.router_replay import set_active_replay

            # Build the dense target off the ORIGINAL [B, seq_len] sequences (the
            # response slice is only meaningful pre-pack), then index_select to
            # the packed [1, nnz] layout by the same nnz_indices the forward's
            # unpad_input used. Both flatten batch-major, so the index_select
            # lands the packed target on the correct rows.
            per_layer_targets, replay_mask = self._build_router_replay_targets(
                rollout_routed_experts,
                sequences,
                num_actions,
                nnz_indices=nnz_indices if self.use_sample_packing else None,
            )
            self._router_replay.begin_replay()
            self._router_replay.set_microbatch_targets(per_layer_targets, replay_mask)
            set_active_replay(self._router_replay)
            replay_installed = True

        # Stage 4/5 (FSDP2 CP): enter torch-native `context_parallel` around the
        # model forward so SDPA dispatches to ring attention on the cp mesh and
        # the listed sequence buffers are sharded by torch's built-in load
        # balancer. cp_size==1 ⇒ `maybe_cp_context` is `contextlib.nullcontext()`
        # (literal no-op, G1). Inside the context the HF forward returns logits
        # sequence-sharded `[B, S/cp, V]`. Stage 5: we keep the logits sharded
        # through the per-token logprob/entropy compute (using the co-sharded
        # `sequences_rolled` labels) and `context_parallel_unshard` ONLY the
        # per-token `[B, S/cp]` outputs back to natural-order `[B, S]` (the
        # loss-aligned seam — mirrors how the Ulysses path gathers per-token
        # logprobs before the response slice).
        if cp_size > 1:
            cp_ctx = maybe_cp_context(
                cp_size,
                self.cp_mesh,
                self.cp_rotate_method,
                buffers=_cp_buffers,
                seq_dims=_cp_seq_dims,
                no_restore=_cp_no_restore,
            )
        else:
            cp_ctx = maybe_cp_context(1, None, None, buffers=[], seq_dims=[])

        defer_teardown = False
        try:
            with cp_ctx:
                # NOTE (sumanthrh): Once we have position_ids, we don't need attention mask with flash attention.
                if self.use_sample_packing and self.attn_implementation == "flash_attention_2":
                    # NOTE (sumanthrh): Don't use attention mask. position_ids is enough.
                    # Not using attention mask leads to higher perf since flash attention varlen func is enabled
                    output = self.model(sequences_fwd, attention_mask=None, position_ids=position_ids_fwd)
                elif cp_size > 1:
                    # CP: force PURE-CAUSAL ring SDPA. We must NOT let HF build its 4D
                    # additive causal bias mask: under CP `create_causal_mask` runs at
                    # model entry and produces a `[B,1,S_q,S_kv]` bias whose key axis
                    # ends up sharded (S/cp) while the query axis stays full → torch
                    # CP's SDPA wrapper then fails `aten.expand` (kv=S/cp vs q=S). HF's
                    # documented escape hatch (modeling_qwen3.py:403) is to pass
                    # `attention_mask` ALREADY as the per-layer-type mask DICT, which
                    # short-circuits `create_causal_mask`; with None entries every
                    # layer gets attention_mask=None → SDPA sets is_causal=True, which
                    # torch CP ring attention shards correctly. Padding masking is
                    # recovered post-hoc (entropy/logprob masks); exact per-token
                    # padding correctness is the Stage-5 concern.
                    if self._cp_mask_dict_supported:
                        cp_mask = {"full_attention": None, "sliding_attention": None}
                        output = self.model(sequences_fwd, attention_mask=cp_mask, position_ids=position_ids_fwd)
                    else:
                        # Qwen3-MoE has no dict escape hatch (modeling_qwen3_moe.py:497
                        # → create_causal_mask → dict.ndim AttributeError). Pass
                        # attention_mask=None with MONOTONIC per-row position_ids so HF's
                        # find_packed_sequence_indices returns None (the pad-filled-to-1
                        # positions would otherwise spawn a spurious packed mask) →
                        # SDPA is_causal=True → ring attention shards cleanly. Pad RoPE is
                        # masked out post-hoc (entropy/logprob masks), so monotonic is safe.
                        mono_pos = (
                            torch.arange(sequences_fwd.size(1), device=sequences_fwd.device)
                            .unsqueeze(0)
                            .expand(sequences_fwd.size(0), -1)
                        )
                        output = self.model(sequences_fwd, attention_mask=None, position_ids=mono_pos)
                else:
                    output = self.model(sequences_fwd, attention_mask=attention_mask_fwd, position_ids=position_ids_fwd)

            # Stage-7 P3 recompute-safety: when this forward builds an autograd
            # graph (the training forward), the replay teardown MUST NOT fire here.
            # Under activation/gradient checkpointing, backward RECOMPUTES this
            # forward; if the controller is already cleared / uninstalled, the
            # grouped/replay shim takes the natural-routing branch on recompute and
            # saves a different number of tensors than the original (replay) forward
            # -> torch CheckpointError ("N vs M tensors"). So the teardown is DEFERRED
            # to after backward: the lifecycle now spans forward -> backward
            # (option (a) of stage7_scope P3). The recompute fires during backward
            # while the controller is still installed; the controller keys layer
            # position on id(module) (router_replay.py), so the recompute forward
            # re-installs the SAME substituted indices and stays byte-deterministic.
            # The owner (Worker.training_step) calls teardown_router_replay() after
            # strategy.backward(). For no-grad forwards (logprob/entropy scoring,
            # eval) there is no backward to span -> tear down immediately in the
            # finally as before. Flag-off (replay_installed=False) is unchanged.
            if replay_installed and torch.is_grad_enabled() and output["logits"].requires_grad:
                defer_teardown = True
        finally:
            if replay_installed and not defer_teardown:
                self.teardown_router_replay()

        logits_BSV = output["logits"]
        logits_BSV.div_(temperature)

        # NOTE: this is slightly inaccurate with sample packing because last token from nth seq -> first token of n+1th seq loss is added.
        # Under CP `logits_BSV` is sequence-sharded `[B, S/cp, V]` and
        # `sequences_rolled` was co-sharded by the SAME zigzag balancer, so this
        # per-token compute is token-for-token aligned on the local shard.
        log_probs = logprobs_from_logits(
            logits_BSV,
            sequences_rolled,
            inplace_backward=True,
        )

        # Stage 5 (FSDP2 CP) — THE correctness seam: unshard the per-token
        # `[B, S/cp]` logprobs back to natural-order `[B, S]` via the inverse of
        # torch's zigzag load balancer. This is the loss-aligned gather (mirrors
        # the Ulysses `gather_outputs_and_unpad` seam below, different gather op):
        # after this the logprobs are in the SAME token order as the cp=1 path, so
        # the response slice / loss / loss_mask / advantages / ref-KL all line up
        # exactly. cp_size==1 ⇒ skipped (G1). Entropy is unsharded separately
        # below (it must be computed unmasked on the shard, then masked post-gather
        # — the full attention_mask can't be applied to a zigzag shard).
        if cp_size > 1:
            # Stage 6: the stock context_parallel_unshard is @torch.no_grad (its
            # in-place index-restore raises "cannot resize variables that require
            # grad"). For a CP TRAINING step the per-token logprobs must stay
            # differentiable (they feed the policy loss -> backward), so when grad
            # is enabled and the tensor needs grad use the autograd-safe unshard
            # (differentiable all_gather + out-of-place reorder, byte-identical
            # natural order). No-grad scoring keeps the cheaper stock unshard.
            if torch.is_grad_enabled() and log_probs.requires_grad:
                log_probs = cp_unshard_grad_safe(self.cp_mesh, log_probs, 1)
            else:
                log_probs = context_parallel_unshard(self.cp_mesh, [log_probs], [1])[0]

        # gather output if sp > 1
        if self.sequence_parallel_size > 1:
            dim = log_probs.ndim - 1
            log_probs = gather_outputs_and_unpad(
                log_probs, gather_dim=dim, unpad_dim=dim, padding_size=pad_size
            )  # shape can be (1, nnz) - with packing or (B, S) - without packing

        if self.use_sample_packing:
            # add padding back - postprocess logprobs to be compatible with original tensor
            batch_size, seqlen = attention_mask.shape
            # (1, nnz-1) -> (batch_size, seqlen). Pad token ID used by flash attention is 0.
            log_probs = pad_input(
                log_probs.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
            ).squeeze(-1)

        if compute_entropy:
            # For sample packing: entropy is calculated on unpacked data, so no attention mask needed
            # For non-sample packing: pass the attention mask to exclude padding tokens
            entropy_mask = None
            if not self.use_sample_packing:
                # Non-sample packing: pass attention mask to handle padding
                # Use attention_mask_fwd which may be sliced (if sequence_parallel_size > 1) or full
                entropy_mask = attention_mask_fwd

            # Stage 5 (FSDP2 CP): logits are sequence-sharded `[B, S/cp, V]`, but the
            # entropy attention_mask is FULL-length `[B, S]` and in NATURAL order —
            # it cannot index a zigzag shard. So compute entropy UNMASKED on the
            # shard, `context_parallel_unshard` it to natural-order `[B, S]`, THEN
            # apply the full mask. This yields the SAME masked entropy as cp=1.
            if cp_size > 1:
                entropy_BS = self.chunked_entropy_from_logits_fn(
                    logits_BSV, requires_grad=entropy_requires_grad, attention_mask=None
                )
                # Stage 6: grad-safe unshard when entropy carries grad (entropy can
                # appear in the loss via an entropy bonus); else the stock no_grad unshard.
                if torch.is_grad_enabled() and entropy_BS.requires_grad:
                    entropy_BS = cp_unshard_grad_safe(self.cp_mesh, entropy_BS, 1)
                else:
                    entropy_BS = context_parallel_unshard(self.cp_mesh, [entropy_BS], [1])[0]
                if entropy_mask is not None:
                    entropy_BS = entropy_BS * entropy_mask.to(entropy_BS.dtype)
            else:
                entropy_BS = self.chunked_entropy_from_logits_fn(
                    logits_BSV, requires_grad=entropy_requires_grad, attention_mask=entropy_mask
                )

            if self.sequence_parallel_size > 1:
                dim = entropy_BS.ndim - 1
                entropy_BS = gather_outputs_and_unpad(
                    entropy_BS, gather_dim=dim, unpad_dim=dim, padding_size=pad_size
                )  # shape can be (1, nnz) - with packing or (B,S) - without packing
            if self.use_sample_packing:
                entropy_BS = pad_input(
                    entropy_BS.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
                ).squeeze(
                    -1
                )  # (1, nnz) -> (B, S)

            output["entropy"] = entropy_BS

        # Stage 4 (FSDP2 CP): strip the right-pad added for the 2*cp divisibility
        # (G4) so the per-token tensors return to the original [B, S] length and
        # the action slice below lands on the real response tokens. The pad region
        # carried attention_mask==0, so the dropped logprobs/entropy are over pad
        # tokens only (real-token values unaffected; no NaN/inf leaks). cp_size==1
        # ⇒ cp_pad_size==0, this block is a no-op (G1).
        if cp_size > 1 and cp_pad_size > 0:
            log_probs = log_probs[:, : log_probs.size(1) - cp_pad_size]
            if compute_entropy:
                output["entropy"] = output["entropy"][:, : output["entropy"].size(1) - cp_pad_size]

        # Stage 5b (FSDP2 CP): if we LEFT-rolled the inputs to right-align a
        # left-padded batch (cp_left_shifts set above), INVERT the roll now so
        # the per-token logprobs/entropy return to the ORIGINAL column order. The
        # forward left-rolled row i by f=cp_left_shifts[i] (gather j -> (j+f)%S);
        # the inverse roll-RIGHT is gather j -> (j-f)%S. After the G4 strip above
        # the tensors are back to length S (the same S the shifts were computed
        # on), so the inverse gather restores the exact cp=1 token order — the
        # trainer's seqnorm loss_mask + TIS rollout_logprobs alignment and the
        # `num_actions` slice below all see natural order. cp_size==1 ⇒
        # cp_left_shifts is None, this block is a no-op (G1).
        if cp_size > 1 and cp_left_shifts is not None:
            S = log_probs.size(1)
            arange_S = torch.arange(S, device=log_probs.device).unsqueeze(0)
            inv_idx = (arange_S - cp_left_shifts.unsqueeze(1)) % S
            log_probs = torch.gather(log_probs, 1, inv_idx)
            if compute_entropy:
                output["entropy"] = torch.gather(output["entropy"], 1, inv_idx)

        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                num_actions = np.array(num_actions)
        action_log_probs = log_probs[:, -num_actions - 1 : -1]

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def teardown_router_replay(self):
        """Uninstall the active replay controller and reset its per-microbatch
        state. Idempotent and a no-op when replay is disabled / no controller.

        Stage-7 P3: the training forward DEFERS teardown to after backward (so
        gradient-checkpoint recompute, which re-runs the MoE forward during
        backward, still sees the installed controller and the same forced
        targets -> no CheckpointError). The owner (Worker.training_step) MUST
        call this after strategy.backward() returns. No-grad scoring forwards
        tear down inline in forward() and this becomes a harmless no-op.
        """
        if self._router_replay is None:
            return
        from skyrl_train.models.router_replay import set_active_replay

        set_active_replay(None)
        self._router_replay.clear()

    def _build_router_replay_targets(self, rollout_routed_experts, sequences, num_actions, nnz_indices=None):
        """Build per-layer forced-topk targets + a per-token replay mask.

        ``rollout_routed_experts`` is ``[B, response_len, L, K]`` (response axis).
        The dense target is built off the ORIGINAL ``[B, seq_len]`` ``sequences``
        (the response slice is only meaningful pre-pack); HF MoE blocks flatten
        ``[B, seq_len] -> (B*seq_len)`` in batch-major (row) order. We build a
        full-sequence target ``[B*seq_len, K]`` per layer and a ``[B*seq_len]``
        bool mask True only on response positions (the last ``num_actions``
        columns) AND non-sentinel rows. Prompt / pad / sentinel rows fall
        through to natural routing.

        Stage 3a — sample packing: when ``nnz_indices`` is not None the forward
        ran ``unpad_input`` and the model sees a packed ``[1, nnz]`` sequence.
        ``unpad_input``'s indices = ``nonzero(attention_mask.flatten())`` select
        valid tokens in the SAME batch-major flatten order this builder uses
        (``reshape(-1)`` / ``permute(2,0,1,3).reshape(L, B*seq_len, K)``), so the
        packed ``(nnz)`` target/mask is a plain ``index_select(0, nnz_indices)``
        of the dense ones. Left-pad rows have ``attention_mask == 0`` → dropped
        by ``nonzero`` → never in ``nnz_indices`` (automatic). The controller is
        layout-agnostic (only checks ``shape[0]``).
        """
        from skyrl_train.models.router_replay import SENTINEL_EXPERT_ID

        if isinstance(num_actions, (list, np.ndarray)):
            raise NotImplementedError(
                "router_replay requires a scalar num_actions (dense unpacked path); " "got a per-sample list/array."
            )
        device = sequences.device
        batch_size, seq_len = sequences.shape
        re = rollout_routed_experts.to(device=device, dtype=torch.long)
        B, response_len, L, K = re.shape
        assert B == batch_size, f"router_replay batch mismatch: {B} vs {batch_size}"
        assert response_len == num_actions, f"router_replay response_len {response_len} != num_actions {num_actions}"

        # Full-seq target [B, seq_len, L, K], sentinel-filled, response copied in.
        full = torch.full((batch_size, seq_len, L, K), SENTINEL_EXPERT_ID, dtype=torch.long, device=device)
        full[:, seq_len - response_len : seq_len, :, :] = re

        # Replay mask: True on response positions whose row is non-sentinel
        # (a row is sentinel iff all K captured experts equal SENTINEL_EXPERT_ID).
        response_pos = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        response_pos[:, seq_len - response_len : seq_len] = True
        # non-sentinel per [B, seq_len, L]; collapse over L: a position is valid
        # for replay only where every layer carries real data. Use layer 0 as the
        # representative (the capture rail writes the same sentinel pattern across
        # layers for a given token), then AND with response_pos.
        non_sentinel = (full != SENTINEL_EXPERT_ID).any(dim=-1).all(dim=-1)  # [B, seq_len]
        replay_mask_BS = response_pos & non_sentinel  # [B, seq_len]

        # Flatten batch-major to match HF's [B, seq_len] -> (B*seq_len).
        replay_mask = replay_mask_BS.reshape(-1)  # [B*seq_len]
        # Per-layer targets: [B*seq_len, K] each, ordered by layer position.
        full_flat = full.permute(2, 0, 1, 3).reshape(L, batch_size * seq_len, K)  # [L, B*seq_len, K]
        per_layer_targets = [full_flat[i] for i in range(L)]

        # Stage 3a: under sample packing the model forward operates on the packed
        # [1, nnz] sequence. Project the dense [B*seq_len] target/mask down to the
        # packed (nnz) layout via the same nnz_indices unpad_input used — both are
        # batch-major flattens of [B, seq_len], so this is a plain index_select.
        if nnz_indices is not None:
            nnz_indices = nnz_indices.to(device)
            replay_mask = replay_mask.index_select(0, nnz_indices)
            per_layer_targets = [t.index_select(0, nnz_indices) for t in per_layer_targets]

        return per_layer_targets, replay_mask

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def _get_critic_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="value_head",
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
    context_parallel_size: int = 1,
    cp_mesh=None,
    cp_rotate_method: str = "allgather",
):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.sequence_parallel_size = sequence_parallel_size
            self.use_sample_packing = use_sample_packing
            # Stage 4 (FSDP2 CP): value forward must CP-shard identically to the
            # policy so value targets align post-unshard (G3). None at cp=1.
            self.context_parallel_size = context_parallel_size
            self.cp_mesh = cp_mesh
            self.cp_rotate_method = cp_rotate_method
            # CP mask contract probe (computed once): dense Qwen3 accepts the
            # per-layer-type mask DICT, Qwen3-MoE does not. Gates the CP forward
            # below. See _cp_mask_dict_supported (mirrors HFModelWrapper).
            self._cp_mask_dict_supported = _cp_mask_dict_supported(
                getattr(self, self.base_model_prefix)
            )
            if use_sample_packing:
                assert (
                    config._attn_implementation == "flash_attention_2"
                ), "Flash attention must be used with sample packing"

            if self.sequence_parallel_size > 1:
                logger.info("Critic model using sequence parallelism with size: ", self.sequence_parallel_size)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            input_ids_fwd = input_ids
            position_ids_fwd = position_ids
            attention_mask_fwd = attention_mask

            if self.use_sample_packing:
                with torch.no_grad():
                    # remove padding. `unpad_input` expects 3 dimensional tensor
                    # version-agnostic unpack (flash_attn 2.6 -> 4-tuple, 2.7+ -> 5-tuple)
                    input_ids_fwd, nnz_indices, *_ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    input_ids_fwd = input_ids_fwd.transpose(0, 1)
                    position_ids_fwd, *_ = unpad_input(
                        position_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    position_ids_fwd = position_ids_fwd.transpose(0, 1)
                    # don't use attention mask with FA2
                    attention_mask_fwd = None

            if self.sequence_parallel_size > 1:
                assert self.use_sample_packing, "sample packing must be true for sequence parallelism"
                # don't pass any attention mask for flash attention 2. this will save an all gather.
                attention_mask_fwd = None if self.config._attn_implementation == "flash_attention_2" else attention_mask
                # slice for sequence parallelism
                # (bsz, seqlen) -> (bsz, seqlen//sp_size)
                input_ids_fwd, position_ids_fwd, attention_mask_fwd, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_fwd, position_ids_fwd, attention_mask_fwd, self.sequence_parallel_size
                )

            # Stage 4/5 (FSDP2 CP): pad to 2*cp divisibility (G4), then wrap the base
            # forward in torch-native context_parallel (ring SDPA) and unshard the
            # per-token hidden states `[B, S/cp, H]` -> natural-order `[B, S, H]`.
            # The value head is a per-token pointwise Linear, so unsharding the
            # hidden states (then projecting) yields natural-order values that are
            # token-for-token aligned with cp=1 — the SAME loss-aligned seam Stage 5
            # uses for policy logprobs. cp_size==1 ⇒ no-op (G1). Mirrors the
            # policy/ref HFModelWrapper.forward exactly so value targets align.
            cp_size = self.context_parallel_size
            cp_pad_size = 0
            if cp_size > 1:
                assert self.sequence_parallel_size == 1, "CP and Ulysses SP are mutually exclusive (G2)"
                assert not self.use_sample_packing, "CP requires the dense (unpacked) path (G2)"
                _, total_seq_len = input_ids_fwd.shape
                multiple = 2 * cp_size
                cp_pad_size = (multiple - total_seq_len % multiple) % multiple
                if cp_pad_size > 0:
                    input_ids_fwd = torch.nn.functional.pad(input_ids_fwd, (0, cp_pad_size), value=0)
                    last_pos = position_ids_fwd[:, -1:]
                    pad_pos = torch.arange(1, cp_pad_size + 1, device=position_ids_fwd.device).unsqueeze(0)
                    position_ids_fwd = torch.cat((position_ids_fwd, last_pos + pad_pos), dim=-1)
                    if attention_mask_fwd is not None:
                        pad_attn = torch.zeros(
                            attention_mask_fwd.size(0),
                            cp_pad_size,
                            dtype=attention_mask_fwd.dtype,
                            device=attention_mask_fwd.device,
                        )
                        attention_mask_fwd = torch.cat((attention_mask_fwd, pad_attn), dim=-1)
                # Only sequences + position_ids are CP-sharded; the 2D mask is NOT
                # passed under CP (the 4D-bias expand fails on a sharded mask) — CP
                # runs pure causal SDPA. See HFModelWrapper.forward for the rationale.
                _cp_buffers = [input_ids_fwd, position_ids_fwd]
                _cp_seq_dims = [1, 1]
                cp_ctx = maybe_cp_context(
                    cp_size,
                    self.cp_mesh,
                    self.cp_rotate_method,
                    buffers=_cp_buffers,
                    seq_dims=_cp_seq_dims,
                    no_restore={input_ids_fwd},
                )
            else:
                cp_ctx = maybe_cp_context(1, None, None, buffers=[], seq_dims=[])

            with cp_ctx:
                if self.sequence_parallel_size > 1 and self.config._attn_implementation == "flash_attention_2":
                    outputs = getattr(self, self.base_model_prefix)(input_ids_fwd, position_ids=position_ids_fwd)
                elif cp_size > 1:
                    # CP: pass the per-layer-type mask DICT (None entries) so HF skips
                    # create_causal_mask → SDPA is_causal=True → ring attention shards
                    # cleanly (see HFModelWrapper.forward for the full rationale).
                    if self._cp_mask_dict_supported:
                        cp_mask = {"full_attention": None, "sliding_attention": None}
                        outputs = getattr(self, self.base_model_prefix)(
                            input_ids_fwd, attention_mask=cp_mask, position_ids=position_ids_fwd
                        )
                    else:
                        # Qwen3-MoE: no dict escape hatch. attention_mask=None +
                        # MONOTONIC position_ids → find_packed_sequence_indices None →
                        # SDPA is_causal=True (mirrors HFModelWrapper.forward).
                        mono_pos = (
                            torch.arange(input_ids_fwd.size(1), device=input_ids_fwd.device)
                            .unsqueeze(0)
                            .expand(input_ids_fwd.size(0), -1)
                        )
                        outputs = getattr(self, self.base_model_prefix)(
                            input_ids_fwd, attention_mask=None, position_ids=mono_pos
                        )
                else:
                    outputs = getattr(self, self.base_model_prefix)(
                        input_ids_fwd, attention_mask=attention_mask_fwd, position_ids=position_ids_fwd
                    )
                last_hidden_states_BSH = outputs["last_hidden_state"]
                if cp_size > 1:
                    # Stage 5: unshard hidden states [B, S/cp, H] -> natural [B, S, H].
                    # Stage 6: grad-safe unshard when training the value head (the
                    # hidden states feed the value loss -> backward); stock no_grad
                    # unshard for inference. The stock unshard is @torch.no_grad and
                    # raises on grad-requiring tensors.
                    if torch.is_grad_enabled() and last_hidden_states_BSH.requires_grad:
                        last_hidden_states_BSH = cp_unshard_grad_safe(self.cp_mesh, last_hidden_states_BSH, 1)
                    else:
                        last_hidden_states_BSH = context_parallel_unshard(
                            self.cp_mesh, [last_hidden_states_BSH], [1]
                        )[0]

            if self.sequence_parallel_size > 1:
                last_hidden_states_SH = last_hidden_states_BSH.squeeze(0)
                # (seqlen*bsz//sp_size, 1) -> (seqlen*bsz, 1)
                last_hidden_states_SH = gather_outputs_and_unpad(
                    last_hidden_states_SH, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
                last_hidden_states_BSH = last_hidden_states_SH.unsqueeze(0)

            values_BSH = getattr(self, self.value_head_prefix)(last_hidden_states_BSH)

            if self.use_sample_packing:
                # add padding back - postprocess logits to be compatible with original tensors
                batch_size, seqlen = attention_mask.shape
                # (1, nnz, 1) -> (nnz, 1) -> (batch_size, seqlen, 1)
                values_BSH = pad_input(values_BSH.squeeze(0), indices=nnz_indices, batch=batch_size, seqlen=seqlen)

            # Stage 4: strip the CP right-pad so values return to [B, S] before the
            # :-1 trim and action slice land on the real response tokens (no-op cp=1).
            if cp_size > 1 and cp_pad_size > 0:
                values_BSH = values_BSH[:, : values_BSH.size(1) - cp_pad_size]

            values = values_BSH.squeeze(-1)[:, :-1]

            if num_actions is None:
                assert return_output
                return outputs

            action_values = values[:, -num_actions:]

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    exclude_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
    attn_backend: str = "auto",
    context_parallel_size: int = 1,
    cp_mesh=None,
    cp_rotate_method: str = "allgather",
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Type of sequence classification model. Only `critic` is supported.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert model_type == "critic", f"Only model_type critic is supported, got: {model_type}."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # Stage 2: resolve attention backend (auto = pre-Stage-2 behavior; CP rejects flash).
    config._attn_implementation = resolve_attn_implementation(
        attn_backend=attn_backend,
        use_flash_attention_2=use_flash_attention_2,
        context_parallel_size=context_parallel_size,
    )

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_critic_model(
        base_pretrained_class,
        base_class,
        value_head_prefix,
        sequence_parallel_size=sequence_parallel_size,
        use_sample_packing=use_sample_packing,
        context_parallel_size=context_parallel_size,
        cp_mesh=cp_mesh,
        cp_rotate_method=cp_rotate_method,
    )

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        logger.info("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            import deepspeed

            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model
