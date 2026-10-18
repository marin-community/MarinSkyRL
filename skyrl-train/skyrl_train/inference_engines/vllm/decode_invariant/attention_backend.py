"""vLLM's CUSTOM attention backend for a decode-invariant engine: each query row's output equals a one-row decode
step of that row."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import ClassVar, NamedTuple

import torch
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from skyrl_train.inference_engines.vllm.decode_invariant.fa3 import (
    FA3_BLOCK_M,
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_HEAD_DIM,
    FA3_INVARIANT_SPLITS,
    FLASH_ATTN_VERSION,
    Fa3Requests,
    fa3_fixed_split_metadata,
    fa3_invariant_requests,
    fa3_request_calls,
    fa3_window_start_rows,
)

HOPPER_COMPUTE_CAPABILITY_MAJOR = 9


class StepCall(NamedTuple):
    """One FA3 call of a step: its query rows, its metadata, and for one-row requests of a sliding-window layer the
    window start of each request."""

    rows: slice
    metadata: DecodeInvariantMetadata
    window_starts: torch.Tensor | None


@dataclass
class DecodeInvariantMetadata(FlashAttentionMetadata):
    """A step's FA3 metadata with the calls, window starts and planned requests of each layer window, which the
    step's first layer of that window fills."""

    calls: dict[int | None, list[StepCall]] = field(default_factory=dict)
    window_starts: dict[int, torch.Tensor] = field(default_factory=dict)
    plans: dict[int | None, Fa3Requests] = field(default_factory=dict)


class DecodeInvariantMetadataBuilder(FlashAttentionMetadataBuilder):
    # Full CUDA graphs cover one-row decode steps; a step with longer requests runs its planned calls eagerly.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_scheduler_metadata(
        self,
        *,
        aot_schedule: bool,
        batch_size: int,
        cu_query_lens: torch.Tensor,
        max_query_len: int,
        seqlens: torch.Tensor,
        max_seq_len: int,
        causal: bool | torch.Tensor,
        max_num_splits: int,
    ) -> torch.Tensor:
        return fa3_fixed_split_metadata(cu_query_lens, self.num_heads_q // self.num_heads_kv, self.num_heads_kv)

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> DecodeInvariantMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.dcp_context_kv_lens is not None or metadata.causal is not True:
            raise NotImplementedError("a decode-invariant engine runs plain causal FA3 varlen steps only")
        metadata.max_num_splits = FA3_INVARIANT_SPLITS
        return DecodeInvariantMetadata(**{item.name: getattr(metadata, item.name) for item in fields(metadata)})

    def update_block_table(
        self, metadata: DecodeInvariantMetadata, blk_table: torch.Tensor, slot_mapping: torch.Tensor
    ) -> DecodeInvariantMetadata:
        # The KV-cache groups of a step share ``plans``; each builds its own calls from its block table.
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping, calls={}, window_starts={})

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class DecodeInvariantAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vllm_flash_attn_version != FLASH_ATTN_VERSION:
            raise RuntimeError(
                f"a decode-invariant engine runs FA{FLASH_ATTN_VERSION}; vLLM selected FA{self.vllm_flash_attn_version} "
                f"for {torch.cuda.get_device_name()}"
            )
        if self.alibi_slopes is not None:
            raise NotImplementedError("a decode-invariant engine runs attention without ALiBi")
        if FA3_BLOCK_M % self.num_queries_per_kv:
            raise NotImplementedError(f"a decode-invariant engine needs query heads per KV head dividing {FA3_BLOCK_M}")
        left, right = self.sliding_window
        self.window: int | None = left + 1 if left >= 0 and right == 0 else None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DecodeInvariantMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        for rows, metadata, window_starts in self._step_calls(attn_metadata):
            if window_starts is None:
                super().forward(
                    layer, query[rows], key, value, kv_cache, metadata, output[rows], output_scale, output_block_scale
                )
            else:
                self._window_rows(layer, query[rows], kv_cache, metadata, output[rows], window_starts)
        return output

    def _step_calls(self, metadata: DecodeInvariantMetadata) -> list[StepCall]:
        if self.window in metadata.calls:
            return metadata.calls[self.window]
        requests = metadata.query_start_loc.numel() - 1
        if metadata.max_query_len <= 1 and requests <= FA3_DYNAMIC_SPLIT_MAX_BATCH:
            calls = [StepCall(slice(None), metadata, self._window_starts(metadata))]
        else:
            calls = self._planned_calls(metadata)
        metadata.calls[self.window] = calls
        return calls

    def _window_starts(self, metadata: DecodeInvariantMetadata) -> torch.Tensor | None:
        """Each request's window start for this layer, computed on the device so a captured CUDA graph replays it."""
        if self.window is None:
            return None
        if self.window not in metadata.window_starts:
            requests = metadata.query_start_loc.numel() - 1
            metadata.window_starts[self.window] = torch.clamp(metadata.seq_lens[:requests] - self.window, min=0)
        return metadata.window_starts[self.window]

    def _planned_calls(self, metadata: DecodeInvariantMetadata) -> list[StepCall]:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "a decode-invariant engine plans a step with longer requests eagerly, not in a CUDA graph"
            )
        requests = metadata.query_start_loc.numel() - 1
        planned = metadata.plans.get(self.window)
        if planned is None:
            planned = fa3_invariant_requests(
                metadata.query_start_loc.tolist(),
                metadata.seq_lens[:requests].tolist(),
                self.window,
                FA3_BLOCK_M // self.num_queries_per_kv,
            )
            metadata.plans[self.window] = planned
        if planned.key_lengths.numel() == requests and requests <= FA3_DYNAMIC_SPLIT_MAX_BATCH:
            return [StepCall(slice(None), metadata, None)]
        device = metadata.query_start_loc.device
        starts = planned.query_start.tolist()
        calls = []
        for begin, end, one_row in fa3_request_calls(
            starts, one_row_calls=self.window is not None, max_requests=FA3_DYNAMIC_SPLIT_MAX_BATCH
        ):
            key_lengths = planned.key_lengths[begin:end]
            call_query_start = (planned.query_start[begin : end + 1] - starts[begin]).to(device)
            call = replace(
                metadata,
                num_actual_tokens=starts[end] - starts[begin],
                max_query_len=max(b - a for a, b in zip(starts[begin:end], starts[begin + 1 : end + 1], strict=True)),
                query_start_loc=call_query_start,
                max_seq_len=int(key_lengths.max()),
                seq_lens=key_lengths.to(device),
                block_table=metadata.block_table[planned.owner[begin:end].to(device)],
                scheduler_metadata=fa3_fixed_split_metadata(
                    call_query_start, self.num_queries_per_kv, self.num_kv_heads
                ),
                calls={},
                window_starts={},
            )
            window_starts = torch.clamp(key_lengths - self.window, min=0).to(device) if one_row else None
            calls.append(StepCall(slice(starts[begin], starts[end]), call, window_starts))
        return calls

    def _window_rows(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: DecodeInvariantMetadata,
        output: torch.Tensor,
        window_starts: torch.Tensor,
    ) -> None:
        """One-row requests of a sliding-window layer on FA3's causal kernel, each over its window alone."""
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        tokens = metadata.num_actual_tokens
        descale_shape = (metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        fa3_window_start_rows(
            query[:tokens],
            key_cache,
            value_cache,
            output[:tokens],
            cu_seqlens_q=metadata.query_start_loc,
            seqused_k=metadata.seq_lens,
            leftpad_k=window_starts,
            max_seqlen_k=metadata.max_seq_len,
            block_table=metadata.block_table,
            softmax_scale=self.scale,
            scheduler_metadata=metadata.scheduler_metadata,
            num_splits=FA3_INVARIANT_SPLITS,
            softcap=self.logits_soft_cap,
            q_descale=layer._q_scale.expand(descale_shape) if self.supports_quant_query_input else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            s_aux=self.sinks,
        )


class DecodeInvariantAttentionBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return AttentionBackendEnum.CUSTOM.name

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == HOPPER_COMPUTE_CAPABILITY_MAJOR

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == FA3_HEAD_DIM

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type[DecodeInvariantAttentionImpl]:
        return DecodeInvariantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DecodeInvariantMetadataBuilder]:
        return DecodeInvariantMetadataBuilder
