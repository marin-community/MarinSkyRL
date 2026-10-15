"""The exact Grug pipeline-stage model: it runs the exact layers with the forward's ``ExactForward`` and carries each
token's input-norm statistic between layers and stages."""

from __future__ import annotations

from collections.abc import Callable

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor

from skyrl_train.config.numerics import FULL_RECOMPUTE, recomputes_one_layer_per_unit
from skyrl_train.models.grug_exact.context import ExactForward, ResidualSum
from skyrl_train.models.grug_exact.layers import ExactEmbeddingGatedNorm, ExactTransformerLayer
from skyrl_train.models.grug_exact.rounding import (
    STAGE_STATISTIC_COLUMNS,
    append_stage_statistic,
    split_stage_statistic,
)
from skyrl_train.models.grug_megatron import GrugGPTModel


def _layer_unit(layer: ExactTransformerLayer, rotary_pos_emb, context: ExactForward) -> Callable:
    """The checkpointed function of one layer: its tensor inputs and outputs, with the residual sum's parts flat."""

    def run(hidden_states: torch.Tensor, statistic: torch.Tensor):
        output, statistic, parts = layer(hidden_states, statistic, rotary_pos_emb, context)
        return output, statistic, parts.residual, parts.mlp

    return run


class GrugExactGPTModel(GrugGPTModel):
    """GrugGPTModel whose forward runs the exact layers; a stage that is not the last appends each token's input-norm
    statistic (one fp32, as two bf16 words) to the hidden states it hands on, and the next stage splits it off."""

    embed_norm_class = ExactEmbeddingGatedNorm

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.recompute_granularity is not None and not recomputes_one_layer_per_unit(
            config.recompute_granularity, config.recompute_method, config.recompute_num_layers
        ):
            raise ValueError(
                "the exact model recomputes activations in one-layer units: recompute_granularity=full, "
                "recompute_method=uniform, recompute_num_layers=1"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs: dict | None = None,
        runtime_gather_output: bool | None = None,
        *,
        exact: ExactForward,
        inference_params=None,
        loss_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        output_processor=None,
        output_processor_context=None,
    ) -> torch.Tensor:
        unsupported = (labels, inference_context, packed_seq_params, inference_params, padding_mask, output_processor)
        if any(item is not None for item in unsupported) or extra_block_kwargs:
            raise NotImplementedError("the exact model runs plain training forwards on unpacked sequences")
        # Grug's own _preprocess applies the stock embedding gated norm; the exact one hands on a statistic.
        decoder_input, rotary_pos_emb, *_ = GPTModel._preprocess(
            self, input_ids=input_ids, position_ids=position_ids, decoder_input=decoder_input
        )
        if self.pre_process:
            hidden_states, statistic = self.embed_norm(decoder_input)
        else:
            received = self.decoder.input_tensor
            if received.shape[-1] != self.config.hidden_size + STAGE_STATISTIC_COLUMNS:
                raise ValueError("an exact pipeline stage receives the hidden states with their input-norm statistic")
            hidden_states, statistic = split_stage_statistic(received, self.config.hidden_size)
            statistic = statistic.reshape(-1)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        hidden_states, statistic, parts = self._decode(hidden_states, statistic, rotary_pos_emb, exact)
        if not self.post_process:
            return append_stage_statistic(hidden_states, statistic.view(*hidden_states.shape[:-1], 1))
        hidden_states = self.decoder.final_layernorm(hidden_states, parts)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            mtp_in_postprocess=self.mtp_process,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            runtime_gather_output=runtime_gather_output,
        )

    def _decode(
        self, hidden_states: torch.Tensor, statistic: torch.Tensor, rotary_pos_emb, context: ExactForward
    ) -> tuple[torch.Tensor, torch.Tensor, ResidualSum]:
        """Run this stage's layers, each as its own checkpoint unit under full activation recompute."""
        config = self.config
        checkpointing = config.recompute_granularity == FULL_RECOMPUTE and self.training
        for layer in self.decoder.layers:
            if checkpointing:
                hidden_states, statistic, *parts = tensor_parallel.checkpoint(
                    _layer_unit(layer, rotary_pos_emb, context),
                    config.distribute_saved_activations,
                    hidden_states,
                    statistic,
                )
                parts = ResidualSum(*parts)
            else:
                hidden_states, statistic, parts = layer(hidden_states, statistic, rotary_pos_emb, context)
        return hidden_states, statistic, parts
