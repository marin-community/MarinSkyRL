# Grug FlashAttention unpadding contract

## Contract

Grug's padded FlashAttention path consumes four values from `flash_attn.bert_padding.unpad_input`: the unpadded
input, flat indices, cumulative sequence lengths, and maximum sequence length. Optional trailing metadata is not
part of Grug's attention contract and must not be required.

The call therefore unpacks the required prefix with a starred remainder. This accepts helper implementations
with or without trailing metadata without version branching or changing the CUDA kernel path.

## Coverage

A CPU contract test calls the real padded-attention method with an `unpad_input` implementation that returns
exactly those four values. It checks the unpadded query, sequence metadata, non-identity kernel result, repadding,
and GQA-expanded value. The H100 parity suite separately covers the installed FlashAttention implementation and
numerical kernel behavior.
