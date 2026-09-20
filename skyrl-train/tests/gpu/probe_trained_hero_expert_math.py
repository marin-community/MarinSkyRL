"""Probe trained Hero layer-0 selected experts with a single GPU reference.

This is an opt-in diagnostic, not a test discovered by the default GPU suite.
It reads only the eight selected expert slices from the immutable checkpoint.
"""

import argparse
from io import BytesIO
import json
import struct

import numpy as np
import torch
from torch.nn import functional as F

from tests.gpu.test_hero_router_replay_live import _s3_client, _s3_target


def _read_object(client, uri: str, byte_range: tuple[int, int] | None = None) -> bytes:
    bucket, key = _s3_target(uri)
    request = {"Bucket": bucket, "Key": key}
    if byte_range is not None:
        request["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    response = client.get_object(**request)
    with response["Body"] as body:
        return body.read()


def _selected_stacked_weights(client, checkpoint: str, index: dict, name: str, selected: list[int]) -> torch.Tensor:
    uri = checkpoint.rstrip("/") + "/" + index["weight_map"][name]
    header_size = struct.unpack("<Q", _read_object(client, uri, (0, 7)))[0]
    metadata = json.loads(_read_object(client, uri, (8, 7 + header_size)))[name]
    shape = metadata["shape"]
    assert metadata["dtype"] == "BF16" and len(shape) == 3
    assert max(selected) < shape[0]
    data_start, _ = metadata["data_offsets"]
    bytes_per_expert = int(np.prod(shape[1:])) * 2
    tensors = []
    for expert in selected:
        start = 8 + header_size + data_start + expert * bytes_per_expert
        raw = _read_object(client, uri, (start, start + bytes_per_expert - 1))
        assert len(raw) == bytes_per_expert
        bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
        values = bits.view("<f4").reshape(shape[1:])
        tensors.append(torch.from_numpy(values.copy()).to(device="cuda", dtype=torch.bfloat16))
    return torch.stack(tensors)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _candidate_metrics(candidate: torch.Tensor, saved: dict[str, np.ndarray]) -> dict:
    values = candidate.float().cpu().numpy()
    return {
        backend: {
            "rms": _rms(values - target),
            "max_abs": float(np.max(np.abs(values - target))),
            "exact_coordinates": int(np.count_nonzero(values == target)),
        }
        for backend, target in saved.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() >= 1
    client = _s3_client()
    arrays = np.load(BytesIO(_read_object(client, args.trace)))
    assert arrays["positions"][0] == 0
    selected = arrays["layer_0_selected_experts"][0].tolist()
    input_vllm = arrays["vllm_layer_0_routed_input"][0]
    input_megatron = arrays["megatron_layer_0_routed_input"][0]
    assert np.array_equal(input_vllm, input_megatron)
    assert len(selected) == 8
    saved = {backend: arrays[f"{backend}_layer_0_routed_latent"][0] for backend in ("vllm", "megatron")}
    weights = {
        backend: torch.from_numpy(arrays[f"{backend}_layer_0_router_combine"][0].copy()).to("cuda")
        for backend in ("vllm", "megatron")
    }
    index = json.loads(_read_object(client, args.checkpoint.rstrip("/") + "/model.safetensors.index.json"))
    prefix = "model.layers.0.mlp.experts."
    gate = _selected_stacked_weights(client, args.checkpoint, index, prefix + "gate_proj.weight", selected)
    up = _selected_stacked_weights(client, args.checkpoint, index, prefix + "up_proj.weight", selected)
    down = _selected_stacked_weights(client, args.checkpoint, index, prefix + "down_proj.weight", selected)
    x = torch.from_numpy(input_vllm.copy()).to(device="cuda", dtype=torch.bfloat16)
    results = {}
    expert_arrays = {}
    for rows in (1, 8):
        expanded = x.expand(rows, -1).contiguous()
        for hidden_mode in ("bf16", "fp32_then_bf16"):
            hidden_values = []
            for expert in range(len(selected)):
                gate_output = F.linear(expanded, gate[expert])
                up_output = F.linear(expanded, up[expert])
                if hidden_mode == "bf16":
                    hidden = F.silu(gate_output) * up_output
                else:
                    hidden = (F.silu(gate_output.float()) * up_output.float()).to(torch.bfloat16)
                hidden_values.append(hidden)
            for down_mode in ("bf16", "fp32"):
                outputs = torch.stack(
                    [
                        F.linear(hidden_values[expert], down[expert])[0]
                        if down_mode == "bf16"
                        else F.linear(hidden_values[expert].float(), down[expert].float())[0]
                        for expert in range(len(selected))
                    ]
                )
                expert_arrays[f"rows_{rows}_hidden_{hidden_mode}_down_{down_mode}"] = (
                    outputs.detach().float().cpu().numpy()
                )
                for backend, combine in weights.items():
                    weighted_fp32 = outputs.float() * combine[:, None]
                    weighted_bf16 = (outputs * combine.to(torch.bfloat16)[:, None]).to(torch.bfloat16)
                    serial = torch.zeros_like(outputs[0], dtype=torch.bfloat16)
                    for contribution in weighted_bf16:
                        serial = serial + contribution
                    candidates = {
                        "fp32_product_fp32_sum": weighted_fp32.sum(dim=0).to(torch.bfloat16),
                        "bf16_product_fp32_sum": weighted_bf16.float().sum(dim=0).to(torch.bfloat16),
                        "bf16_product_serial_bf16_sum": serial,
                    }
                    for sum_mode, candidate in candidates.items():
                        key = f"rows={rows}/hidden={hidden_mode}/down={down_mode}/weights={backend}/sum={sum_mode}"
                        results[key] = _candidate_metrics(candidate, saved)

            # MCore 0.18 TEGroupedMLP weights the SwiGLU activation before FC2,
            # whereas vLLM's TritonExperts weights the FC2 accumulator. Keep
            # the BF16 activation cast explicit so this tests that boundary.
            for backend, combine in weights.items():
                weight_modes = ("fp32_then_bf16", "bf16")
                if hidden_mode == "fp32_then_bf16":
                    weight_modes += ("fp32_fused_swiglu",)
                for weight_mode in weight_modes:
                    weighted_hidden = []
                    for expert, hidden in enumerate(hidden_values):
                        if weight_mode == "fp32_then_bf16":
                            value = (hidden.float() * combine[expert]).to(torch.bfloat16)
                        elif weight_mode == "fp32_fused_swiglu":
                            gate_output = F.linear(expanded, gate[expert])
                            up_output = F.linear(expanded, up[expert])
                            value = (F.silu(gate_output.float()) * up_output.float() * combine[expert]).to(
                                torch.bfloat16
                            )
                        else:
                            value = hidden * combine[expert].to(torch.bfloat16)
                        weighted_hidden.append(value)
                    weighted_outputs = torch.stack(
                        [F.linear(weighted_hidden[expert], down[expert])[0] for expert in range(len(selected))]
                    )
                    expert_arrays[f"rows_{rows}_hidden_{hidden_mode}_weights_{backend}_pre_down_{weight_mode}"] = (
                        weighted_outputs.detach().float().cpu().numpy()
                    )
                    serial = torch.zeros_like(weighted_outputs[0], dtype=torch.bfloat16)
                    for contribution in weighted_outputs:
                        serial = serial + contribution
                    candidates = {
                        "fp32_sum": weighted_outputs.float().sum(dim=0).to(torch.bfloat16),
                        "serial_bf16_sum": serial,
                    }
                    for sum_mode, candidate in candidates.items():
                        key = (
                            f"rows={rows}/hidden={hidden_mode}/weights={backend}/pre_down={weight_mode}/sum={sum_mode}"
                        )
                        results[key] = _candidate_metrics(candidate, saved)

    arrays_uri = args.result.removesuffix(".json") + "-arrays.npz"
    payload = BytesIO()
    np.savez_compressed(payload, **expert_arrays)
    bucket, key = _s3_target(arrays_uri)
    client.put_object(Bucket=bucket, Key=key, Body=payload.getvalue())
    report = {
        "checkpoint": args.checkpoint,
        "trace": args.trace,
        "selected": selected,
        "device": torch.cuda.get_device_name(0),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "saved_cross_backend_rms": _rms(saved["vllm"] - saved["megatron"]),
        "raw_arrays_uri": arrays_uri,
        "candidates": results,
    }
    bucket, key = _s3_target(args.result)
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(report, sort_keys=True).encode())
    best = {
        backend: min(
            ((name, metrics[backend]) for name, metrics in results.items()),
            key=lambda item: item[1]["rms"],
        )
        for backend in saved
    }
    print("HERO_EXPERT_MATH_STATUS=" + json.dumps({"best": best, "result": args.result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
