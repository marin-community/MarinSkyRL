# FP8 Inference for RL Training

FP8 (8-bit floating point) inference reduces GPU memory usage and increases throughput for vLLM inference engines during RL training. This allows fitting more KV cache entries and achieving ~1.5x speedup over BF16.

## Overview

The FP8 inference pipeline runs the vLLM inference engines in FP8 precision while keeping the training policy in BF16. During weight sync (after each training step), BF16 weights are broadcast from the FSDP trainer to the inference engines, where they are requantized to FP8 on the receiver side.

**Key challenge**: vLLM fuses linear layers during model loading (e.g., `q_proj + k_proj + v_proj -> qkv_proj`, `gate_proj + up_proj -> gate_up_proj`). Standard per-parameter weight sync breaks because the unfused BF16 weights don't match the fused FP8 parameter shapes. The solution is a batched weight sync protocol that accumulates all BF16 weights, then fuses and requantizes them.

## Architecture

```
Training (BF16)                    Inference (FP8)
+-----------------+                +-------------------+
| FSDP Policy     |  NCCL broadcast  | vLLM Engine       |
| (BF16 weights)  | ──────────────>  | begin_weight_update()
|                 |   per-param      |   accumulate BF16 |
|                 |   q_proj.weight  |   on CPU          |
|                 |   k_proj.weight  |                   |
|                 |   v_proj.weight  |                   |
|                 |                  | end_weight_update()
|                 |                  |   cat q+k+v       |
|                 |                  |   scaled_fp8_quant|
|                 |                  |   copy to qkv_proj|
+-----------------+                +-------------------+
```

## How to Enable

Set the environment variable before launching training:

```bash
export SKYRL_FUSE_WEIGHTS=1
```

And add `--quantization fp8` to the vLLM engine config:

```bash
# In your launch script or SkyRL config:
generator.quantization=fp8
generator.model_dtype=bfloat16  # weights are broadcast in BF16, quantized on receiver
```

## Changes Required

FP8 inference requires patches at three levels:

### 1. vLLM Patches (applied to the installed vllm package)

Two files need patching in the vLLM installation (`site-packages/vllm/`):

#### `vllm/__init__.py` — No-transpose patch

vLLM's FP8 `process_weights_after_loading` transposes weights from `[out, in]` to `[in, out]` for optimized FP8 GEMM. This breaks weight sync because the FSDP trainer broadcasts `[out, in]` weights. The patch:

1. Runs the original FP8 processing (quantize + transpose)
2. Un-transposes the weight back to `[out, in]` for weight sync compatibility
3. Preserves `weight_loader`, `output_dim`, `subclass_type` and other attributes that FP8 processing destroys (needed for `load_weights` to work)

```python
# In vllm/__init__.py, activated when SKYRL_FUSE_WEIGHTS=1
if os.environ.get("SKYRL_FUSE_WEIGHTS") == "1":
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    original_process = Fp8LinearMethod.process_weights_after_loading

    def _patched_process(self, layer, *args, **kwargs):
        # Save param attributes before FP8 processing
        saved_attrs = {}
        for pname, param in layer.named_parameters():
            attrs = {'subclass_type': type(param)}
            for attr in ('weight_loader', 'output_dim', 'input_dim', ...):
                if hasattr(param, attr):
                    attrs[attr] = getattr(param, attr)
            saved_attrs[pname] = attrs

        result = original_process(self, layer, *args, **kwargs)

        # Un-transpose weight back to [out, in]
        if hasattr(layer, 'weight') and layer.weight.data.dim() == 2:
            layer.weight = torch.nn.Parameter(
                layer.weight.data.t().contiguous(), requires_grad=False)

        # Restore attributes on new parameters
        for pname, param in layer.named_parameters():
            if pname in saved_attrs:
                for attr, val in saved_attrs[pname].items():
                    setattr(param, attr, val)
        return result

    Fp8LinearMethod.process_weights_after_loading = _patched_process
```

#### `vllm/model_executor/layers/quantization/fp8.py` — On-the-fly transpose in `apply()`

Since weights are stored as `[out, in]` (un-transposed), the `apply()` method transposes on-the-fly before passing to the FP8 GEMM kernel. This is a non-contiguous view with zero memory cost.

```python
# In Fp8LinearMethod.apply():
_weight = layer.weight
if os.environ.get("SKYRL_FUSE_WEIGHTS") == "1" and _weight.dim() == 2:
    _weight = _weight.t()  # non-contiguous view for cutlass
return self.fp8_linear.apply(input=x, weight=_weight, ...)
```

### 2. SkyRL Inference Engine (`skyrl_train/inference_engines/vllm/vllm_engine.py`)

The vLLM engine class gains three new methods for batched FP8 weight sync:

#### `begin_weight_update()`
Signals the start of a weight sync batch. Initializes an accumulator list for BF16 weights.

#### `end_weight_update()`
Flushes accumulated BF16 weights with FP8 requantization:

1. **For stacked/fused modules** (qkv_proj, gate_up_proj):
   - Concatenates the individual shard tensors (q + k + v)
   - Moves to GPU, converts to BF16
   - Quantizes with `scaled_fp8_quant` -> FP8 tensor + per-tensor scale
   - Copies FP8 data and scale into the model's fused parameter

2. **For non-stacked modules** (o_proj, down_proj):
   - Moves to GPU, quantizes with `scaled_fp8_quant`
   - Copies directly

3. **For non-FP8 params** (layernorm, embedding):
   - Copies BF16 weights directly (no quantization)

```python
def end_weight_update(self):
    if self._is_fp8_model():
        weight_index = {name: tensor for name, tensor in self._accumulated_weights}
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for mname, module in model.named_modules():
            if not isinstance(module.quant_method, Fp8LinearMethod):
                continue
            # Concatenate stacked shards, quantize to FP8, copy
            if is_stacked:
                full_bf16 = torch.cat(shard_list, dim=0).to(device)
                fp8_full, scale = scaled_fp8_quant(full_bf16)
                param.data.copy_(fp8_full)
                module.weight_scale.data.copy_(scale.squeeze())
```

#### `_apply_fp8_weight_loader_patches()` (static method)
Called during engine initialization. Patches `Fp8LinearMethod.process_weights_after_loading` to preserve `weight_loader` and other custom attributes after FP8 processing. This is the verl-inspired approach that ensures `load_weights` can still dispatch to the correct weight loader.

### 3. FSDP Worker (`skyrl_train/workers/fsdp/fsdp_worker.py`)

The FSDP worker wraps the NCCL weight broadcast loop with begin/end hooks:

```python
_fuse_weights = os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"

if _fuse_weights and rank == 0:
    await inference_engine_client.begin_weight_update()

for chunk in self.weight_extractor.extract_weights(generator_dtype):
    # ... NCCL broadcast per parameter ...

if _fuse_weights and rank == 0:
    await inference_engine_client.end_weight_update()
```

## Patched vLLM Files

The following files in the vLLM installation need to be patched. These are in the `py3.12` conda environment on Jupiter at:

```
/e/data1/datasets/playground/ot/envs/py3.12/lib/python3.12/site-packages/vllm/
```

| File | Change | Lines |
|------|--------|-------|
| `__init__.py` | No-transpose patch + attribute preservation | ~109-151 |
| `model_executor/layers/quantization/fp8.py` | On-the-fly `.t()` in `apply()` | ~611-615 |

These patches are gated by `SKYRL_FUSE_WEIGHTS=1` and have no effect when the env var is unset.

## Known Issues

### Zero gradients with FP8 weight sync (UNRESOLVED)

After FP8 requantization during weight sync, `policy_loss` drops to ~0. The BF16 baseline shows real loss values. The requantized FP8 weights may produce slightly different logprobs than the original BF16 weights, causing the advantage computation to collapse. This needs further investigation.

With RLOO advantage estimation, near-zero `policy_loss` can be expected (advantages sum to 0 per prompt), but combined with 0 rewards it indicates the requantized model isn't generating useful output.

### Memory considerations

During `end_weight_update()`, each FP8 module's BF16 shards are concatenated on GPU, quantized, then deleted. This is done per-module to avoid OOM (the full model in BF16 would be ~64GB for a 32B model). Peak memory during quantization is approximately one module's worth of BF16 weights (~200MB for the largest layers).

## Compatibility

- **vLLM version**: Tested with vLLM 0.11.2 (py3.12 env on Jupiter). The patches target `Fp8LinearMethod` which exists in vLLM >= 0.5.
- **Quantization mode**: Per-tensor FP8 only (`--quantization fp8`). Block FP8 is not supported.
- **Models**: Tested with Qwen3-32B and Qwen3-8B.
- **Training framework**: BenSkyRL with FSDP2 strategy.
