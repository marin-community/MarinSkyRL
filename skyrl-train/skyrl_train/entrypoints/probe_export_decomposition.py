"""One fresh-process arm of the Bridge export decomposition diagnostic.

Run with torchrun --standalone --nproc-per-node=2. A Snowball-width Grug model with two
layers is built at random on Megatron (no Ray, no HF weight load) and exported through the
real ``MegatronWeightExtractor`` into the real ``StreamingBucketSender``; nothing is sent.
Probe-side wrappers bracket the export sub-stages with CUDA events and host timers. This
measures intra-node op costs and counts, not Snowball's cross-node latency or allocator state.
"""

import argparse
import functools
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from skyrl_train.weight_sync.bucket_identity import bucket_identity
from skyrl_train.weight_sync.bucket_sender import StreamingBucketSender
from skyrl_train.weight_sync.export_decomposition_audit import (
    ARM_PARALLELISM,
    ARMS,
    CONNECTIONS,
    CUDA_SUBSTAGES,
    EXPERT_NAME_FRAGMENT,
    HOST_SUBSTAGES,
    LAYERS,
    PASSES,
    REQUIRED_ENVIRONMENT,
    inventory_key,
    layer_of,
    validate_arm_rows,
)
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest
from skyrl_train.weight_sync.megatron_bucket_protocol import complete_exports
from skyrl_train.weight_sync.pipeline_timing import CudaPipelineTiming
from skyrl_train.weight_sync.readback_diagnostics import ENVIRONMENT_KEYS
from skyrl_train.weight_sync.weight_extractor import weight_sync_dtype
from skyrl_train.weight_sync.worker_bucket_protocol import BUCKET_BYTES

SLOT_COUNT = 2
SEED = 17
CHECKPOINT_SHARD_ENTRIES = 12


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def write_snowball_width_checkpoint(path: Path, layers: int = LAYERS, shape_overrides: dict | None = None):
    """Random Grug checkpoint at GrugMoeConfig's Snowball defaults with ``layers`` decoder layers.

    Mirrors tests/gpu/test_grug_megatron.py::_write_tiny_checkpoint but never materialises
    the HF model: names, shapes and dtypes come from a meta-device instance, the bytes are
    random bf16 (router bias fp32) written shard by shard. No tokenizer is needed.
    """
    from safetensors.torch import save_file
    from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, GrugMoeConfig, GrugMoeForCausalLM, is_grug_router_bias

    config = GrugMoeConfig(
        num_hidden_layers=layers,
        max_position_embeddings=128,
        initializer_range=0.02,
        qk_mult=1.37,
        qk_mult_long_scale=1.1,
        **(shape_overrides or {}),
    )
    with torch.device("meta"):
        model = GrugMoeForCausalLM(config)
    # The meta instance is fp32 throughout; the checkpoint is bf16 except the persistent fp32 router bias.
    entries = [
        (name, tuple(tensor.shape), torch.float32 if is_grug_router_bias(GRUG_MOE_MODEL_TYPE, name) else torch.bfloat16)
        for name, tensor in model.state_dict().items()
    ]
    path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(path)
    generator = torch.Generator().manual_seed(SEED)
    weight_map = {}
    total = 0
    for shard, first in enumerate(range(0, len(entries), CHECKPOINT_SHARD_ENTRIES)):
        tensors = {}
        for name, shape, dtype in entries[first : first + CHECKPOINT_SHARD_ENTRIES]:
            if dtype == torch.float32:
                tensor = torch.linspace(-0.3, 0.3, shape[0])
            else:
                tensor = torch.empty(shape, dtype=torch.float32).normal_(std=0.02, generator=generator).to(dtype)
            tensors[name] = tensor.contiguous()
            total += tensor.numel() * tensor.element_size()
        filename = f"model-{shard:05d}.safetensors"
        save_file(tensors, str(path / filename), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, filename))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )
    return [(name, list(shape), _dtype_name(dtype)) for name, shape, dtype in entries]


def checkpoint_inventory(path: Path):
    """HF state dict names, shapes and on-disk dtypes read from the safetensors headers."""
    from safetensors import safe_open

    index = json.loads((path / "model.safetensors.index.json").read_text())
    names = {"BF16": "bfloat16", "F32": "float32", "F16": "float16"}
    entries = []
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(str(path / filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118  safe_open handles are not Mappings
                view = handle.get_slice(name)
                entries.append((name, list(view.get_shape()), names[view.get_dtype()]))
    return sorted(entries)


class SubstageRecorder:
    """CUDA-event and host-clock brackets for one export pass, keyed by sub-stage."""

    def __init__(self, timing, device):
        self.timing = timing
        self.device = device
        self.host_origin = time.perf_counter()
        self.records = {
            stage: {"host_intervals": [], "layers": [], "bytes": [], "calls": 0}
            for stage in (*CUDA_SUBSTAGES, *HOST_SUBSTAGES)
        }

    def cuda(self, stage, layer, call, byte_count):
        stream = torch.cuda.current_stream(self.device)
        token = self.timing.start(stage, stream)
        start = time.perf_counter()
        result = call()
        end = time.perf_counter()
        self.timing.end(token, stream)
        record = self.records[stage]
        record["host_intervals"].append([start - self.host_origin, end - self.host_origin])
        record["layers"].append(layer)
        record["bytes"].append(int(byte_count(result)))
        record["calls"] += 1
        return result

    def host(self, stage, call):
        start = time.perf_counter()
        result = call()
        end = time.perf_counter()
        record = self.records[stage]
        record["host_intervals"].append([start - self.host_origin, end - self.host_origin])
        record["calls"] += 1
        return result


ACTIVE = {"recorder": None}


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def _hf_name(mapping):
    hf_param = getattr(mapping, "hf_param", None)
    if isinstance(hf_param, dict):
        hf_param = next(iter(hf_param.values()), None)
    return hf_param if isinstance(hf_param, str) else None


def _wrap(owner, attribute, stage, layer_from, bytes_from):
    original = getattr(owner, attribute)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        recorder = ACTIVE["recorder"]
        if recorder is None:
            return original(*args, **kwargs)
        return recorder.cuda(stage, layer_from(*args, **kwargs), lambda: original(*args, **kwargs), bytes_from)

    setattr(owner, attribute, wrapped)


def install_wrappers(extractor_class):
    """Probe-only monkeypatches; production modules are untouched on disk."""
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    def mapping_layer(mapping, *args, **kwargs):
        return layer_of(_hf_name(mapping))

    _wrap(MegatronParamMapping, "broadcast_from_pp_rank", "pp_broadcast", mapping_layer, _tensor_bytes)
    _wrap(MegatronParamMapping, "broadcast_obj_from_pp_rank", "pp_broadcast_obj", mapping_layer, lambda r: 0)
    _wrap(MegatronParamMapping, "gather_from_ep_ranks", "ep_gather", mapping_layer, _tensor_bytes)
    _wrap(
        MegatronModelBridge,
        "_accumulate_grouped_export",
        "expert_stack",
        lambda bridge, task, *a, **k: layer_of(_hf_name(task.mapping)),
        _tensor_bytes,
    )

    original_wire = extractor_class._wire_tensor

    @functools.wraps(original_wire)
    def wire_tensor(self, name, tensor, dtype, device):
        recorder = ACTIVE["recorder"]
        if recorder is None:
            return original_wire(self, name, tensor, dtype, device)
        # A no-op cast returns the input; only a real copy counts bytes.
        return recorder.cuda(
            "wire_cast",
            layer_of(name),
            lambda: original_wire(self, name, tensor, dtype, device),
            lambda result: 0 if result.data_ptr() == tensor.data_ptr() else _tensor_bytes(result),
        )

    extractor_class._wire_tensor = wire_tensor

    original_next = StreamingBucketSender._next_source

    @functools.wraps(original_next)
    def next_source(self, entry):
        recorder = ACTIVE["recorder"]
        if recorder is None:
            return original_next(self, entry)
        return recorder.cuda(
            "next_source",
            layer_of(entry.hf_name),
            lambda: original_next(self, entry),
            lambda _: _tensor_bytes(self.current_source),
        )

    StreamingBucketSender._next_source = next_source


def timed_export_hf_weights(bridge, selected_tasks_for):
    """Route ``export_hf_weights`` through per-pass task building and the arm's task filter."""
    original = bridge.export_hf_weights

    @functools.wraps(original)
    def export_hf_weights(model, *args, conversion_tasks=None, **kwargs):
        if conversion_tasks is None:
            recorder = ACTIVE["recorder"]
            build = functools.partial(selected_tasks_for, model)
            conversion_tasks = build() if recorder is None else recorder.cuda("task_build", None, build, lambda r: 0)
        return original(model, *args, conversion_tasks=conversion_tasks, **kwargs)

    bridge.export_hf_weights = export_hf_weights


def build_model(checkpoint: Path, pp: int, ep: int):
    import skyrl_train.models.grug_megatron_bridge  # noqa: F401  registers GrugMoeBridge with Megatron-Bridge
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state, tensor_parallel

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        use_sharp=False,
    )
    torch.manual_seed(SEED)
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)
    bridge = AutoBridge.from_hf_pretrained(str(checkpoint), trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    # Same provider settings as MegatronWorker.init_configs at TP1/CP1/ETP1.
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = pp
    provider.pipeline_dtype = torch.bfloat16
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.attention_backend = "fused"
    provider.variable_seq_lengths = True
    provider.masked_softmax_fusion = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.gradient_accumulation_fusion = False
    for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
        setattr(provider, key, None)
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)
    return bridge, model


def run_arm(arm, device, checkpoint: Path, source_commit):
    from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE
    from skyrl_train.workers.megatron.megatron_worker import MegatronWeightExtractor

    rank = dist.get_rank()
    pp, ep = ARM_PARALLELISM[arm]
    nonexpert = arm.endswith("-nonexpert")
    bridge, model = build_model(checkpoint, pp, ep)
    model_bytes = sum(p.numel() * p.element_size() for module in model for p in module.parameters())
    install_wrappers(MegatronWeightExtractor)
    task_counts = {}

    def selected_tasks_for(actor_module):
        tasks = bridge.get_conversion_tasks(actor_module)
        expert = [task for task in tasks if task.mapping.is_expert]
        selected = [task for task in tasks if not task.mapping.is_expert] if nonexpert else tasks
        task_counts.update(total=len(tasks), expert=len(expert), selected=len(selected))
        return selected

    timed_export_hf_weights(bridge, selected_tasks_for)
    extractor = MegatronWeightExtractor(bridge, model, GRUG_MOE_MODEL_TYPE)
    worker = SimpleNamespace(weight_extractor=extractor)

    on_disk = checkpoint_inventory(checkpoint)
    expected = [
        (name, shape, _dtype_name(weight_sync_dtype(GRUG_MOE_MODEL_TYPE, name, torch.bfloat16)))
        for name, shape, _ in on_disk
        if not (nonexpert and EXPERT_NAME_FRAGMENT in name)
    ]
    # Inventory pass: the yielded order fixes the manifest; nothing is packed.
    yielded = []
    for name, tensor in complete_exports(worker):
        yielded.append((name, list(tensor.shape), _dtype_name(tensor.dtype)))
        del tensor
    torch.cuda.synchronize(device)
    inventory_exact = sorted(inventory_key(e) for e in expected) == sorted(inventory_key(e) for e in yielded)
    specs = [
        TensorSpec(name, tuple(shape), dtype, split_experts=EXPERT_NAME_FRAGMENT in name and len(shape) == 3)
        for name, shape, dtype in yielded
    ]
    manifest = build_manifest(specs, bucket_bytes=BUCKET_BYTES)
    buffers = tuple(torch.empty(BUCKET_BYTES, dtype=torch.uint8, device=device) for _ in range(SLOT_COUNT))
    dist.barrier()
    torch.cuda.synchronize(device)

    passes = []
    for pass_index in range(PASSES):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        timing = CudaPipelineTiming(device)
        recorder = SubstageRecorder(timing, device)
        ACTIVE["recorder"] = recorder
        sender = StreamingBucketSender(manifest, complete_exports(worker), buffers, stage_timing=timing)
        sender.source_copied.synchronize = functools.partial(
            recorder.host, "source_release_wait", sender.source_copied.synchronize
        )
        host_start = time.perf_counter()
        for bucket in range(manifest.bucket_count):
            sender.pack_next_bucket()
            sender.mark_bucket_sent(bucket)  # No NCCL send: the slot is released as soon as it is packed.
        finished = sender.finish()
        host_end = time.perf_counter()
        complete = torch.cuda.Event()
        complete.record(torch.cuda.current_stream(device))
        complete.synchronize()
        ACTIVE["recorder"] = None
        receipt = timing.finish()
        receipt["intervals"] = {k: [[round(s, 6), round(e, 6)] for s, e in v] for k, v in receipt["intervals"].items()}
        for detail in recorder.records.values():
            detail["host_intervals"] = [[round(s, 6), round(e, 6)] for s, e in detail["host_intervals"]]
        passes.append(
            {
                "pass_index": pass_index,
                "warmup": pass_index == 0,
                "timing": receipt,
                "substages": recorder.records,
                "host_pass_seconds": host_end - host_start,
                "source_count": finished["source_count"],
                "bucket_count": finished["bucket_count"],
                "allocated_before": allocated_before,
                "peak_extra_bytes": int(torch.cuda.max_memory_allocated(device) - allocated_before),
                "operation_complete": complete.query(),
            }
        )
        dist.barrier()
    return {
        "arm": arm,
        "layers": LAYERS,
        "pp": pp,
        "ep": ep,
        "rank": rank,
        "source_commit": source_commit,
        "identity": bucket_identity(device),
        "environment": {
            key: os.environ.get(key)
            for key in dict.fromkeys(("CUDA_DEVICE_MAX_CONNECTIONS", *REQUIRED_ENVIRONMENT, *ENVIRONMENT_KEYS))
        },
        "manifest_id": manifest.manifest_id,
        "bucket_bytes": BUCKET_BYTES,
        "bucket_count": manifest.bucket_count,
        "wire_bytes": sum(entry.nbytes for entry in manifest.entries),
        "model_bytes": model_bytes,
        "conversion_tasks": task_counts,
        "expected_inventory": expected,
        "yielded_inventory": yielded,
        "checkpoint_inventory_dtypes": sorted({dtype for _, _, dtype in on_disk}),
        "tensor_inventory_exact": inventory_exact,
        "export_stream": "caller current stream (no separate export stream)",
        "passes": passes,
        "scope": "intra-node op costs and counts at two layers; no cross-node PP latency, no Snowball allocator state",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == str(CONNECTIONS)
    assert all(os.environ.get(key) == value for key, value in REQUIRED_ENVIRONMENT.items())
    assert not torch.cuda.is_initialized(), "Environment must be established before CUDA initialization"
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
    assert dist.get_world_size() == 2 and torch.cuda.device_count() == 2
    try:
        warm = torch.ones(1024, device=device)
        dist.broadcast(warm, 0)
        torch.cuda.synchronize(device)
        local = run_arm(args.arm, device, args.checkpoint, args.source_commit)
        gathered = [None, None]
        dist.all_gather_object(gathered, local)
        if rank == 0:
            cell = {"source_commit": args.source_commit, "rows": gathered}
            print("K11_EXPORT_DECOMPOSITION_RECEIPT " + json.dumps(cell), flush=True)
            validate_arm_rows(gathered, args.arm)
            args.output.write_text(json.dumps(cell) + "\n")
            print(
                f"K11_EXPORT_DECOMPOSITION_ARM_PASS arm={args.arm} layers={LAYERS} "
                f"tensor_inventory_exact={str(all(r['tensor_inventory_exact'] for r in gathered)).lower()} "
                "events_complete=true stage_times_are_observations=true",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
