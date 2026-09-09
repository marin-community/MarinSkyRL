import ast
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync import worker_bucket_protocol as protocol
from skyrl_train.weight_sync.manifest import pack_bucket
from tests.cpu.weight_sync.test_bucket_receiver import fixture_parts


def worker_methods():
    path = Path(__file__).parents[3] / "skyrl_train/inference_engines/vllm/vllm_engine.py"
    source = ast.parse(path.read_text())
    worker = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "WorkerWrap")
    methods = [
        node for node in worker.body if isinstance(node, ast.FunctionDef) and "diagnostic_weight_sync" in node.name
    ]
    assert len(methods) == 5
    body = ast.ClassDef(name="NativeWorkerMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["NativeWorkerMethods"]


@pytest.fixture
def native_protocol(monkeypatch):
    parts = fixture_parts()
    manifest, sources, parameters, maps, _ = parts
    log = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            log.append(("wait", self.name, event.number))

    default = Stream("receive")
    load = Stream("load")

    class Event:
        counter = 0

        def __init__(self):
            self.number = Event.counter
            Event.counter += 1

        def record(self, stream):
            log.append(("record", stream.name, self.number))

        def synchronize(self):
            log.append(("join", self.number))

        def query(self):
            return ("join", self.number) in log

    @contextmanager
    def stream_context(stream):
        assert stream is load
        yield

    monkeypatch.setattr(protocol, "BUCKET_BYTES", 32)
    monkeypatch.setattr(protocol, "REPLAY_SCRATCH_BYTES", 7)
    monkeypatch.setattr(protocol, "MAX_REPLAY_EXTRA_BYTES", 16)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: log.append(("device_join",)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1000, 2000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 4000)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4011)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: log.append(("reset_peak",)))
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: load)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: default)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    broadcasts = []

    def broadcast(buffer, src, group):
        assert src == 0 and group == "native-custom-group"
        bucket = len(broadcasts) % manifest.bucket_count
        source = torch.empty(32, dtype=torch.uint8)
        count = pack_bucket(manifest, bucket, sources, source)
        assert buffer.numel() == count
        log.append(("broadcast", bucket))
        buffer.copy_(source[:count])
        broadcasts.append(bucket)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    prefix = next(iter(maps))
    module = SimpleNamespace(
        w13_weight=parameters[prefix + ".w13_weight"],
        w2_weight=parameters[prefix + ".w2_weight"],
        quant_method=SimpleNamespace(unquantized_backend=SimpleNamespace(name="TRITON")),
        _map_global_expert_id_to_local_expert_id=lambda expert: maps[prefix][expert],
    )
    worker = worker_methods()()
    worker.device = torch.device("cpu")
    worker._model_update_group = "native-custom-group"
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(named_modules=lambda: [(prefix, module)], named_parameters=lambda: parameters.items())
    )
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="grug_moe", num_experts=4, num_hidden_layers=1), quantization=None
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False),
    )
    return SimpleNamespace(worker=worker, parts=parts, log=log, broadcasts=broadcasts, module=module)


def prepare(case):
    manifest = case.parts[0]
    return case.worker.prepare_diagnostic_weight_sync_buckets(asdict(manifest), manifest.manifest_id)


def test_actual_worker_methods_install_join_then_every_byte_replay(native_protocol):
    case = native_protocol
    receipt = prepare(case)
    manifest = case.parts[0]
    assert receipt["expert_modules"] == 1
    for bucket in range(manifest.bucket_count):
        result = case.worker.receive_diagnostic_weight_sync_bucket(bucket)
        assert result["load_completion_event_recorded"]
    with pytest.raises(ValueError, match="explicit installed-weight completion join"):
        case.worker.receive_diagnostic_weight_sync_bucket(0, replay=True)
    assert len(case.broadcasts) == manifest.bucket_count
    assert case.worker.finish_diagnostic_weight_sync_install()["install_complete"]
    assert ("join", 2) in case.log and ("join", 3) in case.log
    for bucket in range(manifest.bucket_count):
        case.worker.receive_diagnostic_weight_sync_bucket(bucket, replay=True)
    proof = case.worker.finish_diagnostic_weight_sync_replay()
    assert proof["mismatches"] == 0 and proof["coverage"] == 1.0
    assert proof["expected_bytes"] == proof["compared_bytes"] == receipt["installed_parameter_bytes"]
    assert proof["replay_peak_extra_bytes"] == 11
    assert sum(row[:2] == ("wait", "load") for row in case.log) == 2 * manifest.bucket_count
    assert sum(row[:2] == ("wait", "receive") for row in case.log) == 2 * manifest.bucket_count - 2
    case.worker.close_diagnostic_weight_sync_buckets()
    assert not hasattr(case.worker, "_diagnostic_bucket_state")


def test_slot_reuse_waits_before_collective_and_records_receive_before_load(native_protocol):
    case = native_protocol
    prepare(case)
    for bucket in range(3):
        case.worker.receive_diagnostic_weight_sync_bucket(bucket)
    third = case.log.index(("broadcast", 2))
    assert case.log[third - 1] == ("wait", "receive", 2)
    assert case.log[third + 1 : third + 4] == [("record", "receive", 0), ("wait", "load", 0), ("record", "load", 2)]


def test_bad_sequence_rejected_before_broadcast(native_protocol):
    case = native_protocol
    prepare(case)
    with pytest.raises(ValueError, match="manifest order"):
        case.worker.receive_diagnostic_weight_sync_bucket(1)
    assert not case.broadcasts
    with pytest.raises(ValueError, match="missing manifest buckets"):
        case.worker.finish_diagnostic_weight_sync_install()


def test_insufficient_native_headroom_rejects_before_allocation(native_protocol, monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (79, 2000))
    with pytest.raises(ValueError, match="lacks free space"):
        prepare(native_protocol)
    assert not hasattr(native_protocol.worker, "_diagnostic_bucket_state")


@pytest.mark.parametrize("failure", ["backend", "layer_count", "quantization", "rebalancing"])
def test_native_layout_rejects_unsupported_state(native_protocol, failure):
    case = native_protocol
    if failure == "backend":
        case.module.quant_method.unquantized_backend.name = "FLASHINFER"
    elif failure == "layer_count":
        case.worker.vllm_config.model_config.hf_config.num_hidden_layers = 2
    elif failure == "quantization":
        case.worker.vllm_config.model_config.quantization = "fp8"
    else:
        case.worker.vllm_config.parallel_config.enable_eplb = True
    with pytest.raises(ValueError):
        prepare(case)
    assert not hasattr(case.worker, "_diagnostic_bucket_state")


def test_peak_gate_counts_more_than_the_boolean_scratch(native_protocol, monkeypatch):
    case = native_protocol
    prepare(case)
    for bucket in range(case.parts[0].bucket_count):
        case.worker.receive_diagnostic_weight_sync_bucket(bucket)
    case.worker.finish_diagnostic_weight_sync_install()
    for bucket in range(case.parts[0].bucket_count):
        case.worker.receive_diagnostic_weight_sync_bucket(bucket, replay=True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4017)
    with pytest.raises(ValueError, match="1 MiB total scratch limit"):
        case.worker.finish_diagnostic_weight_sync_replay()


def test_incomplete_load_event_prevents_install_completion_and_replay(native_protocol, monkeypatch):
    case = native_protocol
    prepare(case)
    for bucket in range(case.parts[0].bucket_count):
        case.worker.receive_diagnostic_weight_sync_bucket(bucket)
    monkeypatch.setattr(type(case.worker._diagnostic_bucket_state["load_events"][0]), "query", lambda self: False)
    with pytest.raises(RuntimeError, match="remains incomplete"):
        case.worker.finish_diagnostic_weight_sync_install()
    with pytest.raises(ValueError, match="explicit installed-weight completion join"):
        case.worker.receive_diagnostic_weight_sync_bucket(0, replay=True)
    assert len(case.broadcasts) == case.parts[0].bucket_count
