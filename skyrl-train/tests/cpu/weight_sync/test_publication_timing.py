import pytest

from skyrl_train.weight_sync.publication_timing import PublicationStageTimer, publication_stage_walls


def test_stage_timer_reports_wall_and_gpu_ms_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    clock = iter([1.0, 1.25, 2.0, 2.5, 3.0, 4.0])
    timer = PublicationStageTimer(enabled=True, now=lambda: next(clock))
    with timer.span("export"):
        pass
    with timer.span("export"):
        pass
    with timer.span("barrier"):
        pass
    assert timer.finish() == {
        "export": {"wall_seconds": 0.75, "gpu_ms": None, "calls": 2},
        "barrier": {"wall_seconds": 1.0, "gpu_ms": None, "calls": 1},
    }
    assert timer.finish() == {}


def test_stage_timer_rejects_unbalanced_start_stop():
    timer = PublicationStageTimer(enabled=True)
    with pytest.raises(RuntimeError, match="not started"):
        timer.stop("export")
    timer.start("export")
    with pytest.raises(RuntimeError, match="already started"):
        timer.start("export")
    with pytest.raises(RuntimeError, match="unfinished"):
        timer.finish()
    timer.stop("export")
    assert timer.finish()["export"]["calls"] == 1


def test_disabled_timing_does_not_read_clocks_or_emit_results():
    def forbidden_clock():
        raise AssertionError("disabled clocks must not perturb publication")

    timer = PublicationStageTimer(enabled=False, now=forbidden_clock)
    with timer.span("export"):
        pass
    assert timer.finish() == {}


def test_cuda_events_are_resolved_only_after_the_publication(monkeypatch):
    completed = False

    class Event:
        def __init__(self, *, enable_timing):
            self.recorded = False

        def record(self):
            self.recorded = True

        def elapsed_time(self, other):
            assert completed, "CUDA events cannot be read before the final synchronization"
            assert self.recorded and other.recorded
            return 2.5

    def synchronize():
        nonlocal completed
        completed = True

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", Event)
    monkeypatch.setattr("torch.cuda.synchronize", synchronize)
    timer = PublicationStageTimer(enabled=True)
    with timer.span("nccl_send"):
        assert not completed
    assert not completed
    assert timer.finish()["nccl_send"]["gpu_ms"] == 2.5


def test_rank_wall_fold_preserves_slowest_rank_without_summing_concurrent_workers():
    receipt = {
        "trainer": [
            {"rank": 0, "stages": {"export": {"wall_seconds": 2.0}}},
            {"rank": 1, "stages": {"export": {"wall_seconds": 3.0}}},
        ],
        "receiver": [
            [{"rank": 0, "stages": {"load": {"wall_seconds": 1.5}}}],
            [{"rank": 0, "stages": {"load": {"wall_seconds": 1.0}}}],
        ],
    }
    assert publication_stage_walls(receipt) == {"weight_broadcast/export": 3.0, "weight_broadcast/load": 1.5}


def _native_receiver_receipt(monkeypatch):
    """Execute WorkerWrap's actual CPU-safe readback without importing vLLM."""
    import ast
    import os
    from pathlib import Path
    import socket
    from types import SimpleNamespace

    source = Path(__file__).parents[3] / "skyrl_train/inference_engines/vllm/vllm_engine.py"
    tree = ast.parse(source.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorkerWrap")
    method = next(
        node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "read_publication_timing"
    )
    namespace = {
        "torch": SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: 3)),
        "socket": socket,
        "os": os,
        "PublicationStageTimer": PublicationStageTimer,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    stages = {
        "recv": {"wall_seconds": 0.01, "gpu_ms": 12.5, "calls": 8},
        "load": {"wall_seconds": 0.02, "gpu_ms": 25.0, "calls": 8},
        "finalize": {"wall_seconds": 0.001, "gpu_ms": 1.25, "calls": 1},
    }
    actor = SimpleNamespace(_publication_timer=SimpleNamespace(finish=lambda: stages), _publication_step=7)
    receipt = namespace["read_publication_timing"](actor)
    assert receipt["hostname"] == socket.gethostname()
    assert receipt["pid"] == os.getpid()
    assert not actor._publication_timer.enabled
    assert receipt["stages"] is stages
    return receipt


def test_native_receiver_rpc_receipt_survives_transport_and_exports_origin(monkeypatch):
    import json
    from skyrl_train.weight_sync.publication_timing import record_receiver_publication_stages

    receipt = _native_receiver_receipt(monkeypatch)
    transported = json.loads(json.dumps([[receipt], [receipt]]))
    events = []
    monkeypatch.setattr(
        "skyrl_train.telemetry.record_event", lambda name, fields, **kwargs: events.append((name, fields, kwargs))
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda: pytest.fail("receipt transport cannot synchronize CUDA"))
    record_receiver_publication_stages(transported, step=7)
    assert len(events) == 6
    for engine_index in range(2):
        for index, (stage, values) in enumerate(receipt["stages"].items()):
            name, body, kwargs = events[3 * engine_index + index]
            assert name == "publication_stage"
            assert body == values
            assert kwargs["attributes"] == {
                "role": "inference",
                "exporter_role": "trainer",
                "engine_index": str(engine_index),
                "origin_host": receipt["hostname"],
                "origin_pid": str(receipt["pid"]),
                "rank": "3",
                "step": "7",
                "stage": stage,
            }
    assert publication_stage_walls({"trainer": [], "receiver": transported}) == {
        "weight_broadcast/recv": 0.01,
        "weight_broadcast/load": 0.02,
        "weight_broadcast/finalize": 0.001,
    }


@pytest.mark.parametrize("damage", ["missing", "step", "origin", "gpu", "wall", "stages", "duplicate"])
def test_receiver_receipt_export_rejects_incomplete_batch_atomically(monkeypatch, damage):
    from copy import deepcopy
    from skyrl_train.weight_sync.publication_timing import record_receiver_publication_stages

    good = _native_receiver_receipt(monkeypatch)
    bad = deepcopy(good)
    if damage == "step":
        bad["step"] = 6
    elif damage == "origin":
        bad["hostname"] = ""
    elif damage == "gpu":
        bad["stages"]["recv"]["gpu_ms"] = float("nan")
    elif damage == "wall":
        bad["stages"]["load"]["wall_seconds"] = -1
    elif damage == "stages":
        bad["stages"] = {}
    receipts = [[good], [bad]]
    if damage == "missing":
        receipts[1] = []
    elif damage == "duplicate":
        receipts[1].append(bad)
    events = []
    monkeypatch.setattr("skyrl_train.telemetry.record_event", lambda *args, **kwargs: events.append(args))
    with pytest.raises(ValueError):
        record_receiver_publication_stages(receipts, step=7)
    assert events == []
