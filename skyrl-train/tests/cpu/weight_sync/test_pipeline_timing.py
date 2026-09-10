"""Independent interval arithmetic and CUDA completion boundary contracts."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.pipeline_timing import CudaPipelineTiming, merged_intervals, overlap_seconds


def test_interval_union_and_intersection_do_not_count_same_stage_twice():
    export = [[0, 4], [1, 3], [6, 8]]
    send = [[2, 7]]
    assert merged_intervals(export) == [[0, 4], [6, 8]]
    assert overlap_seconds(export, send) == 3
    assert overlap_seconds(export, [[8, 10]]) == 0


@pytest.mark.parametrize("interval", [[-1, 2], [2, 1], [0, float("nan")], [float("inf"), float("inf")]])
def test_invalid_device_intervals_are_rejected(interval):
    with pytest.raises(ValueError):
        merged_intervals([interval])


def test_cuda_catalogue_uses_one_origin_and_joins_before_event_readback(monkeypatch):
    clock, waits = [0.0], []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.joined = False

        def record(self, stream):
            self.at = clock[0]

        def synchronize(self):
            self.joined = True

        def query(self):
            return self.joined

        def elapsed_time(self, other):
            return other.at - self.at

    stream = SimpleNamespace(wait_event=lambda event: waits.append(event))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    timer = CudaPipelineTiming("cpu")
    start = timer.start("export", stream)
    clock[0] = 2000
    send = timer.start("nccl_send", stream)
    clock[0] = 4000
    timer.end(start, stream)
    clock[0] = 6000
    timer.end(send, stream)
    receipt = timer.finish()
    assert waits == [timer.origin, timer.origin]
    assert receipt["stage_seconds"] == {"export": 4.0, "nccl_send": 4.0}
    assert receipt["export_send_overlap_seconds"] == 2.0
    assert receipt["events_complete"] and not timer.events
    with pytest.raises(RuntimeError, match="already complete"):
        timer.finish()
