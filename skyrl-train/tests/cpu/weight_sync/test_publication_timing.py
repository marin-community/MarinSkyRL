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
