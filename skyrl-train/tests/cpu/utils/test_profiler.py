import json
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.utils.profiler import Profiler


@pytest.mark.parametrize("export_chrome_trace", [False, True])
def test_profiler_captures_one_forward_micro_batch_after_warm_update(tmp_path, monkeypatch, export_chrome_trace):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    config = SimpleNamespace(
        enable=True,
        ranks=[0],
        save_path=str(tmp_path),
        capture_update_index=1,
        capture_mini_batch_index=0,
        record_shapes=False,
        with_stack=False,
        export_chrome_trace=export_chrome_trace,
    )
    profiler = Profiler(config)

    profiler.begin_update()
    profiler.start_capture_if_selected(0)
    with torch.profiler.record_function("outside_before"):
        torch.ones(4).sum()
    profiler.stop_capture()
    profiler.save()
    assert not list(tmp_path.iterdir())

    profiler.begin_update()
    profiler.start_capture_if_selected(0)
    with torch.profiler.record_function("selected_forward_micro_batch"):
        torch.ones(4).sum()
    profiler.stop_capture()

    with torch.profiler.record_function("outside_after"):
        torch.ones(4).sum()
    profiler.save()

    table = (tmp_path / "prof_rank_0.txt").read_text()
    assert "selected_forward_micro_batch" in table
    assert "outside_before" not in table
    assert "outside_after" not in table
    metadata = json.loads((tmp_path / "prof_rank_0_metadata.json").read_text())
    assert metadata["capture_update_index"] == 1
    assert metadata["capture_mini_batch_index"] == 0
    assert metadata["capture_wall_seconds"] > 0
    assert 0 <= metadata["stop_wall_seconds"] <= metadata["capture_wall_seconds"]
    assert metadata["peak_rss_bytes_after_export"] > 0
    if export_chrome_trace:
        trace = json.loads((tmp_path / "prof_rank_0.json").read_text())
        names = {event["name"] for event in trace["traceEvents"] if "name" in event}
        assert "selected_forward_micro_batch" in names
        assert "outside_before" not in names
        assert "outside_after" not in names
        assert metadata["trace_bytes"] == (tmp_path / "prof_rank_0.json").stat().st_size
    else:
        assert metadata["trace_bytes"] == 0
        assert not (tmp_path / "prof_rank_0.json").exists()
