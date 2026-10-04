from contextlib import nullcontext
import json
from types import SimpleNamespace

import torch

from skyrl_train.utils.profiler import Profiler


def test_profiler_saves_forward_and_backward_after_one_warm_update(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    profiler = Profiler(SimpleNamespace(ranks=[0], save_path=str(tmp_path)))

    for update_index in range(2):
        profiler.begin_update()
        for mini_batch_index in range(2):
            selected = profiler.for_mini_batch(mini_batch_index)
            with selected.capture_training() if selected is not None else nullcontext():
                for micro_batch_index in range(2):
                    with torch.profiler.record_function(
                        f"update_{update_index}_mini_{mini_batch_index}_micro_{micro_batch_index}"
                    ):
                        x = torch.ones(4, requires_grad=True)
                        value = x.square().sum()
                    with torch.profiler.record_function(f"backward_update_{update_index}_mini_{mini_batch_index}"):
                        value.backward()
        profiler.save()

    table = (tmp_path / "prof_rank_0.txt").read_text()
    assert "update_1_mini_0_micro_0" in table
    assert "update_0_mini_0_micro_0" not in table
    assert "update_1_mini_0_micro_1" in table
    assert "update_1_mini_1_micro_0" not in table
    trace = json.loads((tmp_path / "prof_rank_0.json").read_text())
    names = {event["name"] for event in trace["traceEvents"]}
    assert "backward_update_1_mini_0" in names
    assert "backward_update_0_mini_0" not in names
    assert "backward_update_1_mini_1" not in names
    assert "PowBackward0" in names
