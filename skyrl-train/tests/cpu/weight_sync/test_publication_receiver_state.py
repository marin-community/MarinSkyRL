import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Backend(Enum):
    TRITON = "triton"


def test_actual_worker_probe_preserves_parameters_and_reports_unknown_backend(monkeypatch):
    class Experts(torch.nn.Module):
        def __init__(self, backend):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.arange(24).reshape(2, 4, 3).bfloat16())
            self.w2_weight = torch.nn.Parameter(torch.arange(12).reshape(2, 3, 2).bfloat16())
            self.quant_method = SimpleNamespace(unquantized_backend=backend)

        def _map_global_expert_id_to_local_expert_id(self, expert):
            return expert - 2 if expert >= 2 else -1

    model = torch.nn.ModuleList([Experts(Backend.TRITON), Experts(None)])
    originals = [(p.data_ptr(), p.clone()) for p in model.parameters()]
    synchronized = False

    def synchronize(device):
        nonlocal synchronized
        assert device == "cpu"
        synchronized = True

    def memory(device):
        assert synchronized and device == "cpu"
        return 3 * 2**30, 80 * 2**30

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 7)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 11)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("readback allocated a tensor"))
    source = Path(__file__).parents[3] / "skyrl_train/inference_engines/vllm/vllm_engine.py"
    worker_class = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "WorkerWrap"
    )
    method = next(
        n for n in worker_class.body if isinstance(n, ast.FunctionDef) and n.name == "read_publication_receiver_state"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    worker = SimpleNamespace(
        device="cpu",
        model_runner=SimpleNamespace(model=model),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="grug_moe", num_local_experts=4))
        ),
    )
    receipt = namespace["read_publication_receiver_state"](worker)
    assert receipt["free_bytes"] == 3 * 2**30 and receipt["total_bytes"] == 80 * 2**30
    assert receipt["allocated_bytes"] == 7 and receipt["reserved_bytes"] == 11
    assert [layer["backend"] for layer in receipt["layers"]] == ["TRITON", None]
    assert all(layer["expert_map"] == [-1, -1, 0, 1] for layer in receipt["layers"])
    assert receipt["layers"][0]["parameters"]["w13_weight"]["shape"] == [2, 4, 3]
    for parameter, (pointer, value) in zip(model.parameters(), originals, strict=True):
        assert parameter.data_ptr() == pointer and torch.equal(parameter, value)
