import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import weakref

from omegaconf import OmegaConf
import pytest
import torch

from skyrl_train.weight_sync.publication_timing import PublicationStageTimer
from skyrl_train.weight_sync.wire_inventory import WireInventory


def test_inventory_counts_actual_mixed_dtype_bytes_without_retaining_tensors():
    inventory = WireInventory()
    tensor = torch.zeros((2, 3), dtype=torch.bfloat16)
    ref = weakref.ref(tensor)
    inventory.observe("projection", tensor)
    del tensor
    assert ref() is None
    inventory.observe("bias", torch.zeros(3, dtype=torch.float32))
    result = inventory.finish(completed_update=20)
    assert result["wire_bytes"] == 24 and result["tensors"] == 2
    assert [row["dtype"] for row in result["entries"]] == ["bfloat16", "float32"]
    assert result["completed_update"] == 20
    changed = WireInventory()
    changed.observe("bias", torch.zeros(3, dtype=torch.float32))
    changed.observe("projection", torch.zeros((2, 3), dtype=torch.bfloat16))
    assert changed.finish(completed_update=20)["ordered_metadata_sha256"] != result["ordered_metadata_sha256"]


def test_inventory_rejects_missing_duplicate_and_strided_tensors():
    inventory = WireInventory()
    with pytest.raises(ValueError, match="No successful"):
        inventory.finish(completed_update=0)
    inventory.observe("first", torch.zeros(2))
    with pytest.raises(ValueError, match="unique"):
        inventory.observe("first", torch.zeros(2))
    with pytest.raises(ValueError, match="contiguous"):
        inventory.observe("strided", torch.zeros((2, 3)).T)


def production_method():
    source = Path(__file__).resolve().parents[3] / "skyrl_train/workers/megatron/megatron_worker.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerBase")
    method = next(node for node in cls.body if node.name == "_broadcast_to_inference_engines")
    namespace = {
        "torch": torch,
        "asyncio": asyncio,
        "PublicationStageTimer": PublicationStageTimer,
        "WireInventory": WireInventory,
        "str_to_torch_dtype": lambda value: getattr(torch, value),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.asyncio
async def test_actual_native_broadcast_path_preserves_calls_and_invalidates_failed_receipt(monkeypatch):
    trace = []

    async def immediate_dispatch(function, *args):
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_dispatch)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: trace.append("barrier"))
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, src, group: trace.append(("broadcast", tensor.numel()))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class Chunk(SimpleNamespace):
        def __len__(self):
            return 1

    class Client:
        fail = False

        async def begin_weight_reload(self):
            trace.append("begin")

        async def finish_weight_reload(self):
            trace.append("finish")

        async def update_named_weights(self, metadata):
            trace.append(("load", metadata["names"][0]))
            if self.fail:
                raise RuntimeError("native receiver failure")

    tensors = [torch.zeros((2, 3), dtype=torch.bfloat16), torch.zeros(3, dtype=torch.float32)]
    chunks = [
        Chunk(names=[name], tensors=[tensor], dtypes=[str(tensor.dtype)])
        for name, tensor in zip(("projection", "bias"), tensors)
    ]
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "generator": {
                    "publication_stage_timing": False,
                    "weight_sync_wire_inventory": False,
                    "enable_prefix_caching": False,
                    "model_dtype": "bfloat16",
                    "fuse_weights": False,
                }
            }
        ),
        use_cuda_ipc=False,
        _completed_update=20,
        _model_update_group=object(),
        weight_extractor=SimpleNamespace(extract_weights=lambda dtype: iter(chunks)),
    )
    method = production_method()
    client = Client()
    await method(worker, client)
    original_calls = trace[:]
    assert not hasattr(worker, "_weight_sync_wire_receipt")
    trace.clear()
    worker.cfg.generator.weight_sync_wire_inventory = True
    await method(worker, client)
    assert trace == original_calls
    assert worker._weight_sync_wire_receipt["wire_bytes"] == 24
    assert worker._weight_sync_wire_receipt["tensors"] == 2
    client.fail = True
    with pytest.raises(RuntimeError, match="native receiver failure"):
        await method(worker, client)
    assert worker._weight_sync_wire_receipt is None
