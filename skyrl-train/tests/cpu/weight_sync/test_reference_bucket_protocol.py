"""Reference lifecycle with actual WorkerWrap load hook and independent CPU values."""

import ast
from contextlib import nullcontext
import gc
from pathlib import Path
import weakref

import pytest
import torch

from skyrl_train.weight_sync import reference_bucket_protocol as reference
from skyrl_train.weight_sync.vllm_weight_conversion import load_weights_into_vllm
from tests.cpu.weight_sync.test_worker_bucket_protocol import native_protocol as protocol_fixture, prepare

native_protocol = protocol_fixture


def attach_actual_load_method(worker):
    path = Path(__file__).parents[3] / "skyrl_train/inference_engines/vllm/vllm_engine.py"
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "WorkerWrap")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights")
    namespace = {"NamedWeightsUpdateRequest": dict, "load_weights_into_vllm": load_weights_into_vllm}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    worker.load_weights = namespace["load_weights"].__get__(worker)
    worker._publication_timer = type("Timer", (), {"span": lambda self, name: nullcontext()})()


def replace_and_load(case, *, missing=False, fail_load=False, reverse=False):
    manifest, sources, parameters, maps, _ = case.parts
    old = {name: weakref.ref(tensor) for name, tensor in parameters.items()}
    new = {name: torch.full_like(tensor, -9) for name, tensor in parameters.items()}
    parameters.clear()
    parameters.update(new)
    prefix = next(iter(maps))
    case.module.w13_weight = new[prefix + ".w13_weight"]
    case.module.w2_weight = new[prefix + ".w2_weight"]
    gc.collect()
    assert all(ref() is None for ref in old.values()), "prepared proof retained the previous installed model"

    def load(weights):
        if fail_load:
            raise RuntimeError("original loader failed")
        loaded = set()
        for name, tensor in weights:
            if ".experts." not in name:
                parameters[name].copy_(tensor)
                loaded.add(name)
                continue
            for global_id, local_id in enumerate(maps[prefix]):
                if local_id < 0:
                    continue
                if ".gate_proj." in name:
                    parameters[prefix + ".w13_weight"][local_id, :2].copy_(tensor[global_id])
                elif ".up_proj." in name:
                    parameters[prefix + ".w13_weight"][local_id, 2:].copy_(tensor[global_id])
                else:
                    parameters[prefix + ".w2_weight"][local_id].copy_(tensor[global_id])
            loaded.add(name)
        return loaded

    case.worker.model_runner.model.load_weights = load
    attach_actual_load_method(case.worker)
    case.worker._skyrl_weight_update_active = True
    values = list(sources.items())
    if missing:
        values.pop()
    if reverse:
        values.reverse()
    case.worker._weight_receiver = type("Receiver", (), {"receive_weights": lambda self, request: iter(values)})()
    case.worker.load_weights({})
    case.worker._skyrl_weight_update_active = False
    return new


def test_reference_rebinds_new_storage_after_actual_load_hook_then_full_replay(native_protocol):
    case = native_protocol
    prepare(case)
    identity = {"manifest_id": case.parts[0].manifest_id, "publication_id": 1}
    state = case.worker._diagnostic_bucket_state
    pointers = [value.data_ptr() for value in state["receiver"].buffers]
    reference.begin_reference_sync(case.worker, **identity)
    with pytest.raises(ValueError, match="storage is released"):
        case.worker.receive_diagnostic_weight_sync_bucket(0, replay=True, **identity)
    replace_and_load(case)
    receipt = reference.finish_reference_sync(case.worker, **identity)
    assert receipt["original_reload_complete"]
    assert receipt["wire_inventory"]["wire_bytes"] == sum(entry.nbytes for entry in case.parts[0].entries)
    assert pointers == [value.data_ptr() for value in state["receiver"].buffers]
    assert state["receiver"]._next_install == 0  # No fabricated bucket installs.
    for bucket in range(case.parts[0].bucket_count):
        case.worker.receive_diagnostic_weight_sync_bucket(bucket, replay=True, **identity)
    proof = case.worker.finish_diagnostic_weight_sync_replay(**identity)
    assert proof["compared_bytes"] == receipt["installed_parameter_bytes"]
    assert proof["mismatches"] == 0
    reference.begin_reference_sync(case.worker, manifest_id=identity["manifest_id"], publication_id=2)
    assert pointers == [value.data_ptr() for value in state["receiver"].buffers]


@pytest.mark.parametrize(
    "fault", ["missing", "failed_load", "active_reload", "changed_layout", "changed_map", "reordered"]
)
def test_reference_completion_rejects_incomplete_or_changed_original_load(native_protocol, fault):
    case = native_protocol
    prepare(case)
    identity = {"manifest_id": case.parts[0].manifest_id, "publication_id": 1}
    reference.begin_reference_sync(case.worker, **identity)
    if fault == "failed_load":
        with pytest.raises(RuntimeError, match="original loader failed"):
            replace_and_load(case, fail_load=True)
        assert not case.worker._diagnostic_bucket_state["reference_inventory"].entries
        return
    replace_and_load(case, missing=fault == "missing", reverse=fault == "reordered")
    if fault == "active_reload":
        case.worker._skyrl_weight_update_active = True
    if fault == "changed_layout":
        name = "model.embed_tokens.weight"
        case.parts[2][name] = case.parts[2][name].reshape(2, 8)
    if fault == "changed_map":
        case.module._map_global_expert_id_to_local_expert_id = lambda expert: -1
    with pytest.raises(ValueError, match="inventory|must finish|layout|placement"):
        reference.finish_reference_sync(case.worker, **identity)
    assert not case.worker._diagnostic_bucket_state["install_complete"]
