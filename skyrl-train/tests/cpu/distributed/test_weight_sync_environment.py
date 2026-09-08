"""Exercise the pinned override, actual worker hooks, and vLLM extension MRO on CPU."""

import ast
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from loguru import logger
from omegaconf import OmegaConf
import pytest
import torch

from skyrl_train.distributed import utils as distributed_utils
from skyrl_train.distributed import weight_sync_environment as control
from tests.invariant_environment_reference import (
    PINNED_WORKER_METHODS,
    override_envs_for_invariance,
    resolve_extended_worker,
)


@pytest.fixture
def pinned_override(monkeypatch):
    module = ModuleType("vllm.model_executor.layers.batch_invariant")
    module.override_envs_for_invariance = override_envs_for_invariance
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "0")
    for key in control.EXPECTED_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return module


def test_actual_override_readback_is_allowlisted_and_pre_pg(pinned_override, capfd, monkeypatch):
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "must-never-be-exported")
    control.apply_weight_sync_environment(True, role="trainer")
    line = capfd.readouterr().out
    receipt = json.loads(line.removeprefix("WEIGHT_SYNC_ENVIRONMENT_PRE_PG "))
    assert receipt["values"] == control.EXPECTED_ENVIRONMENT
    assert receipt["default_process_group_initialized"] is False
    assert receipt["override_source_sha256"] == control.OVERRIDE_SHA256
    assert receipt["origin_pid"] == os.getpid()
    assert "UNRELATED_TEST_SECRET" not in line and "must-never-be-exported" not in line


def test_disabled_path_does_not_import_mutate_or_emit(monkeypatch, capfd):
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.batch_invariant", None)
    before = dict(os.environ)
    control.apply_weight_sync_environment(False, role="trainer")
    assert dict(os.environ) == before
    assert capfd.readouterr().out == ""


@pytest.mark.parametrize("failure", ["environment", "already_initialized", "source"])
def test_enabled_path_fails_closed(pinned_override, monkeypatch, failure):
    if failure == "environment":
        monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    elif failure == "already_initialized":
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    else:
        monkeypatch.setattr(control, "OVERRIDE_SHA256", "0" * 64)
    with pytest.raises((ValueError, RuntimeError)):
        control.apply_weight_sync_environment(True, role="trainer")
    assert all(key not in os.environ for key in control.EXPECTED_ENVIRONMENT)


def test_actual_trainer_device_and_pg_hooks_observe_override(pinned_override, monkeypatch):
    calls = []

    def observe(label):
        assert {key: os.environ.get(key) for key in control.EXPECTED_ENVIRONMENT} == control.EXPECTED_ENVIRONMENT
        calls.append(label)

    monkeypatch.setattr(torch.cuda, "set_device", lambda _: observe("device"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **_: observe("default_pg"))
    distributed_utils.init_worker_process_group_with_device(10, weight_sync_invariant_env=True)
    assert calls == ["device", "default_pg"]


def _load_class_from_source(path, class_name):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


def test_actual_receiver_subclass_keeps_extension_and_runs_before_super(pinned_override, monkeypatch, capfd):
    calls = []

    class GPUWorker:
        rank = 5
        local_rank = 2

        def init_device(self):
            assert {key: os.environ.get(key) for key in control.EXPECTED_ENVIRONMENT} == control.EXPECTED_ENVIRONMENT
            calls.append("native_init_device")
            return "initialized"

    gpu_module = ModuleType("vllm.v1.worker.gpu_worker")
    gpu_module.Worker = GPUWorker
    monkeypatch.setitem(sys.modules, gpu_module.__name__, gpu_module)
    name = "skyrl_train.inference_engines.vllm.invariant_worker"
    monkeypatch.delitem(sys.modules, name, raising=False)
    worker = importlib.import_module(name).InvariantWeightSyncWorker
    source = Path(control.__file__).parents[1] / "inference_engines/vllm/vllm_engine.py"
    extension = _load_class_from_source(source, "WorkerWrap")
    assert not set(PINNED_WORKER_METHODS).intersection(name for name in dir(extension) if not name.startswith("__"))
    extension_name = "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap"
    names = {control.WORKER_CLASS: worker, extension_name: extension}
    config = SimpleNamespace(worker_cls=control.WORKER_CLASS, worker_extension_cls=extension_name)
    resolved = resolve_extended_worker(config, names.__getitem__, logger)
    assert resolved is worker
    monkeypatch.delenv("RANK")
    monkeypatch.delenv("LOCAL_RANK")
    assert resolved().init_device() == "initialized"
    receipt = json.loads(capfd.readouterr().out.removeprefix("WEIGHT_SYNC_ENVIRONMENT_PRE_PG "))
    assert (receipt["rank"], receipt["local_rank"]) == (5, 2)
    assert receipt["environment_rank"] is None and receipt["environment_local_rank"] is None
    assert calls == ["native_init_device"]
    assert resolved().test_rpc(7, named=8) == ((7,), {"named": 8})
    assert resolved.__bases__ == (GPUWorker, extension)
    assert resolve_extended_worker(config, names.__getitem__, logger) is resolved
    monkeypatch.delitem(sys.modules, name)


@pytest.mark.parametrize(
    "worker_path,class_name",
    [
        ("workers/worker.py", "DistributedTorchRayActor"),
        ("workers/megatron/megatron_worker.py", "MegatronPolicyWorkerBase"),
        ("workers/megatron/megatron_worker.py", "MegatronRefWorkerBase"),
    ],
)
def test_actual_worker_methods_forward_opt_in(worker_path, class_name, monkeypatch):
    path = Path(control.__file__).parents[1] / worker_path
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "init_worker_process_group")
    seen = []

    class ReachedProcessGroup(Exception):
        pass

    def capture(**kwargs):
        seen.append(kwargs)
        raise ReachedProcessGroup

    namespace = {"init_worker_process_group_with_device": capture}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    for enabled in (False, True):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {"weight_sync_invariant_env": enabled},
                    "distributed": {"worker_collective_timeout_seconds": 60},
                }
            }
        )
        with pytest.raises(ReachedProcessGroup):
            namespace[method.name](SimpleNamespace(cfg=cfg))
    assert [item["weight_sync_invariant_env"] for item in seen] == [False, True]


@pytest.mark.parametrize("change", [None, "flag", "class", "remote", "backend", "environment"])
def test_config_requires_paired_control_and_local_vllm(monkeypatch, change):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    cfg = OmegaConf.create(
        {
            "trainer": {"algorithm": {"weight_sync_invariant_env": True}},
            "generator": {
                "backend": "vllm",
                "run_engines_locally": True,
                "engine_init_kwargs": {"worker_cls": control.WORKER_CLASS},
            },
        }
    )
    if change == "flag":
        cfg.trainer.algorithm.weight_sync_invariant_env = False
    elif change == "class":
        cfg.generator.engine_init_kwargs.worker_cls = "auto"
    elif change == "remote":
        cfg.generator.run_engines_locally = False
    elif change == "backend":
        cfg.generator.backend = "sglang"
    elif change == "environment":
        monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    if change:
        with pytest.raises(ValueError):
            control.validate_weight_sync_environment_config(cfg)
    else:
        control.validate_weight_sync_environment_config(cfg)
    cfg.trainer.algorithm.weight_sync_invariant_env = False
    cfg.generator.engine_init_kwargs = {}
    control.validate_weight_sync_environment_config(cfg)
