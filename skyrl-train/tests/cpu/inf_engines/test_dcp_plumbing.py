"""Stage-1 plumbing/wiring test for vLLM Decode Context Parallel (DCP).

Asserts the engine-launch wiring, with NO GPU and NO real Ray actor / vLLM init:

  (seam / G5) `create_ray_wrapped_inference_engines_from_config` reads
      `cfg.generator.inference_engine_decode_context_parallel_size` and forwards it as
      `decode_context_parallel_size`. This single config-assembly seam is shared by both
      `BasePPOExp` (standard) and `TerminalBenchExp` entrypoints (both inherit
      `_setup_trainer`), so wiring here covers both (G5).

  (G1 byte-identity) `create_ray_wrapped_inference_engines` forwards
      `decode_context_parallel_size` to the vLLM actor `.remote(...)` ONLY when `> 1`.
      With dcp=1 (the default) the kwarg is ABSENT from the remote call → vLLM engine
      init is byte-identical to today. With dcp=2 it is present and `== 2`.

  (G4 GPU-neutrality) DCP does not change the PACK PG bundle count or the per-actor
      `bundle_indices` — DCP rides the TP GPUs and adds no GPUs. The captured bundle
      geometry is identical for dcp=1 vs dcp=2.

See notes/RL/skyrl/vllm_dcp_rollout_stages/stage1_vllm_support_and_plumbing_scope.md.

Run:
    uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_dcp_plumbing.py -v
"""

import sys
import types
import pytest
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config

DCP_KEY = "inference_engine_decode_context_parallel_size"


# ===================================================================== seam / G5
def test_from_config_forwards_dcp_value(monkeypatch):
    """The config-assembly seam reads the DCP key and forwards it as a kwarg.

    Covers both entrypoints (BasePPOExp + TerminalBenchExp) since they share this seam.
    """
    pytest.importorskip("hydra")
    # main_base transitively imports the trainer, which imports torchdata; on the Mac
    # dev box torchdata is absent (documented env artifact) -> skip there. On Jupiter /
    # the production runtime torchdata is present and this exercises the real seam.
    pytest.importorskip("torchdata", reason="torchdata absent (Mac dev-env artifact)")
    from skyrl_train.entrypoints import main_base

    captured = {}

    def fake_create(**engine_kwargs):
        captured.update(engine_kwargs)
        return []  # no engines

    # The function imports create_ray_wrapped_inference_engines lazily from this module,
    # so patch it at the source module.
    import skyrl_train.inference_engines.ray_wrapped_inference_engine as rwie

    monkeypatch.setattr(rwie, "create_ray_wrapped_inference_engines", fake_create)

    # dcp=1 (default): forwarded as 1 (the signature default => not passed to vLLM downstream).
    cfg = get_default_config()
    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)
    assert captured["decode_context_parallel_size"] == 1

    # dcp=2 with admissible TP: forwarded as 2.
    captured.clear()
    cfg2 = get_default_config()
    cfg2.generator.inference_engine_tensor_parallel_size = 8
    cfg2.generator[DCP_KEY] = 2
    main_base.create_ray_wrapped_inference_engines_from_config(cfg2, colocate_pg=None, tokenizer=None)
    assert captured["decode_context_parallel_size"] == 2


# ===================================================== remote forwarding G1 + G4
class _RemoteCapture:
    """Captures the kwargs passed to the (mocked) vLLM actor .options(...).remote(...)."""

    def __init__(self):
        self.remote_calls = []
        self.options_calls = []

    def make_actor_class(self):
        capture = self

        class _Actor:
            @staticmethod
            def options(**opts):
                capture.options_calls.append(opts)

                class _Bound:
                    @staticmethod
                    def remote(**kwargs):
                        capture.remote_calls.append(kwargs)
                        return object()  # a fake actor handle

                return _Bound()

        return _Actor


def _run_create(monkeypatch, dcp: int):
    """Drive the real create_ray_wrapped_inference_engines with Ray/PG/actor mocked.

    Uses tp=1, pp=1 (uni backend) so no real GPU/PG bundle reservation is needed; the
    placement_group + readiness + rendezvous + actor are all stubbed. Returns the
    _RemoteCapture so the caller can inspect the kwargs forwarded to .remote(...).
    """
    import skyrl_train.inference_engines.ray_wrapped_inference_engine as rwie

    capture = _RemoteCapture()

    # Stub the vllm import + actor classes (the module imports them lazily in the vllm branch).
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.20.2rc0.dev0+test"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_engine_mod = types.ModuleType("skyrl_train.inference_engines.vllm.vllm_engine")
    fake_engine_mod.VLLMRayActor = capture.make_actor_class()
    fake_engine_mod.AsyncVLLMRayActor = capture.make_actor_class()
    fake_engine_mod.WorkerWrap = object
    monkeypatch.setitem(sys.modules, "skyrl_train.inference_engines.vllm.vllm_engine", fake_engine_mod)

    # Stub Ray plumbing referenced inside create_ray_wrapped_inference_engines.
    class _FakeRemote:
        def remote(self):
            return None

    monkeypatch.setattr(rwie, "placement_group", lambda bundles, strategy=None: ("PG", tuple(len(bundles) for _ in [0])))
    # Stub the helpers pulled from skyrl_train.utils inside the function body.
    import skyrl_train.utils as skutils

    monkeypatch.setattr(skutils, "ray_noset_visible_devices", lambda *a, **k: False, raising=False)

    fake_get_all = types.SimpleNamespace(remote=lambda: None)
    monkeypatch.setattr(skutils, "get_all_env_variables", fake_get_all, raising=False)
    monkeypatch.setattr(skutils, "get_ray_pg_ready_with_timeout", lambda *a, **k: None, raising=False)

    # ray.get(...) is called on get_all_env_variables.remote() — return a dummy env dict.
    monkeypatch.setattr(rwie.ray, "get", lambda *a, **k: {})
    # get_rendezvous_addr_port is only used for data_parallel_size>1; stub anyway.
    monkeypatch.setattr(rwie, "get_rendezvous_addr_port", lambda pg, idx: ("127.0.0.1", 12345))
    # RayWrappedInferenceEngine wraps each handle — keep it trivial.
    monkeypatch.setattr(rwie, "RayWrappedInferenceEngine", lambda h: h)

    rwie.create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=1,
        model_dtype="bfloat16",
        pretrain="dummy/model",
        seed=0,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=dcp,
        shared_pg=None,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=False,
        async_engine=False,
        backend="vllm",
    )
    return capture


def test_dcp_disabled_kwarg_absent_from_remote(monkeypatch):
    """G1: dcp=1 => decode_context_parallel_size is NOT passed to the vLLM actor.

    The remote call (hence vllm.LLM/AsyncEngineArgs) is byte-identical to today.
    """
    capture = _run_create(monkeypatch, dcp=1)
    assert len(capture.remote_calls) == 1
    assert "decode_context_parallel_size" not in capture.remote_calls[0], (
        "dcp=1 must NOT forward decode_context_parallel_size (G1 byte-identity)"
    )


def test_dcp_enabled_kwarg_present_in_remote(monkeypatch):
    """dcp=2 => decode_context_parallel_size=2 is forwarded to the vLLM actor."""
    capture = _run_create(monkeypatch, dcp=2)
    assert len(capture.remote_calls) == 1
    assert capture.remote_calls[0].get("decode_context_parallel_size") == 2


def test_dcp_does_not_change_bundle_geometry(monkeypatch):
    """G4: per-actor bundle_indices / TP geometry identical for dcp=1 vs dcp=2.

    DCP rides the TP GPUs; it must not change placement-group / GPU sizing.
    """
    cap1 = _run_create(monkeypatch, dcp=1)
    cap2 = _run_create(monkeypatch, dcp=2)

    def geometry(cap):
        rc = cap.remote_calls[0]
        oc = cap.options_calls[0]
        return {
            "bundle_indices": rc.get("bundle_indices"),
            "tensor_parallel_size": rc.get("tensor_parallel_size"),
            "pipeline_parallel_size": rc.get("pipeline_parallel_size"),
            "num_gpus_option": oc.get("num_gpus"),
            "num_cpus_option": oc.get("num_cpus"),
        }

    assert geometry(cap1) == geometry(cap2), "DCP must not change GPU/placement geometry (G4)"
