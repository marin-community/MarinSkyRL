"""Opt-in GPU qualifier for Grug's pinned public Megatron optimizer seam.

This is a bounded qualification program, not a pytest test. Run ``check`` on a
CPU host before committing a frozen input. Run ``qualify`` only under the
reviewed one-H100 contract recorded in MarinSkyRL issue #313.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.optimizer import LayerWiseDistributedOptimizer, OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from skyrl_train.distributed.grug_muonh import AdamH, MuonH, grug_muonh_route


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "skyrl-train/tests/cpu/distributed/fixtures/grug_muonh_jax_golden.npz"
LOCKFILE = ROOT / "uv.lock"
EXPECTED_FIXTURE_SHA256 = "57a66c2b0d36f1fbaffe1646b016457b7b92773fbc631aac318ceac45c9cb387"
EXPECTED_LOCK_SHA256 = "ae294a553ed94a377b25d1e9674d2d20c9a5bc9148f550f1947d2590bf93c0fa"
EXPECTED_VERSIONS = {
    "megatron-bridge": "0.5.0",
    "megatron-core": "0.18.0",
    "torch": "2.11.0",
}
PARAMETER_NAMES = {
    "embed": "model.embed_tokens.weight",
    "q_proj": "model.layers.0.self_attn.q_proj.weight",
    "attn_gate": "model.layers.0.self_attn.attn_gate.weight",
    "router": "model.layers.0.mlp.router.weight",
    "expert": "model.layers.0.mlp.experts.gate_proj.weight",
    "shared": "model.layers.0.mlp.shared_expert.up_proj.weight",
    "gated_norm": "model.embed_gated_norm.down_proj.weight",
    "norm": "model.layers.0.input_layernorm.weight",
    "bias": "model.layers.0.bias",
    "output": "lm_head.weight",
}
ROUTE_INDEX = {"muonh": 0, "adamh": 1, "adam": 2}
TIGHT_RTOL = 2e-6
TIGHT_ATOL = 5e-7
MUON_PARAMETER_RTOL = 3e-3
MUON_PARAMETER_ATOL = 1.5e-3


class QualifierStageError(RuntimeError):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.stage = stage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution).split("+")[0]


def _assert_frozen_inputs() -> None:
    assert _sha256(FIXTURE) == EXPECTED_FIXTURE_SHA256
    assert _sha256(LOCKFILE) == EXPECTED_LOCK_SHA256
    for distribution, expected in EXPECTED_VERSIONS.items():
        assert _version(distribution) == expected, (distribution, _version(distribution), expected)


def _build_exact_routes(device: torch.device):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        identifiers = fixture["metadata_names"].tolist()
        parameters = {
            PARAMETER_NAMES[identifier]: nn.Parameter(
                torch.from_numpy(np.asarray(fixture[f"initial__{identifier}"]).copy()).to(device)
            )
            for identifier in identifiers
        }
        expected_routes = dict(zip(identifiers, fixture["metadata_routes"].tolist()))
        shared_lr = float(fixture["metadata_shared_lr"])
        adam_lr = float(fixture["metadata_adam_lr"])

    routed: dict[str, list[tuple[str, nn.Parameter]]] = {"muonh": [], "adamh": [], "adam": []}
    actual_routes = {}
    for identifier, name in PARAMETER_NAMES.items():
        route = grug_muonh_route(name, parameters[name])
        actual_routes[identifier] = route
        routed[route].append((name, parameters[name]))

    assert actual_routes == expected_routes
    assert sum(len(values) for values in routed.values()) == len(parameters)
    assert len({id(parameter) for values in routed.values() for _, parameter in values}) == len(parameters)

    def group(values, *, lr: float, is_expert_parallel: bool = False):
        return {
            "params": [parameter for name, parameter in values if (".experts." in name) == is_expert_parallel],
            "lr": lr,
            "lr_mult": lr / shared_lr,
            "wd_mult": 1.0,
            "is_expert_parallel": is_expert_parallel,
            "is_decoupled_lr": False,
        }

    muon_groups = [
        group(routed["muonh"], lr=shared_lr),
        group(routed["muonh"], lr=shared_lr, is_expert_parallel=True),
    ]
    optimizers = [
        MuonH(
            [value for value in muon_groups if value["params"]],
            lr=shared_lr,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            eps=1e-8,
        ),
        AdamH(
            [group(routed["adamh"], lr=shared_lr)],
            lr=shared_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        ),
        torch.optim.Adam(
            [group(routed["adam"], lr=adam_lr)],
            lr=adam_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    ]
    return parameters, optimizers, actual_routes


def _init_muon_state(optimizer, _config=None) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter].setdefault("momentum_buffer", torch.zeros_like(parameter))


def _init_adamh_state(optimizer, _config=None) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            state.setdefault("step", torch.zeros((), dtype=torch.int64, device=parameter.device))
            state.setdefault("exp_avg", torch.zeros_like(parameter))
            state.setdefault("exp_avg_sq", torch.zeros_like(parameter))


def _init_adam_state(optimizer, _config=None) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            state.setdefault("step", torch.zeros((), dtype=torch.float32, device=parameter.device))
            state.setdefault("exp_avg", torch.zeros_like(parameter))
            state.setdefault("exp_avg_sq", torch.zeros_like(parameter))


INIT_STATE_FNS = [_init_muon_state, _init_adamh_state, _init_adam_state]


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        lr=0.03,
        min_lr=0.0,
        weight_decay=0.0,
        bf16=True,
        fp16=False,
        params_dtype=torch.float32,
        use_distributed_optimizer=False,
        clip_grad=0.0,
        grad_norm_skip_threshold=float("inf"),
        log_num_zeros_in_grad=False,
        overlap_param_gather=False,
    )


def _build_public_optimizer(device: torch.device, groups: ProcessGroupCollection):
    parameters, optimizers, routes = _build_exact_routes(device)
    wrapped = LayerWiseDistributedOptimizer(
        optimizers,
        _optimizer_config(),
        pg_collection=groups,
        init_state_fn_list=INIT_STATE_FNS,
    )
    assert type(wrapped) is LayerWiseDistributedOptimizer
    assert len(wrapped.chained_optimizers) == 3
    return parameters, wrapped, routes


def _set_gradients(parameters: dict[str, nn.Parameter], fixture, step: int) -> None:
    for identifier, name in PARAMETER_NAMES.items():
        gradient = torch.from_numpy(np.asarray(fixture[f"gradient_{step}__{identifier}"]).copy())
        gradient = gradient.to(parameters[name].device)
        assert torch.isfinite(gradient).all()
        parameters[name].main_grad = gradient


def _expected(fixture, key: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(fixture[key]).copy()).to(device)


def _record_error(metrics: dict[str, list[torch.Tensor]], key: str, actual, expected) -> None:
    metrics.setdefault(key, []).append((actual.detach().float() - expected.detach().float()).abs().flatten().cpu())


def _assert_oracle_step(
    parameters: dict[str, nn.Parameter],
    wrapped: LayerWiseDistributedOptimizer,
    routes: dict[str, str],
    fixture,
    step: int,
    metrics: dict[str, list[torch.Tensor]],
) -> None:
    for identifier, name in PARAMETER_NAMES.items():
        parameter = parameters[name]
        route = routes[identifier]
        expected_parameter = _expected(fixture, f"parameter_{step}__{identifier}", parameter.device)
        if route == "muonh":
            torch.testing.assert_close(
                parameter,
                expected_parameter,
                rtol=MUON_PARAMETER_RTOL,
                atol=MUON_PARAMETER_ATOL,
            )
        else:
            torch.testing.assert_close(parameter, expected_parameter, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)
        _record_error(metrics, f"{route}.parameter", parameter, expected_parameter)

        state = wrapped.chained_optimizers[ROUTE_INDEX[route]].optimizer.state[parameter]
        if route == "muonh":
            expected_state = _expected(fixture, f"momentum_{step}__{identifier}", parameter.device)
            torch.testing.assert_close(state["momentum_buffer"], expected_state, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)
            _record_error(metrics, "muonh.state", state["momentum_buffer"], expected_state)
        elif route == "adamh":
            for state_key, fixture_key in (
                ("exp_avg", f"adamh_mu_{step}__{identifier}"),
                ("exp_avg_sq", f"adamh_nu_{step}__{identifier}"),
            ):
                expected_state = _expected(fixture, fixture_key, parameter.device)
                torch.testing.assert_close(state[state_key], expected_state, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)
                _record_error(metrics, "adamh.state", state[state_key], expected_state)
        else:
            for state_key, fixture_key in (
                ("exp_avg", f"adam_mu_{step}__{identifier}"),
                ("exp_avg_sq", f"adam_nu_{step}__{identifier}"),
            ):
                expected_state = _expected(fixture, fixture_key, parameter.device)
                torch.testing.assert_close(state[state_key], expected_state, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)
                _record_error(metrics, "adam.state", state[state_key], expected_state)


def _assert_trees_equal(left: Any, right: Any, path: str = "state") -> None:
    assert type(left) is type(right), (path, type(left), type(right))
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_trees_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _assert_trees_equal(left_value, right_value, f"{path}.{index}")
    else:
        assert left == right, (path, left, right)


def _assert_deterministic_skip(parameters, wrapped, fixture) -> float:
    parameter_snapshot = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    state_snapshot = copy.deepcopy(wrapped.state_dict())
    _set_gradients(parameters, fixture, 1)
    wrapped.config.grad_norm_skip_threshold = 0.0
    success, grad_norm, _ = wrapped.step()
    wrapped.config.grad_norm_skip_threshold = float("inf")
    assert success is False
    assert grad_norm is not None and torch.isfinite(torch.as_tensor(grad_norm)) and grad_norm > 0
    for name, parameter in parameters.items():
        assert torch.equal(parameter, parameter_snapshot[name]), name
    _assert_trees_equal(wrapped.state_dict(), state_snapshot)
    return float(grad_norm)


def _model_sharded_state(parameters: dict[str, nn.Parameter]):
    return {
        name: ShardedTensor.from_rank_offsets(name, parameter, replica_id=(0, 0, 0))
        for name, parameter in parameters.items()
    }


def _copy_loaded_model(parameters, loaded_model) -> None:
    for name, parameter in parameters.items():
        parameter.data.copy_(loaded_model[name])


def _run_step(parameters, wrapped, fixture, step: int):
    _set_gradients(parameters, fixture, step)
    success, grad_norm, _ = wrapped.step()
    assert success is True
    assert grad_norm is not None and torch.isfinite(torch.as_tensor(grad_norm))
    return float(grad_norm)


def _error_summary(metrics: dict[str, list[torch.Tensor]]) -> dict[str, dict[str, float]]:
    summary = {}
    for key, tensors in sorted(metrics.items()):
        errors = torch.cat(tensors)
        summary[key] = {
            "max_abs": float(errors.max()),
            "mean_abs": float(errors.mean()),
        }
    return summary


def _run_qualifier() -> dict[str, Any]:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if not dist.is_initialized():
        handle, rendezvous = tempfile.mkstemp(prefix="grug-megatron-qualifier-")
        os.close(handle)
        os.unlink(rendezvous)
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    assert dist.get_world_size() == 1
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    assert parallel_state.model_parallel_is_initialized()
    groups = ProcessGroupCollection(dp_cp=dist.group.WORLD, expt_dp=dist.group.WORLD)

    metrics: dict[str, list[torch.Tensor]] = {}
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        try:
            oracle_parameters, oracle_optimizer, routes = _build_public_optimizer(device, groups)
            oracle_grad_norms = []
            for step in range(1, int(fixture["metadata_steps"]) + 1):
                oracle_grad_norms.append(_run_step(oracle_parameters, oracle_optimizer, fixture, step))
                _assert_oracle_step(oracle_parameters, oracle_optimizer, routes, fixture, step, metrics)
                oracle_optimizer.zero_grad()
        except Exception as error:
            raise QualifierStageError("construction_or_three_step_oracle", error) from error

        try:
            skipped_grad_norm = _assert_deterministic_skip(oracle_parameters, oracle_optimizer, fixture)
        except Exception as error:
            raise QualifierStageError("deterministic_skipped_update", error) from error

        error_summary = _error_summary(metrics)
        print(
            json.dumps(
                {
                    "errors": error_summary,
                    "event": "oracle_and_skip_complete",
                    "oracle_grad_norms": oracle_grad_norms,
                    "routes": routes,
                    "skipped_grad_norm": skipped_grad_norm,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        try:
            checkpoint_parameters, checkpoint_optimizer, _ = _build_public_optimizer(device, groups)
            for step in (1, 2):
                _run_step(checkpoint_parameters, checkpoint_optimizer, fixture, step)
                checkpoint_optimizer.zero_grad()
        except Exception as error:
            raise QualifierStageError("checkpoint_source_steps", error) from error

        with tempfile.TemporaryDirectory(prefix="grug-megatron-dcp-") as checkpoint_dir:
            model_state = _model_sharded_state(checkpoint_parameters)
            try:
                optimizer_state = checkpoint_optimizer.sharded_state_dict(model_state)
            except Exception as error:
                raise QualifierStageError("public_sharded_state_materialize", error) from error
            try:
                dist_checkpointing.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_dir)
            except Exception as error:
                raise QualifierStageError("public_dcp_save", error) from error

            try:
                resumed_parameters, resumed_optimizer, _ = _build_public_optimizer(device, groups)
                resumed_model_state = _model_sharded_state(resumed_parameters)
                load_template = {
                    "model": resumed_model_state,
                    "optimizer": resumed_optimizer.sharded_state_dict(resumed_model_state, is_loading=True),
                }
                loaded = dist_checkpointing.load(load_template, checkpoint_dir)
                _copy_loaded_model(resumed_parameters, loaded["model"])
                resumed_optimizer.load_state_dict(loaded["optimizer"])

                _assert_trees_equal(resumed_optimizer.state_dict(), checkpoint_optimizer.state_dict())
                for name in checkpoint_parameters:
                    assert torch.equal(checkpoint_parameters[name], resumed_parameters[name]), name
            except Exception as error:
                raise QualifierStageError("public_dcp_rebuild_load", error) from error

            try:
                _run_step(checkpoint_parameters, checkpoint_optimizer, fixture, 3)
                _run_step(resumed_parameters, resumed_optimizer, fixture, 3)
                for name in checkpoint_parameters:
                    assert torch.equal(checkpoint_parameters[name], resumed_parameters[name]), name
                _assert_trees_equal(resumed_optimizer.state_dict(), checkpoint_optimizer.state_dict())
            except Exception as error:
                raise QualifierStageError("exact_next_step", error) from error

    return {
        "dcp_rebuild_load": "PASS",
        "errors": error_summary,
        "exact_next_step": "PASS",
        "oracle_grad_norms": oracle_grad_norms,
        "routes": routes,
        "skipped_grad_norm": skipped_grad_norm,
        "skipped_update": "PASS",
        "successful_updates": 3,
    }


def _metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "commit": os.environ.get("QUALIFIER_COMMIT", "unknown"),
        "fixture_sha256": _sha256(FIXTURE),
        "gpu": properties.name if properties else "none",
        "iris_job_id": os.environ.get("IRIS_JOB_ID", os.environ.get("QUALIFIER_JOB_ID", "unknown")),
        "lock_sha256": _sha256(LOCKFILE),
        "megatron_bridge": _version("megatron-bridge"),
        "megatron_core": _version("megatron-core"),
        "qualifier_sha256": _sha256(Path(__file__)),
        "topology": {"world_size": 1, "tp": 1, "ep": 1, "pp": 1, "cp": 1, "vp": 1},
        "torch": importlib.metadata.version("torch"),
        "tolerances": {
            "muon_parameter": {"rtol": MUON_PARAMETER_RTOL, "atol": MUON_PARAMETER_ATOL},
            "other_parameter_and_state": {"rtol": TIGHT_RTOL, "atol": TIGHT_ATOL},
        },
    }


def _raise_timeout(_signum, _frame) -> None:
    raise TimeoutError("qualifier exceeded its 18-minute self-termination limit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "qualify"))
    args = parser.parse_args()
    started = time.monotonic()
    stage = "frozen_inputs"
    try:
        _assert_frozen_inputs()
        if args.mode == "check":
            stage = "device_independent_routes"
            _, _, routes = _build_exact_routes(torch.device("cpu"))
            print(json.dumps({"metadata": _metadata(), "result": "PASS", "routes": routes}, sort_keys=True))
            return 0
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(18 * 60)
        stage = "public_optimizer_qualifier"
        result = _run_qualifier()
        print(
            json.dumps(
                {
                    "metadata": _metadata(),
                    "result": "PASS",
                    "runtime_seconds": time.monotonic() - started,
                    **result,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "failure_stage": getattr(error, "stage", stage),
                    "metadata": _metadata(),
                    "result": "FAIL",
                    "runtime_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            )
        )
        traceback.print_exc()
        return 1
    finally:
        if args.mode == "qualify" and dist.is_initialized():
            parallel_state.destroy_model_parallel()
            assert not parallel_state.model_parallel_is_initialized()
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
