"""Real CPU Adam states; CUDA/export and optional Megatron protocol boundaries.

These tests do not qualify the native GPU adapter or numerical precision arms.
"""

from dataclasses import replace
from types import SimpleNamespace

from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
import pytest
from rigging.telemetry import serialization
import torch
import yaml

from skyrl_train import optimizer_state_metrics as metrics
from skyrl_train import telemetry as training_telemetry
from skyrl_train.config.utils import CONFIG_DIR, get_default_config
from skyrl_train.distributed.megatron.optimizer import megatron_optimizer_kwargs
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import example_dummy_config


def adam_state(dtype=torch.float32):
    model = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 4, dtype=dtype)).float().sum().backward()
    optimizer.step()
    model.weight.main_grad = model.weight.grad
    part = metrics.OptimizerStatePart(
        parameters=[model.weight],
        state=optimizer.state,
        scales={},
        overflow_buffer=None,
        master_weights=False,
        decoupled_grad=False,
        settings={"exp_avg_dtype": str(dtype), "exp_avg_sq_dtype": str(dtype)},
    )
    return model, optimizer, part


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_adam_state_reports_actual_dtypes_without_double_counting_storage(dtype):
    model, optimizer, part = adam_state(dtype)
    inventory = metrics.collect_optimizer_inventory(list(model.parameters()), [part])
    rows = {row["category"]: row for row in inventory["rows"]}
    element_bytes = 8 * model.weight.element_size()
    assert inventory["complete"]
    for category in ("model_parameter", "model_main_grad", "optimizer_parameter", "optimizer_grad", "master_state"):
        assert rows[category]["dtype"] == str(dtype)
        assert rows[category]["logical_bytes"] == element_bytes
    assert rows["exp_avg"]["logical_bytes"] == rows["exp_avg_sq"]["logical_bytes"] == element_bytes
    assert rows["other_optimizer_state"]["logical_bytes"] == optimizer.state[model.weight]["step"].element_size()
    # Parameter, gradient, two moments, scalar step: role aliases add no storage.
    assert inventory["unique_storage_count"] == 5
    assert inventory["unique_retained_storage_bytes"] == 4 * element_bytes + 4
    assert inventory["unique_cpu_storage_bytes"] == 4 * element_bytes + 4
    assert inventory["unique_cuda_storage_bytes"] == 0
    assert inventory["logical_bytes_including_role_aliases"] == 7 * element_bytes + 4
    assert inventory["coverage"]["optimizer_parameters"] == inventory["coverage"]["exp_avg_present"] == 1


def test_views_count_entire_backing_storage_once_across_dtype_and_role_aliases():
    buffer = torch.zeros(128, dtype=torch.float32)
    first = buffer[32:40]
    integer_alias = first.view(torch.int32)
    inventory = metrics.StorageInventory()
    inventory.add("master_state", first)
    inventory.add("master_state", first)
    inventory.add("optimizer_parameter", integer_alias)
    result = inventory.summary()
    assert result["unique_retained_storage_bytes"] == 512
    assert result["logical_bytes_including_role_aliases"] == 64
    assert result["unique_storage_count"] == 1
    assert {row["dtype"] for row in result["rows"]} == {"torch.float32", "torch.int32"}
    assert all(row["tensor_count"] == 1 and row["unique_retained_storage_bytes"] == 512 for row in result["rows"])


@pytest.mark.parametrize("name", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype"])
def test_missing_or_malformed_moments_cannot_qualify_even_when_scale_is_present(name, corruption):
    model, optimizer, part = adam_state()
    state = optimizer.state[model.weight]
    if corruption == "missing":
        del state[name]
    elif corruption == "shape":
        state[name] = state[name].flatten()
    else:
        state[name] = state[name].bfloat16()
    part = replace(part, scales={model.weight: {name: torch.ones(1)}})
    result = metrics.collect_optimizer_inventory(list(model.parameters()), [part])
    assert result["complete"] is False
    expected = "_present" if corruption == "missing" else f"_{corruption}_mismatch"
    assert result["coverage"][name + expected] == (0 if corruption == "missing" else 1)


def test_decoupled_gradient_and_raw_remainder_are_recorded_without_reconstruction():
    model, optimizer, part = adam_state(torch.bfloat16)
    model.weight.grad = None
    model.weight.decoupled_grad = torch.ones(2, 4, dtype=torch.float32)
    optimizer.state[model.weight]["master_param"] = torch.zeros(2, 4, dtype=torch.int16)
    part = replace(part, master_weights=True, decoupled_grad=True)
    result = metrics.collect_optimizer_inventory(list(model.parameters()), [part])
    rows = {row["category"]: row for row in result["rows"]}
    assert result["complete"]
    assert rows["optimizer_grad"]["dtype"] == "torch.float32"
    assert rows["optimizer_grad"]["logical_bytes"] == 32
    assert rows["master_state"]["dtype"] == "torch.int16"
    assert rows["master_state"]["logical_bytes"] == 16


@pytest.mark.parametrize("missing", ["gradient", "master", "component", "empty_tensor", "all"])
def test_empty_or_partially_initialized_inventory_is_explicitly_incomplete(missing):
    model, optimizer, part = adam_state()
    if missing == "gradient":
        model.weight.grad = None
    elif missing == "master":
        part = replace(part, master_weights=True)
    parts = [part]
    if missing == "component":
        parts.append(replace(part, parameters=[]))
    elif missing == "all":
        parts = []
    elif missing == "empty_tensor":
        empty = torch.nn.Parameter(torch.empty(0))
        empty.grad = torch.empty(0)
        parts = [
            replace(part, parameters=[empty], state={empty: {"exp_avg": torch.empty(0), "exp_avg_sq": torch.empty(0)}})
        ]
    result = metrics.collect_optimizer_inventory(list(model.parameters()), parts)
    assert result["complete"] is False


def megatron_protocol(model, optimizer):
    """Wrap real CPU Adam state in the optional native runtime's metadata protocol."""
    optimizer.master_weights = False
    optimizer.use_decoupled_grad = False
    optimizer.store_param_remainders = False
    optimizer._scales = {}
    optimizer._dummy_overflow_buf = None
    component = SimpleNamespace(
        optimizer=optimizer,
        config=SimpleNamespace(
            params_dtype=torch.float32,
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            use_precision_aware_optimizer=False,
            optimizer_cuda_graph=False,
        ),
        ddp_config=SimpleNamespace(grad_reduce_in_fp32=False, average_in_collective=True),
    )
    return [SimpleNamespace(module=model)], SimpleNamespace(chained_optimizers=[component])


def observations(monkeypatch):
    events = []

    def event(name, body, *, attributes):
        serialization.validate_attributes(attributes)
        fields = serialization.event_fields(body, budget=16_384)
        serialization.json_bytes({"body": fields, "attributes": attributes})
        events.append({"name": name, "body": fields, "attributes": dict(attributes)})

    # No synchronization/reset methods: the observer must inspect metadata only.
    cuda = SimpleNamespace(
        current_device=lambda: 2,
        get_device_properties=lambda device: SimpleNamespace(uuid="GPU-test-two"),
        memory_stats=lambda device: {"allocated_bytes.all.current": 1234, "reserved_bytes.all.current": 2048},
    )
    monkeypatch.setattr(metrics.torch, "cuda", cuda)
    monkeypatch.setattr(training_telemetry.telemetry, "event", event)
    return events


def test_observer_waits_for_success_then_emits_one_bounded_complete_inventory(monkeypatch):
    # Adam initializes Torch's accelerator support; create it before CUDA is replaced.
    model, optimizer, _ = adam_state()
    chunks, native = megatron_protocol(model, optimizer)
    with monkeypatch.context() as patch:
        events = observations(patch)
        observer = metrics.OptimizerStateObserver(enabled=True, rank=7)
        observer.after_step(False, model_chunks=None, optimizer=None, step=1, minibatch=1)
        assert events == []
        observer.after_step(True, model_chunks=chunks, optimizer=native, step=2, minibatch=1)
        # State has now been cleared, but the one-shot observer cannot overwrite evidence.
        optimizer.zero_grad()
        observer.after_step(True, model_chunks=None, optimizer=None, step=3, minibatch=1)
    summary = events[-1]
    assert summary["name"] == "optimizer_state_inventory"
    assert summary["body"]["complete"] is True
    assert summary["body"]["skipped_update_attempts_before_inventory"] == 1
    assert summary["body"]["allocated_bytes"] == 1234 and summary["body"]["reserved_bytes"] == 2048
    assert summary["attributes"]["rank"] == "7" and summary["attributes"]["step"] == "2"
    assert summary["attributes"]["boundary"] == "after_optimizer_step_before_zero_grad"
    assert summary["attributes"]["gpu_uuid"] == "GPU-test-two"
    rows = [event for event in events if event["name"] == "optimizer_state_storage"]
    assert len(rows) == summary["body"]["storage_row_count"] == 8
    assert all("parameter_name" not in row["body"] and "data_ptr" not in row["body"] for row in rows)
    setting = next(event["body"] for event in events if event["name"] == "optimizer_state_settings")
    assert setting["main_grads_dtype"] == "torch.float32"
    assert setting["grad_reduce_in_fp32"] is False  # Actual DDP flag, independent of that declared dtype.
    assert setting["average_in_collective"] is True
    assert setting["optimizer_class"] == "torch.optim.adamw.AdamW"


def test_disabled_observer_needs_no_native_runtime_or_cuda(monkeypatch):
    events = observations(monkeypatch)
    observer = metrics.OptimizerStateObserver(enabled=False, rank=0)
    observer.after_step(True, model_chunks=None, optimizer=None, step=1, minibatch=1)
    assert events == []


def test_failed_export_never_emits_complete_summary_or_changes_optimizer_result(monkeypatch):
    model, optimizer, _ = adam_state()
    chunks, native = megatron_protocol(model, optimizer)
    state_before = model.weight.detach().clone()
    with monkeypatch.context() as patch:
        events = observations(patch)

        def unavailable(name, body, *, attributes):
            raise OSError("export unavailable")

        patch.setattr(training_telemetry.telemetry, "event", unavailable)
        observer = metrics.OptimizerStateObserver(enabled=True, rank=0)
        observer.after_step(True, model_chunks=chunks, optimizer=native, step=1, minibatch=1)
        observer.after_step(True, model_chunks=None, optimizer=None, step=2, minibatch=1)
    assert events == []
    assert torch.equal(model.weight, state_before)
    assert model.weight.grad is not None


@pytest.mark.parametrize(
    "enabled,strategy,spans", [(1, "megatron", True), (True, "fsdp2", True), (True, "megatron", False)]
)
def test_inventory_requires_boolean_opt_in_megatron_and_phase_memory_observation(enabled, strategy, spans):
    cfg = get_default_config()
    cfg.trainer.optimizer_state_metrics = enabled
    cfg.trainer.strategy = strategy
    cfg.trainer.policy_train_spans = spans
    with pytest.raises(ValueError, match="optimizer_state_metrics"):
        validate_cfg(cfg)


@pytest.mark.parametrize("enabled", [False, True])
def test_inventory_opt_in_preserves_optimizer_precision_and_gradient_reduction(enabled):
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.optimizer_state_metrics = enabled
    cfg.trainer.policy_train_spans = True
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.micro_forward_batch_size_per_gpu = cfg.trainer.micro_train_batch_size_per_gpu
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.num_inference_engines = 1
    before = OmegaConf.to_container(cfg.trainer.policy.megatron_config, resolve=True)
    validate_cfg(cfg)
    assert OmegaConf.to_container(cfg.trainer.policy.megatron_config, resolve=True) == before
    assert before["ddp_config"]["grad_reduce_in_fp32"] is True
    assert before["optimizer_config_kwargs"]["use_precision_aware_optimizer"] is False


@pytest.mark.parametrize("second", ["float32", "bfloat16"])
def test_launcher_optimizer_kwargs_compose_and_reach_native_dtype_validation(tmp_path, second):
    declared = {
        "use_precision_aware_optimizer": True,
        "optimizer_cuda_graph": False,
        "store_param_remainders": False,
        "main_params_dtype": "float32",
        "main_grads_dtype": "float32",
        "exp_avg_dtype": "bfloat16",
        "exp_avg_sq_dtype": second,
    }
    path = tmp_path / "precision.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "context_budget": {"request_window_tokens": 2048, "max_new_tokens_per_turn": 1024, "max_turns": 1},
                "trainer": {
                    "strategy": "megatron",
                    "policy_train_spans": True,
                    "optimizer_state_metrics": True,
                    "policy": {"megatron_config": {"optimizer_config_kwargs": declared}},
                },
            }
        )
    )
    parsed = parse_rl_config(str(path))
    arguments = build_skyrl_hydra_args(
        parsed, {"job_name": "precision-test", "num_nodes": 2}, SimpleNamespace(gpus_per_node=8)
    )
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="ppo_base_config.yaml", overrides=arguments)
    effective = cfg.trainer.policy.megatron_config.optimizer_config_kwargs
    assert all(effective[key] == value for key, value in declared.items())
    assert effective.optimizer_cpu_offload is False  # Undeclared native defaults survive the merge.
    kwargs = megatron_optimizer_kwargs(cfg.trainer.policy.optimizer_config, effective)
    assert kwargs["exp_avg_dtype"] is torch.bfloat16
    assert kwargs["exp_avg_sq_dtype"] is (torch.float32 if second == "float32" else torch.bfloat16)
    assert cfg.trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32 is True
    assert cfg.trainer.optimizer_state_metrics is True


def test_launcher_still_rejects_unknown_fields_outside_native_kwargs_namespace(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "context_budget": {"request_window_tokens": 2048, "max_new_tokens_per_turn": 1024, "max_turns": 1},
                "trainer": {"optimizer_state_metricss": True},
            }
        )
    )
    parsed = parse_rl_config(str(path))
    arguments = build_skyrl_hydra_args(
        parsed, {"job_name": "typo-test", "num_nodes": 2}, SimpleNamespace(gpus_per_node=8)
    )
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        with pytest.raises(ConfigCompositionException, match="optimizer_state_metricss"):
            compose(config_name="ppo_base_config.yaml", overrides=arguments)
