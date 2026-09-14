"""Behavior tests for the bounded online EAGLE trainer inputs."""

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

import skyrl_train.inference_engines.vllm.online_eagle_trainer as online_eagle_trainer
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    ONLINE_EAGLE_CAPTURE_TRANSFER_FORMAT,
    _candidate_state,
    _configure_exact_mask_attention,
    _convert_trainable_parameters,
    _restore_trainable_master_state,
    _load_batch,
    _load_window_group,
    _offload_optimizer_state,
    _pack_windows,
    _restore_rng_states,
    _sdpa_kernel_context,
    candidate_is_acceptable,
    catalog_online_eagle_capture,
    export_served_speculator_checkpoint,
    OnlineEagleTrainerRuntime,
    partition_capture_windows,
    plan_online_eagle_capture_transfer,
    preserve_online_eagle_failure,
    publish_speculator_checkpoint,
    remove_online_eagle_scratch,
    restore_speculator_checkpoint,
)

from marinskyrl.hf_model import sha256_file


def _write_rank_capture(
    root: Path,
    worker_rank: int,
    windows: list[tuple[str, str]],
    *,
    target_value: float = 1.0,
    step: int = 7,
    draft_revision: str = "draft-step-6",
    worker_rank_override: int | None = None,
    tokens_by_request: dict[str, int] | None = None,
) -> None:
    directory = root / f"rank-{worker_rank:05d}"
    directory.mkdir(parents=True)
    target_path = directory / "target.safetensors"
    config_path = directory / "target-config.json"
    save_file(
        {
            "model.embed_tokens.weight": torch.full((2, 2), target_value),
            "lm_head.weight": torch.full((2, 2), target_value),
        },
        str(target_path),
    )
    config_path.write_text('{"model_type":"test"}')
    manifest_windows = []
    for index, (request_id, group_id) in enumerate(windows):
        window_path = directory / f"window-{index:06d}.safetensors"
        tokens = (tokens_by_request or {}).get(request_id, 3)
        save_file({"input_ids": torch.arange(tokens)}, str(window_path))
        manifest_windows.append(
            {
                "path": window_path.name,
                "request_id": request_id,
                "group_id": group_id,
                "tokens": tokens,
                "supervised_tokens": max(1, tokens - 1),
                "sha256": sha256_file(window_path),
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "vllm-online-eagle-capture",
                "format_version": 1,
                "active": True,
                "step": step,
                "worker_rank": worker_rank if worker_rank_override is None else worker_rank_override,
                "target_revision": f"policy-step-{step - 1}",
                "draft_revision": draft_revision,
                "aux_layer_ids": [1, 2, 3],
                "head_input_semantics": "post_final_norm_target_lm_head_input",
                "windows": manifest_windows,
                "captured_rows": sum(window["tokens"] for window in manifest_windows),
                "dropped_requests": 0,
                "dropped_windows": 0,
                "target": {
                    "weights_path": target_path.name,
                    "weights_sha256": sha256_file(target_path),
                    "config_path": config_path.name,
                    "config_sha256": sha256_file(config_path),
                    "inventory": {"test": {"shape": [2, 2], "dtype": "torch.float32"}},
                },
            }
        )
    )


def test_capture_transfer_plan_uses_metadata_only_and_all_source_ranks(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    _write_rank_capture(root, 0, [("request-a", "group-a")])
    _write_rank_capture(root, 1, [("request-b", "group-b")])
    catalogs = [
        catalog_online_eagle_capture(root, worker_rank=0, transfer_rank=1),
        catalog_online_eagle_capture(root, worker_rank=1, transfer_rank=2),
    ]

    plan = plan_online_eagle_capture_transfer(
        catalogs,
        expected_workers=2,
        expected_step=7,
        max_tokens=16,
        max_sequences_per_prompt_group=2,
        max_window_tokens=16,
    )

    assert plan["format"] == ONLINE_EAGLE_CAPTURE_TRANSFER_FORMAT
    assert plan["capture_manifest"]["captured_rows"] == 6
    assert len(plan["capture_manifest"]["windows"]) == 2
    assert {operation["source_rank"] for operation in plan["operations"]} == {1, 2}
    assert plan["total_bytes"] == sum(operation["tensor"]["bytes"] for operation in plan["operations"])


def test_capture_transfer_plan_is_globally_bounded_and_rank_assignment_independent(tmp_path: Path) -> None:
    assignments = (
        ([("request-a", "group-a"), ("request-c", "group-c")], [("request-b", "group-b")]),
        ([("request-b", "group-b")], [("request-c", "group-c"), ("request-a", "group-a")]),
    )
    selected_orders = []
    for index, (rank_zero, rank_one) in enumerate(assignments):
        root = tmp_path / str(index)
        _write_rank_capture(root, 0, rank_zero)
        _write_rank_capture(root, 1, rank_one)
        plan = plan_online_eagle_capture_transfer(
            [
                catalog_online_eagle_capture(root, worker_rank=0, transfer_rank=1),
                catalog_online_eagle_capture(root, worker_rank=1, transfer_rank=2),
            ],
            expected_workers=2,
            expected_step=7,
            max_tokens=6,
            max_sequences_per_prompt_group=1,
            max_window_tokens=100,
        )
        selected_orders.append([window["request_id"] for window in plan["capture_manifest"]["windows"]])
        assert plan["capture_manifest"]["captured_rows"] == 6
        assert plan["capture_manifest"]["unselected_windows"] == 1

    assert selected_orders[0] == selected_orders[1]


def test_capture_transfer_plan_rejects_different_target_snapshots(tmp_path: Path) -> None:
    root = tmp_path / "target-mismatch"
    _write_rank_capture(root, 0, [("request-a", "group-a")])
    _write_rank_capture(root, 1, [("request-b", "group-b")], target_value=2.0)

    with pytest.raises(ValueError, match="one target snapshot"):
        plan_online_eagle_capture_transfer(
            [
                catalog_online_eagle_capture(root, worker_rank=0, transfer_rank=1),
                catalog_online_eagle_capture(root, worker_rank=1, transfer_rank=2),
            ],
            expected_workers=2,
            expected_step=7,
            max_tokens=10,
            max_sequences_per_prompt_group=1,
            max_window_tokens=100,
        )


def test_capture_transfer_plan_excludes_windows_above_the_forward_bound(tmp_path: Path) -> None:
    root = tmp_path / "forward-bound"
    _write_rank_capture(
        root,
        0,
        [("request-at-limit", "group-at-limit"), ("request-over-limit", "group-over-limit")],
        tokens_by_request={"request-at-limit": 5, "request-over-limit": 6},
    )

    plan = plan_online_eagle_capture_transfer(
        [catalog_online_eagle_capture(root, worker_rank=0, transfer_rank=1)],
        expected_workers=1,
        expected_step=7,
        max_tokens=20,
        max_sequences_per_prompt_group=1,
        max_window_tokens=4,
    )

    assert [window["request_id"] for window in plan["capture_manifest"]["windows"]] == ["request-at-limit"]
    assert plan["capture_manifest"]["oversized_windows"] == 1


def test_rng_restore_moves_device_mapped_generator_states_back_to_cpu(monkeypatch) -> None:
    class DeviceMappedState:
        def __init__(self, cpu_state):
            self.cpu_state = cpu_state

        def cpu(self):
            return self.cpu_state

    torch_rng_state = torch.tensor([1], dtype=torch.uint8)
    cuda_rng_state = torch.tensor([2], dtype=torch.uint8)
    python_rng_state = random.getstate()
    restored = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda state: restored.update(torch=state))
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, *, device: restored.update(cuda=(state, device)),
    )
    monkeypatch.setattr(random, "setstate", lambda state: restored.update(python=state))

    _restore_rng_states(
        {
            "torch_rng_state": DeviceMappedState(torch_rng_state),
            "cuda_rng_state": DeviceMappedState(cuda_rng_state),
            "python_rng_state": python_rng_state,
        },
        torch.device("cuda:0"),
    )

    assert restored == {
        "torch": torch_rng_state,
        "cuda": (cuda_rng_state, torch.device("cuda:0")),
        "python": python_rng_state,
    }


def test_capture_partition_is_deterministic_disjoint_and_satisfies_minima() -> None:
    windows = [{"request_id": f"request-{index}", "group_id": f"group-{index // 2}"} for index in range(20)]

    first = partition_capture_windows(
        windows,
        step=7,
        holdout_fraction=0.2,
        min_train_sequences=10,
        min_holdout_sequences=4,
    )
    second = partition_capture_windows(
        list(reversed(windows)),
        step=7,
        holdout_fraction=0.2,
        min_train_sequences=10,
        min_holdout_sequences=4,
    )

    assert first == second
    train, holdout = first
    assert len(train) == 16
    assert len(holdout) == 4
    assert {item["group_id"] for item in train}.isdisjoint(item["group_id"] for item in holdout)


def test_pack_windows_keeps_one_admitted_history_larger_than_the_packing_target() -> None:
    windows = [{"request_id": "request-long", "tokens": 10}]

    assert _pack_windows(windows, max_tokens=8, max_window_tokens=9) == [windows]


def test_pack_windows_rejects_one_forward_larger_than_the_window_bound() -> None:
    with pytest.raises(ValueError, match="request-long.*tokens=9 limit=8"):
        _pack_windows(
            [{"request_id": "request-long", "tokens": 10}],
            max_tokens=4,
            max_window_tokens=8,
        )


def test_exact_mask_attention_uses_sdpa_and_pins_the_efficient_cuda_kernel(monkeypatch) -> None:
    config = SimpleNamespace(transformer_layer_config=SimpleNamespace(_attn_implementation="eager"))
    selected = []
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda backend: selected.append(backend) or nullcontext())

    _configure_exact_mask_attention(config)
    with _sdpa_kernel_context(torch.device("cuda")):
        pass

    assert config.transformer_layer_config._attn_implementation == "sdpa"
    assert selected == [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]


def test_failed_update_preserves_hardlinked_capture_and_incumbent(tmp_path: Path) -> None:
    process_root = tmp_path / "process-id"
    capture_dir = process_root / "step-7" / "merged"
    incumbent_dir = process_root / "candidates" / "step-6"
    output_dir = process_root / "candidates" / "step-7"
    capture_dir.mkdir(parents=True)
    incumbent_dir.mkdir(parents=True)
    (capture_dir / "window.safetensors").write_bytes(b"capture")
    (incumbent_dir / "model.safetensors").write_bytes(b"incumbent")
    job = SimpleNamespace(
        step=7,
        capture_dir=str(capture_dir),
        draft_model_dir=str(incumbent_dir),
        output_dir=str(output_dir),
    )

    failure_dir = Path(preserve_online_eagle_failure(job, FloatingPointError("bad loss")))

    manifest = json.loads((failure_dir / "manifest.json").read_text())
    assert manifest["format"] == "marinskyrl-online-eagle-failure"
    assert manifest["error"] == "FloatingPointError: bad loss"
    assert (failure_dir / "capture" / "window.safetensors").stat().st_ino == (
        capture_dir / "window.safetensors"
    ).stat().st_ino
    assert (failure_dir / "incumbent" / "model.safetensors").stat().st_ino == (
        incumbent_dir / "model.safetensors"
    ).stat().st_ino

    job.step = 8
    job.output_dir = str(process_root / "candidates" / "step-8")
    next_failure_dir = Path(preserve_online_eagle_failure(job, RuntimeError("next failure")))

    assert not failure_dir.exists()
    assert next_failure_dir.name == "step-8"


def test_scratch_cleanup_refuses_root_or_unmanaged_path(tmp_path: Path, monkeypatch) -> None:
    scratch_root = tmp_path / "marinskyrl-online-eagle"
    scratch_root.mkdir()
    monkeypatch.setattr(online_eagle_trainer, "ONLINE_EAGLE_SCRATCH_ROOT", scratch_root)

    with pytest.raises(ValueError, match="outside a process tree"):
        remove_online_eagle_scratch(scratch_root)
    with pytest.raises(ValueError, match="outside a process tree"):
        remove_online_eagle_scratch(tmp_path / "unmanaged")


def test_capture_partition_rejects_a_vacuous_holdout() -> None:
    with pytest.raises(ValueError, match="at least 12 captured windows"):
        partition_capture_windows(
            [{"request_id": f"request-{index}"} for index in range(11)],
            step=1,
            holdout_fraction=0.1,
            min_train_sequences=8,
            min_holdout_sequences=4,
        )


def test_candidate_gate_enforces_declared_loss_and_agreement_tolerances() -> None:
    common = {
        "incumbent_loss": 1.0,
        "incumbent_agreement": 0.8,
        "max_loss_increase": 0.02,
        "max_agreement_decrease": 0.01,
    }

    assert candidate_is_acceptable(candidate_loss=1.02, candidate_agreement=0.79, **common)
    assert not candidate_is_acceptable(candidate_loss=1.021, candidate_agreement=0.8, **common)
    assert not candidate_is_acceptable(candidate_loss=1.0, candidate_agreement=0.789, **common)
    assert not candidate_is_acceptable(candidate_loss=float("nan"), candidate_agreement=0.8, **common)


def test_candidate_state_must_equal_declared_serving_dtype() -> None:
    model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)

    with pytest.raises(ValueError, match=r"dtype torch.float32, expected torch.bfloat16"):
        _candidate_state(model, serving_dtype=torch.bfloat16)

    _convert_trainable_parameters(model, torch.bfloat16)
    state = _candidate_state(model, serving_dtype=torch.bfloat16)

    assert state["weight"].dtype == torch.bfloat16


def test_optimizer_state_is_offloaded_before_candidate_evaluation() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    _offload_optimizer_state(optimizer)

    assert model.weight.grad is None
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_offloaded_adam_state_round_trip_matches_continuous_training() -> None:
    torch.manual_seed(7)
    continuous_model = torch.nn.Linear(3, 2)
    continuous_optimizer = torch.optim.AdamW(continuous_model.parameters(), lr=1e-3)
    inputs = torch.randn(4, 3)

    continuous_optimizer.zero_grad(set_to_none=True)
    continuous_model(inputs).square().sum().backward()
    continuous_optimizer.step()
    _offload_optimizer_state(continuous_optimizer)

    resumed_model = torch.nn.Linear(3, 2)
    resumed_model.load_state_dict(continuous_model.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(copy.deepcopy(continuous_optimizer.state_dict()))

    for model, optimizer in (
        (continuous_model, continuous_optimizer),
        (resumed_model, resumed_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().sum().backward()
        optimizer.step()

    for continuous, resumed in zip(continuous_model.parameters(), resumed_model.parameters()):
        assert torch.equal(continuous, resumed)


def test_persistent_runtime_rollback_restores_master_and_optimizer_state() -> None:
    runtime = OnlineEagleTrainerRuntime.__new__(OnlineEagleTrainerRuntime)
    runtime.device = torch.device("cpu")
    runtime.serving_dtype = torch.float32
    runtime.model = torch.nn.Linear(3, 2)
    runtime.trainable = list(runtime.model.parameters())
    runtime.optimizer = torch.optim.AdamW(runtime.trainable, lr=1e-3)
    runtime._pending_revision = None
    runtime._pending_incumbent = None

    runtime.optimizer.zero_grad(set_to_none=True)
    runtime.model(torch.ones(2, 3)).square().sum().backward()
    runtime.optimizer.step()
    _offload_optimizer_state(runtime.optimizer)
    incumbent = runtime._snapshot()
    incumbent_parameters = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    incumbent_optimizer = copy.deepcopy(runtime.optimizer.state_dict())

    with torch.no_grad():
        for parameter in runtime.model.parameters():
            parameter.add_(10)
    runtime._pending_revision = "draft-step-4"
    runtime._pending_incumbent = incumbent
    runtime.rollback("draft-step-4")

    assert all(torch.equal(runtime.model.state_dict()[name], value) for name, value in incumbent_parameters.items())
    assert runtime.optimizer.state_dict()["param_groups"] == incumbent_optimizer["param_groups"]
    for restored, expected in zip(
        runtime.optimizer.state_dict()["state"].values(),
        incumbent_optimizer["state"].values(),
        strict=True,
    ):
        assert restored.keys() == expected.keys()
        assert all(torch.equal(restored[name], expected[name]) for name in restored)
    assert runtime._pending_revision is None
    assert runtime._pending_incumbent is None


def test_fp32_master_must_round_to_the_served_checkpoint() -> None:
    model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)
    served = model.weight.detach().clone()
    master = served.float()
    _convert_trainable_parameters(model, torch.float32)

    _restore_trainable_master_state(model, {"weight": master}, serving_dtype=torch.bfloat16)

    assert torch.equal(model.weight, master)
    assert torch.equal(model.weight.to(torch.bfloat16), served)


def test_fp32_master_rejects_different_served_bytes() -> None:
    model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)
    master = model.weight.detach().float() + 1
    _convert_trainable_parameters(model, torch.float32)

    with pytest.raises(ValueError, match="does not round to the served tensor"):
        _restore_trainable_master_state(model, {"weight": master}, serving_dtype=torch.bfloat16)


def test_online_batch_uses_previous_aux_state_and_exact_next_target_head_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "window.safetensors"
    save_file(
        {
            "input_ids": torch.tensor([10, 20, 30]),
            "hidden_states": torch.tensor([[1.0], [2.0], [3.0]]),
            "head_input_hidden_states": torch.tensor([[11.0], [12.0], [13.0]]),
            "loss_mask": torch.tensor([False, True, True]),
            "position_ids": torch.tensor([4, 5, 6]),
        },
        str(path),
    )

    batch = _load_batch(path, torch.device("cpu"))

    assert batch["input_ids"].tolist() == [[20, 30]]
    assert batch["hidden_states"].tolist() == [[[1.0], [2.0]]]
    assert batch["verifier_last_hidden_states"].tolist() == [[[12.0], [13.0]]]
    assert batch["loss_mask"].tolist() == [[True, True]]
    assert batch["position_ids"].tolist() == [[5, 6]]


def test_online_window_group_keeps_teacher_forcing_histories_independent(tmp_path: Path) -> None:
    paths = [tmp_path / "first.safetensors", tmp_path / "second.safetensors"]
    save_file(
        {
            "input_ids": torch.tensor([10, 20, 30]),
            "hidden_states": torch.tensor([[1.0], [2.0], [3.0]]),
            "head_input_hidden_states": torch.tensor([[11.0], [12.0], [13.0]]),
            "loss_mask": torch.tensor([False, True, True]),
            "position_ids": torch.tensor([4, 5, 6]),
        },
        str(paths[0]),
    )
    save_file(
        {
            "input_ids": torch.tensor([40, 50, 60, 70]),
            "hidden_states": torch.tensor([[4.0], [5.0], [6.0], [7.0]]),
            "head_input_hidden_states": torch.tensor([[14.0], [15.0], [16.0], [17.0]]),
            "loss_mask": torch.tensor([False, True, False, True]),
            "position_ids": torch.tensor([10, 11, 12, 13]),
        },
        str(paths[1]),
    )

    windows = [{"path": path.name} for path in paths]
    loaded = list(_load_window_group(windows, tmp_path, torch.device("cpu")))

    assert [window for window, _batch in loaded] == windows
    assert loaded[0][1]["input_ids"].tolist() == [[20, 30]]
    assert loaded[1][1]["input_ids"].tolist() == [[50, 60, 70]]
    assert loaded[0][1]["document_ids"].tolist() == [[0, 0]]
    assert loaded[1][1]["document_ids"].tolist() == [[0, 0, 0]]


def test_online_batch_rejects_nonfinite_capture_with_tensor_and_path(tmp_path: Path) -> None:
    path = tmp_path / "corrupt-window.safetensors"
    save_file(
        {
            "input_ids": torch.tensor([10, 20, 30]),
            "hidden_states": torch.tensor([[1.0], [float("nan")], [3.0]]),
            "head_input_hidden_states": torch.tensor([[11.0], [12.0], [13.0]]),
            "loss_mask": torch.tensor([False, True, True]),
            "position_ids": torch.tensor([4, 5, 6]),
        },
        str(path),
    )

    with pytest.raises(FloatingPointError, match=r"corrupt-window.*hidden_states"):
        _load_batch(path, torch.device("cpu"))


def test_served_speculator_checkpoint_is_exact_idempotent_and_restorable(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "config.json").write_text('{"speculators_model_type":"eagle3"}')
    save_file({"owned.weight": torch.ones(2, 2)}, str(source / "model.safetensors"))
    destination = tmp_path / "global_step_7" / "speculator"

    first = publish_speculator_checkpoint(
        str(source),
        str(destination),
        draft_revision="draft-step-7",
        served_target_revision="policy-step-7",
    )
    second = publish_speculator_checkpoint(
        str(source),
        str(destination),
        draft_revision="draft-step-7",
        served_target_revision="policy-step-7",
    )
    restored = restore_speculator_checkpoint(str(destination), str(tmp_path / "restored"))

    assert first == second
    assert restored["draft_revision"] == "draft-step-7"
    assert restored["served_target_revision"] == "policy-step-7"
    assert restored["lineage"] == {"initial_source_identity": "draft-step-7"}
    assert (tmp_path / "restored" / "model.safetensors").read_bytes() == (source / "model.safetensors").read_bytes()
    normalized = json.loads((tmp_path / "restored" / "manifest.json").read_text())
    assert normalized["format"] == "marinskyrl-online-eagle-candidate"
    assert normalized["complete"] is True


def test_restore_rejects_legacy_online_trainer_state(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "config.json").write_text('{"speculators_model_type":"eagle3"}')
    save_file({"owned.weight": torch.ones(2, 2)}, str(source / "model.safetensors"))
    torch.save({"optimizer": {}}, source / "trainer_state.pt")
    checkpoint = tmp_path / "global_step_7" / "speculator"
    publish_speculator_checkpoint(
        str(source),
        str(checkpoint),
        draft_revision="draft-step-7",
        served_target_revision="policy-step-7",
    )

    with pytest.raises(ValueError, match="Incompatible online EAGLE trainer state"):
        restore_speculator_checkpoint(str(checkpoint), str(tmp_path / "restored"))


def test_served_speculator_export_preserves_the_exact_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "config.json").write_text('{"speculators_model_type":"eagle3"}')
    save_file({"owned.weight": torch.ones(2, 2)}, str(source / "model.safetensors"))
    checkpoint = tmp_path / "global_step_7" / "speculator"
    publish_speculator_checkpoint(
        str(source),
        str(checkpoint),
        draft_revision="draft-step-7",
        served_target_revision="policy-step-7",
    )
    destination = tmp_path / "exports" / "global_step_7" / "speculator"

    first = export_served_speculator_checkpoint(str(checkpoint), str(destination))
    second = export_served_speculator_checkpoint(str(checkpoint), str(destination))

    assert first == second
    assert first["draft_revision"] == "draft-step-7"
    assert (destination / "weights" / "model.safetensors").read_bytes() == (
        checkpoint / "weights" / "model.safetensors"
    ).read_bytes()
