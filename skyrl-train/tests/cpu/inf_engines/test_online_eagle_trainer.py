"""Behavior tests for bounded online EAGLE training inputs."""

import copy
import json
from pathlib import Path
import random

import pytest
from safetensors.torch import save_file
import torch

from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    OnlineEagleTrainerRuntime,
    _candidate_state,
    _convert_trainable_parameters,
    _load_batch,
    _load_window_group,
    _offload_optimizer_state,
    _pack_windows,
    _restore_rng_states,
    _refresh_target_owned_weights,
    _restore_trainable_master_state,
    capture_config_for_worker,
    candidate_is_acceptable,
    merge_online_eagle_captures,
    partition_capture_windows,
    per_worker_capture_token_credit,
    request_group_from_id,
)


class _TargetOwnedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t2d = torch.tensor([True, False, True])
        self.embed_tokens = torch.nn.Embedding(3, 2)
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.verifier_lm_head = torch.nn.Linear(2, 2, bias=False)
        self.verifier_norm = torch.nn.LayerNorm(2)
        self.verifier_gate_down = torch.nn.Linear(2, 2)
        self.verifier_gate_up = torch.nn.Linear(2, 2)


class _DeviceSensitiveVocabularyMap:
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def to(self, *, device=None, dtype=None) -> torch.Tensor:
        assert device == self.mask.device
        return self.mask.to(dtype=dtype)

    def numel(self) -> int:
        return self.mask.numel()


def test_refresh_target_owned_weights_moves_vocabulary_mask_to_head_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _TargetOwnedModel()
    model.t2d = _DeviceSensitiveVocabularyMap(model.t2d)
    monkeypatch.setattr(torch, "any", lambda _: True)
    target = {
        "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "lm_head.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }
    save_file(target, str(tmp_path / "target.safetensors"))

    _refresh_target_owned_weights(model, tmp_path)

    torch.testing.assert_close(model.embed_tokens.weight, target["model.embed_tokens.weight"])
    torch.testing.assert_close(model.lm_head.weight, target["lm_head.weight"][[0, 2]])
    torch.testing.assert_close(model.verifier_lm_head.weight, target["lm_head.weight"][[0, 2]])
    assert isinstance(model.verifier_norm, torch.nn.Identity)


def test_capture_config_activates_every_data_parallel_worker() -> None:
    resolved = capture_config_for_worker(
        {
            "step": 1,
            "max_tokens": 131_072,
            "max_window_tokens": 16_384,
            "max_sequences_per_prompt_group": 2,
            "target_revision": "target",
            "draft_revision": "draft",
            "reserved_gpu_memory_gib": 8,
        },
        worker_count=8,
        worker_index=3,
    )

    assert resolved["trainer_rank"] == 3
    assert resolved["capture_target_snapshot"] is False
    assert resolved["max_tokens"] == 32_768


def test_per_worker_capture_credit_has_bounded_fragmentation_slack() -> None:
    credits = [
        per_worker_capture_token_credit(
            global_max_tokens=131_072,
            max_window_tokens=16_384,
            worker_count=8,
            worker_index=index,
        )
        for index in range(8)
    ]

    assert credits == [32_768] * 8
    assert sum(credits) == 131_072 + 8 * 16_384


def test_per_worker_capture_credit_distributes_global_remainder() -> None:
    credits = [
        per_worker_capture_token_credit(
            global_max_tokens=10,
            max_window_tokens=4,
            worker_count=3,
            worker_index=index,
        )
        for index in range(3)
    ]

    assert credits == [8, 7, 7]


def _write_rank_capture(
    root: Path,
    worker_rank: int,
    windows: list[tuple[str, str]],
    *,
    step: int = 7,
    draft_revision: str = "draft-step-6",
    tokens_by_request: dict[str, int] | None = None,
) -> None:
    directory = root / f"rank-{worker_rank:05d}"
    directory.mkdir(parents=True)
    target_path = directory / "target.safetensors"
    config_path = directory / "target-config.json"
    if worker_rank == 0:
        save_file(
            {
                "model.embed_tokens.weight": torch.ones(2, 2),
                "lm_head.weight": torch.ones(2, 2),
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
            }
        )
    target = None
    if worker_rank == 0:
        target = {
            "weights_path": target_path.name,
            "config_path": config_path.name,
            "inventory": {"test": {"shape": [2, 2], "dtype": "torch.float32"}},
        }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "vllm-online-eagle-capture",
                "format_version": 1,
                "active": True,
                "step": step,
                "worker_rank": worker_rank,
                "target_revision": f"policy-step-{step - 1}",
                "draft_revision": draft_revision,
                "aux_layer_ids": [1, 2, 3],
                "head_input_semantics": "post_final_norm_target_lm_head_input",
                "windows": manifest_windows,
                "captured_rows": sum(window["tokens"] for window in manifest_windows),
                "dropped_requests": 0,
                "dropped_windows": 0,
                "target": target,
            }
        )
    )


def test_capture_merge_is_globally_bounded_and_rank_assignment_independent(tmp_path: Path) -> None:
    assignments = (
        ([("request-a", "group-a"), ("request-c", "group-c")], [("request-b", "group-b")]),
        ([("request-b", "group-b")], [("request-c", "group-c"), ("request-a", "group-a")]),
    )
    selected_orders = []
    for index, (rank_zero, rank_one) in enumerate(assignments):
        root = tmp_path / f"capture-{index}"
        _write_rank_capture(root, 0, rank_zero)
        _write_rank_capture(root, 1, rank_one)
        manifest = merge_online_eagle_captures(
            root,
            tmp_path / f"merged-{index}",
            expected_step=7,
            max_tokens=6,
            max_sequences_per_prompt_group=1,
            max_window_tokens=100,
        )
        selected_orders.append([window["request_id"] for window in manifest["windows"]])
        assert manifest["captured_rows"] == 6
        assert manifest["unselected_windows"] == 1

    assert selected_orders[0] == selected_orders[1]


def test_capture_merge_rejects_different_lineage(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    _write_rank_capture(root, 0, [("request-a", "group-a")])
    _write_rank_capture(root, 1, [("request-b", "group-b")], draft_revision="different")

    with pytest.raises(ValueError, match="one target/draft identity"):
        merge_online_eagle_captures(
            root,
            tmp_path / "merged",
            expected_step=7,
            max_tokens=10,
            max_sequences_per_prompt_group=1,
            max_window_tokens=100,
        )


def test_capture_merge_excludes_windows_above_forward_bound(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    _write_rank_capture(
        root,
        0,
        [("at-limit", "group-a"), ("over-limit", "group-b")],
        tokens_by_request={"at-limit": 5, "over-limit": 6},
    )

    manifest = merge_online_eagle_captures(
        root,
        tmp_path / "merged",
        expected_step=7,
        max_tokens=20,
        max_sequences_per_prompt_group=1,
        max_window_tokens=4,
    )

    assert [window["request_id"] for window in manifest["windows"]] == ["at-limit"]
    assert manifest["oversized_windows"] == 1


def test_capture_merge_rejects_escaped_window_path(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    _write_rank_capture(root, 0, [("request-a", "group-a")])
    manifest_path = root / "rank-00000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["windows"][0]["path"] = "../outside.safetensors"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing or escaped"):
        merge_online_eagle_captures(
            root,
            tmp_path / "merged",
            expected_step=7,
            max_tokens=10,
            max_sequences_per_prompt_group=1,
            max_window_tokens=100,
        )


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
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, *, device: restored.update(cuda=(state, device)))
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


def test_capture_partition_is_deterministic_and_group_disjoint() -> None:
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


def test_capture_partition_derives_prompt_groups_from_skyrl_request_ids() -> None:
    windows = [{"request_id": f"skyrl-group-{index // 2:08x}-attempt{index}"} for index in range(20)]

    train, holdout = partition_capture_windows(
        windows,
        step=7,
        holdout_fraction=0.2,
        min_train_sequences=10,
        min_holdout_sequences=4,
    )

    train_groups = {request_group_from_id(item["request_id"]) for item in train}
    holdout_groups = {request_group_from_id(item["request_id"]) for item in holdout}
    assert train_groups.isdisjoint(holdout_groups)


def test_capture_partition_rejects_a_vacuous_holdout() -> None:
    with pytest.raises(ValueError, match="at least 12 captured windows"):
        partition_capture_windows(
            [{"request_id": f"request-{index}"} for index in range(11)],
            step=1,
            holdout_fraction=0.1,
            min_train_sequences=8,
            min_holdout_sequences=4,
        )


def test_pack_windows_keeps_one_admitted_history_larger_than_target() -> None:
    windows = [{"request_id": "request-long", "tokens": 10}]
    assert _pack_windows(windows, max_tokens=8, max_window_tokens=9) == [windows]


def test_pack_windows_rejects_a_forward_above_the_window_bound() -> None:
    with pytest.raises(ValueError, match="request-long.*tokens=9 limit=8"):
        _pack_windows([{"request_id": "request-long", "tokens": 10}], max_tokens=4, max_window_tokens=8)


def test_candidate_gate_enforces_loss_and_agreement_tolerances() -> None:
    common = {
        "incumbent_loss": 1.0,
        "incumbent_agreement": 0.8,
        "max_loss_increase": 0.02,
        "max_agreement_decrease": 0.01,
    }
    assert candidate_is_acceptable(candidate_loss=1.02, candidate_agreement=0.79, **common)
    assert not candidate_is_acceptable(candidate_loss=1.021, candidate_agreement=0.8, **common)
    assert not candidate_is_acceptable(candidate_loss=1.0, candidate_agreement=0.789, **common)


def test_candidate_state_contains_only_trainable_serving_dtype() -> None:
    class Draft(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.owned = torch.nn.Linear(2, 2, bias=False)
            self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
            self.register_buffer("vocabulary_map", torch.arange(2))

    model = Draft().to(dtype=torch.float32)
    with pytest.raises(ValueError, match=r"dtype torch.float32, expected torch.bfloat16"):
        _candidate_state(model, serving_dtype=torch.bfloat16)

    _convert_trainable_parameters(model, torch.bfloat16)
    state = _candidate_state(model, serving_dtype=torch.bfloat16)
    assert set(state) == {"owned.weight"}


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
    continuous_model(inputs).square().sum().backward()
    continuous_optimizer.step()
    _offload_optimizer_state(continuous_optimizer)

    resumed_model = torch.nn.Linear(3, 2)
    resumed_model.load_state_dict(continuous_model.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(copy.deepcopy(continuous_optimizer.state_dict()))

    for model, optimizer in ((continuous_model, continuous_optimizer), (resumed_model, resumed_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().sum().backward()
        optimizer.step()

    for continuous, resumed in zip(continuous_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(continuous, resumed)


def test_persistent_runtime_snapshot_restores_model_and_optimizer() -> None:
    runtime = OnlineEagleTrainerRuntime.__new__(OnlineEagleTrainerRuntime)
    runtime.device = torch.device("cpu")
    runtime.serving_dtype = torch.float32
    runtime.model = torch.nn.Linear(3, 2)
    runtime.trainable = list(runtime.model.parameters())
    runtime.optimizer = torch.optim.AdamW(runtime.trainable, lr=1e-3)
    runtime.model(torch.ones(2, 3)).square().sum().backward()
    runtime.optimizer.step()
    _offload_optimizer_state(runtime.optimizer)
    incumbent = runtime._snapshot()
    incumbent_parameters = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    incumbent_optimizer = copy.deepcopy(runtime.optimizer.state_dict())

    with torch.no_grad():
        for parameter in runtime.model.parameters():
            parameter.add_(10)
    runtime._restore_snapshot(incumbent)

    assert all(torch.equal(runtime.model.state_dict()[name], value) for name, value in incumbent_parameters.items())
    assert runtime.optimizer.state_dict()["param_groups"] == incumbent_optimizer["param_groups"]


def test_fp32_master_must_round_to_the_served_checkpoint() -> None:
    model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)
    served = model.weight.detach().clone()
    master = served.float()
    _convert_trainable_parameters(model, torch.float32)

    _restore_trainable_master_state(model, {"weight": master}, serving_dtype=torch.bfloat16)

    assert torch.equal(model.weight, master)
    assert torch.equal(model.weight.to(torch.bfloat16), served)


def _write_training_window(path: Path, *, nonfinite: bool = False) -> None:
    hidden = [[1.0], [float("nan") if nonfinite else 2.0], [3.0]]
    save_file(
        {
            "input_ids": torch.tensor([10, 20, 30]),
            "hidden_states": torch.tensor(hidden),
            "head_input_hidden_states": torch.tensor([[11.0], [12.0], [13.0]]),
            "loss_mask": torch.tensor([False, True, True]),
            "position_ids": torch.tensor([4, 5, 6]),
        },
        str(path),
    )


def test_online_batch_uses_previous_aux_state_and_next_target_head_input(tmp_path: Path) -> None:
    path = tmp_path / "window.safetensors"
    _write_training_window(path)

    batch = _load_batch(path, torch.device("cpu"))

    assert batch["input_ids"].tolist() == [[20, 30]]
    assert batch["hidden_states"].tolist() == [[[1.0], [2.0]]]
    assert batch["verifier_last_hidden_states"].tolist() == [[[12.0], [13.0]]]
    assert batch["loss_mask"].tolist() == [[True, True]]


def test_online_window_group_keeps_histories_independent(tmp_path: Path) -> None:
    paths = [tmp_path / "first.safetensors", tmp_path / "second.safetensors"]
    for path in paths:
        _write_training_window(path)
    windows = [{"path": path.name} for path in paths]

    loaded = list(_load_window_group(windows, tmp_path, torch.device("cpu")))

    assert [window for window, _batch in loaded] == windows
    assert loaded[0][1]["document_ids"].tolist() == [[0, 0]]
    assert loaded[1][1]["document_ids"].tolist() == [[0, 0]]


def test_online_batch_rejects_nonfinite_capture_with_path(tmp_path: Path) -> None:
    path = tmp_path / "corrupt-window.safetensors"
    _write_training_window(path, nonfinite=True)

    with pytest.raises(FloatingPointError, match=r"corrupt-window.*hidden_states"):
        _load_batch(path, torch.device("cpu"))
