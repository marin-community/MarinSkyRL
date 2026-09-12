"""Behavior tests for the bounded online EAGLE trainer inputs."""

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    _load_batch,
    _load_packed_batch,
    candidate_is_acceptable,
    child_cuda_visible_device,
    export_served_speculator_checkpoint,
    partition_capture_windows,
    publish_speculator_checkpoint,
    restore_speculator_checkpoint,
)


@pytest.mark.parametrize(
    ("visible_devices", "device_index", "child_device"),
    [
        (None, None, None),
        (None, 2, "2"),
        ("4,7", 1, "7"),
        ("GPU-first,GPU-second", 0, "GPU-first"),
    ],
)
def test_child_cuda_visible_device_follows_parent_mapping(visible_devices, device_index, child_device) -> None:
    assert child_cuda_visible_device(visible_devices, device_index) == child_device


@pytest.mark.parametrize(("visible_devices", "device_index"), [("4,7", -1), ("4,7", 2), ("4,,7", 1)])
def test_child_cuda_visible_device_rejects_invalid_mapping(visible_devices, device_index) -> None:
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        child_cuda_visible_device(visible_devices, device_index)


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


def test_online_batch_packs_captures_with_distinct_attention_documents(tmp_path: Path) -> None:
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

    batch = _load_packed_batch(paths, torch.device("cpu"))

    assert batch["input_ids"].tolist() == [[20, 30, 50, 60, 70]]
    assert batch["hidden_states"].tolist() == [[[1.0], [2.0], [4.0], [5.0], [6.0]]]
    assert batch["verifier_last_hidden_states"].tolist() == [[[12.0], [13.0], [15.0], [16.0], [17.0]]]
    assert batch["loss_mask"].tolist() == [[True, True, True, False, True]]
    assert batch["position_ids"].tolist() == [[5, 6, 11, 12, 13]]
    assert batch["document_ids"].tolist() == [[0, 0, 1, 1, 1]]


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
