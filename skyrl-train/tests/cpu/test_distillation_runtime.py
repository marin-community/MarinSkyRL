import pytest
import torch

from skyrl_train.distillation import SampledReverseKLInput
from skyrl_train.distillation_adapters import RoutedScoredDistillationBatch
from skyrl_train.distillation_runtime import AsyncDistillationRuntime
from skyrl_train.teacher_routing import TeacherRoute
from skyrl_train.training_batch import TrainingInputBatch


def _scored_group(trajectory_id: str, logprobs: list[float], *, revision: str):
    width = len(logprobs)
    route = TeacherRoute(
        route_id="default",
        teacher_id="teacher-a",
        objective_id="sampled_reverse_kl",
        weight=1.0,
        plan_version="routes-r1",
    )
    return RoutedScoredDistillationBatch(
        trajectory_ids=(trajectory_id,),
        routes=(route,),
        teacher_revisions=(revision,),
        plan_version="routes-r1",
        evidence=(),
        distillation=SampledReverseKLInput(
            teacher_action_log_probs=torch.tensor([logprobs]),
            valid_mask=torch.ones((1, width), dtype=torch.bool),
            loss_weights=torch.full((1, width), 0.5),
        ),
    )


def test_async_runtime_attaches_admission_ordered_evidence_with_safe_padding():
    runtime = object.__new__(AsyncDistillationRuntime)
    training_input = TrainingInputBatch({"response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0]])})
    training_input.metadata = {"pad_size": 1}

    runtime.attach_to_training_input(
        training_input,
        (
            _scored_group("trajectory-a", [-0.1, -0.2], revision="teacher-r1"),
            _scored_group("trajectory-b", [-0.3], revision="teacher-r2"),
        ),
    )

    torch.testing.assert_close(
        training_input["teacher_action_log_probs"],
        torch.tensor([[-0.1, -0.2, torch.nan], [-0.3, torch.nan, torch.nan], [torch.nan] * 3]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        training_input["distillation_loss_weights"],
        torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert not training_input["teacher_valid_mask"][2].any()
    assert training_input.metadata["distillation_teacher_revisions"] == ("teacher-r1", "teacher-r2")
    assert training_input.metadata["distillation_plan_versions"] == ("routes-r1", "routes-r1")


def test_async_runtime_rejects_evidence_that_does_not_match_the_unpadded_batch():
    runtime = object.__new__(AsyncDistillationRuntime)
    training_input = TrainingInputBatch({"response_mask": torch.ones((2, 2), dtype=torch.bool)})
    training_input.metadata = {"pad_size": 0}

    with pytest.raises(ValueError, match="scored distillation rows must match"):
        runtime.attach_to_training_input(
            training_input,
            (_scored_group("trajectory-a", [-0.1], revision="teacher-r1"),),
        )
