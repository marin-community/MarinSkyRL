"""The sender refuses to send for the wrong update, or after its parameters were reallocated."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.expert_block.stream import InstallReport, storage_identity


def bound_sender(completed_update):
    worker = SimpleNamespace(_model_version_step=completed_update)
    sender = ExpertBlockSender(worker, parallel_state=None)
    sender.sources = {"decoder.layers.0.weight": torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))}
    sender.identity = storage_identity(sender.sources)
    sender.stream = SimpleNamespace(run=lambda version: InstallReport(0, version, 2, 48, 0.01))
    return sender


def test_sends_the_update_it_finished_and_the_loaded_weights_before_any_update():
    assert bound_sender(completed_update=4).send_weights({"version": 4})["version"] == 4
    assert bound_sender(completed_update=None).send_weights({"version": 9})["version"] == 9


def test_refuses_a_version_that_is_not_the_completed_update():
    with pytest.raises(RuntimeError, match="names update 5 but this rank last completed 4"):
        bound_sender(completed_update=4).send_weights({"version": 5})


def test_refuses_when_a_parameter_was_reassigned_new_storage_since_preparation():
    sender = bound_sender(completed_update=1)
    # An in-place update keeps the storage and is accepted. Reassigning ``.data`` is not.
    sender.sources["decoder.layers.0.weight"].data.fill_(1)
    assert sender.send_weights({"version": 1})["expert_matrices"] == 2
    sender.sources["decoder.layers.0.weight"].data = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="storage changed"):
        sender.send_weights({"version": 1})
