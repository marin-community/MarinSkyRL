"""diag/*: declared-done reward decomposition (f, q) and the length-stop mask fraction."""

from skyrl_train.utils import diag_utils


def _diag(stop_reasons, successes, loss_masks=None):
    n = len(stop_reasons)
    return diag_utils.compute_group_diagnostics(
        rewards=[1.0 if s else 0.0 for s in successes],
        successes=successes,
        uids=[i // 2 for i in range(n)],
        response_lengths=[10] * n,
        stop_reasons=stop_reasons,
        n_samples_per_prompt=2,
        loss_masks=loss_masks,
    )


def test_declared_done_fraction_and_reward_given_done():
    stops = ["task_complete", "task_complete", "complete", "length"]
    succ = [True, False, False, False]
    m = _diag(stops, succ)
    assert m["diag/declared_done_fraction"] == 0.5
    assert m["diag/reward_given_done"] == 0.5
    assert m["diag/truncated_fraction"] == 0.25


def test_reward_given_done_is_zero_when_nothing_declared():
    m = _diag(["complete", "length"], [False, False])
    assert m["diag/declared_done_fraction"] == 0.0
    assert m["diag/reward_given_done"] == 0.0


def test_length_stop_masked_fraction_counts_only_zeroed_length_stops():
    stops = ["length", "length", "task_complete", "complete"]
    masks = [[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1]]  # second length stop still trainable; the zeroed task_complete does not count
    m = _diag(stops, [False, False, True, False], loss_masks=masks)
    assert m["diag/length_stop_masked_fraction"] == 0.25
    assert "diag/length_stop_masked_fraction" not in _diag(stops, [False] * 4)
