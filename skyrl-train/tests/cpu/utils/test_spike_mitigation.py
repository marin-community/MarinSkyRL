import pytest
import torch

from skyrl_train.utils.stale_clip import StaleClip
from skyrl_train.utils.zclip import ZClip


def test_zclip_clamps_spike_after_warmup_and_resumes_from_state_dict():
    zclip = ZClip(warmup_steps=3, max_grad_norm=5.0)
    # Warmup returns the static ceiling, then EMA mean=1.0, var=0.02/3.
    assert [zclip.compute_max_norm(g) for g in (1.0, 1.1, 0.9)] == [5.0, 5.0, 5.0]
    # An in-distribution norm leaves the static ceiling in place.
    assert zclip.compute_max_norm(1.0) == 5.0

    restored = ZClip(warmup_steps=3, max_grad_norm=5.0)
    restored.load_state_dict(zclip.state_dict())

    # Hand-computed: mean=1.0, std=sqrt(0.97 * 0.02 / 3)=0.08042, z=24.87,
    # threshold = mean + z_thresh * std / (z / z_thresh) = 1.0202.
    for clipper in (zclip, restored):
        assert clipper.compute_max_norm(3.0) == pytest.approx(1.0202, abs=1e-4)
        assert clipper.last_decision["triggered"] == 1.0


@pytest.mark.parametrize(
    ("entropies", "stale_min", "expected_scale"),
    [
        ([0.1, 0.1], 2, 0.4),
        ([0.1, 0.1], 5, 0.1),  # floored at min_lr_scale
        ([0.1, 0.1], 0, 1.0),  # on-policy anchor present
        ([0.1, 0.3], 2, 1.0),  # rolling mean 0.2 is above the threshold
        ([], 2, 1.0),  # no entropy history yet
    ],
)
def test_stale_clip_damps_learning_rate_only_for_stale_concentrated_batches(entropies, stale_min, expected_scale):
    stale_clip = StaleClip(alpha=0.3, entropy_threshold=0.15, min_lr_scale=0.1)
    for entropy in entropies:
        stale_clip.update_entropy(entropy)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.5)

    scale = stale_clip.compute_lr_scale(stale_min)
    original = StaleClip.apply_scale(optimizer, scale)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * expected_scale)

    StaleClip.restore_lrs(optimizer, original)
    assert optimizer.param_groups[0]["lr"] == 0.5
