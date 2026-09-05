"""generator.mask_length_stops: zero the loss mask of length-stopped samples by the runner's stop signals."""

from types import SimpleNamespace

from omegaconf import OmegaConf

from skyrl_train.trajectory_runners.projections import _loss_masks, is_length_stopped, mask_length_stops


def _output(mask, *, stop_reason="complete", turn_truncated=False, loss_eligible=True):
    return SimpleNamespace(
        loss_mask=list(mask),
        disposition=SimpleNamespace(loss_eligible=loss_eligible),
        evidence=SimpleNamespace(stop_reason=stop_reason),
        turn_truncated=turn_truncated,
    )


class _Tok:
    eos_token_id = 128001


def test_is_length_stopped_reads_turn_flag_and_terminal_reason():
    assert is_length_stopped(_output([1], turn_truncated=True))
    assert is_length_stopped(_output([1], stop_reason="length"))
    assert not is_length_stopped(_output([1]))
    assert not is_length_stopped(SimpleNamespace())  # no evidence, no flag


def test_mask_length_stops_zeroes_only_flagged_samples():
    outputs = [_output([1, 1, 0]), _output([1, 1], stop_reason="length"), _output([1, 0, 1], turn_truncated=True)]
    masks = mask_length_stops([o.loss_mask for o in outputs], outputs)
    assert masks == [[1, 1, 0], [0, 0], [0, 0, 0]]


def test_loss_masks_default_off_is_unchanged():
    outputs = [_output([1, 1], stop_reason="length")]
    cfg = OmegaConf.create({"apply_overlong_filtering": False})
    assert _loss_masks(outputs, [[5, 6]], cfg, _Tok()) == [[1, 1]]


def test_loss_masks_with_flag_masks_length_stops_but_not_eos_endings():
    outputs = [_output([1, 1], stop_reason="length"), _output([1, 1])]
    cfg = OmegaConf.create({"apply_overlong_filtering": False, "mask_length_stops": True})
    # second response ends in <|eot_id|> (128009), not the tokenizer eos: must stay trainable
    assert _loss_masks(outputs, [[5, 6], [7, 128009]], cfg, _Tok()) == [[0, 0], [1, 1]]


def test_apply_overlong_filtering_alone_silences_eot_id_endings():
    # documents why mask_length_stops exists for Llama-3 style templates
    outputs = [_output([1, 1])]
    cfg = OmegaConf.create({"apply_overlong_filtering": True})
    assert _loss_masks(outputs, [[7, 128009]], cfg, _Tok()) == [[0, 0]]
