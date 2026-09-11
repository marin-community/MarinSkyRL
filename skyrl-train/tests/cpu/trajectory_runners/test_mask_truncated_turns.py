"""generator.mask_truncated_turns: zero the loss mask over only the sampled tokens of turns that
hit the per-turn output cap (max_generate_length), leaving the rest of the trajectory trainable."""

from types import SimpleNamespace

from omegaconf import OmegaConf

from skyrl_train.trajectory_runners.harbor.truncation_penalty import truncated_turn_spans
from skyrl_train.trajectory_runners.projections import _loss_masks, mask_truncated_turns


class _Tok:
    eos_token_id = 128001


def _output(mask, *, spans=None, loss_eligible=True):
    return SimpleNamespace(
        loss_mask=list(mask),
        disposition=SimpleNamespace(loss_eligible=loss_eligible),
        evidence=SimpleNamespace(stop_reason="complete"),
        turn_truncated=bool(spans),
        truncated_turn_spans=spans,
    )


# --- span detection (runner side) -------------------------------------------------------------

# Full-TITO layout: prompt_0 = [p p p], turn 0 = [a a], obs = [o], turn 1 = [b b b b] (capped at 4),
# obs = [o o], turn 2 = [c].
_P0 = [1, 1, 1]
_TURN_IDS = [[10, 11], [20, 21, 22, 23], [30]]
_RESPONSE = [10, 11, 7, 20, 21, 22, 23, 8, 8, 30]
_PROMPT_IDS = [
    _P0,  # turn 0 prompt
    _P0 + [10, 11, 7],  # turn 1 prompt: p0 + turn0 + obs
    _P0 + [10, 11, 7, 20, 21, 22, 23, 8, 8],  # turn 2 prompt
]


def test_spans_cover_exactly_the_capped_turn():
    spans = truncated_turn_spans(_TURN_IDS, _PROMPT_IDS, len(_P0), _RESPONSE, max_generate_length=4)
    assert spans == [(3, 7)]
    assert _RESPONSE[3:7] == _TURN_IDS[1]


def test_spans_empty_when_no_turn_reaches_the_cap():
    assert truncated_turn_spans(_TURN_IDS, _PROMPT_IDS, len(_P0), _RESPONSE, max_generate_length=5) == []


def test_spans_skip_turns_whose_offset_does_not_match_the_served_ids():
    # Re-tokenized context: the response no longer carries the served ids at the TITO offset.
    response = [10, 11, 7, 20, 99, 22, 23, 8, 8, 30]
    assert truncated_turn_spans(_TURN_IDS, _PROMPT_IDS, len(_P0), response, max_generate_length=4) == []


def test_spans_safe_defaults_on_missing_inputs():
    assert truncated_turn_spans(None, _PROMPT_IDS, 3, _RESPONSE, 4) == []
    assert truncated_turn_spans(_TURN_IDS, None, 3, _RESPONSE, 4) == []
    assert truncated_turn_spans(_TURN_IDS, _PROMPT_IDS, 3, _RESPONSE, 0) == []
    # Out-of-range offset (prompt ids shorter than the initial prompt) is skipped, not raised.
    assert truncated_turn_spans([[20, 21, 22, 23]], [[1]], 3, _RESPONSE, 4) == []


def test_multiple_capped_turns_each_get_a_span():
    turn_ids = [[10, 11, 12, 13], [20, 21, 22, 23]]
    prompt_ids = [_P0, _P0 + [10, 11, 12, 13, 7]]
    response = [10, 11, 12, 13, 7, 20, 21, 22, 23]
    assert truncated_turn_spans(turn_ids, prompt_ids, len(_P0), response, 4) == [(0, 4), (5, 9)]


# --- mask projection ---------------------------------------------------------------------------


def test_mask_truncated_turns_zeroes_only_the_spans():
    outputs = [
        _output([1, 1, 0, 1, 1, 1, 1, 0, 0, 1], spans=[(3, 7)]),
        _output([1, 1, 0, 1], spans=None),
    ]
    masks = mask_truncated_turns([o.loss_mask for o in outputs], outputs)
    assert masks == [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 1]]
    # inputs untouched
    assert outputs[0].loss_mask == [1, 1, 0, 1, 1, 1, 1, 0, 0, 1]


def test_mask_truncated_turns_clips_spans_to_the_mask_length():
    outputs = [_output([1, 1, 1], spans=[(2, 10)])]
    assert mask_truncated_turns([o.loss_mask for o in outputs], outputs) == [[1, 1, 0]]


def test_loss_masks_default_off_is_unchanged():
    outputs = [_output([1, 1, 1, 1], spans=[(1, 3)])]
    cfg = OmegaConf.create({"apply_overlong_filtering": False})
    assert _loss_masks(outputs, [[5, 6, 7, 8]], cfg, _Tok()) == [[1, 1, 1, 1]]


def test_loss_masks_with_flag_masks_the_capped_turn_and_keeps_the_rest():
    outputs = [_output([1, 1, 1, 1], spans=[(1, 3)]), _output([1, 1], spans=None)]
    cfg = OmegaConf.create({"apply_overlong_filtering": False, "mask_truncated_turns": True})
    assert _loss_masks(outputs, [[5, 6, 7, 8], [5, 128009]], cfg, _Tok()) == [[1, 0, 0, 1], [1, 1]]


def test_mask_truncated_turns_composes_with_mask_length_stops():
    # mask_length_stops zeroes the whole sample; the per-turn mask has nothing left to do.
    outputs = [_output([1, 1, 1, 1], spans=[(1, 3)])]
    cfg = OmegaConf.create({"apply_overlong_filtering": False, "mask_length_stops": True, "mask_truncated_turns": True})
    assert _loss_masks(outputs, [[5, 6, 7, 8]], cfg, _Tok()) == [[0, 0, 0, 0]]
