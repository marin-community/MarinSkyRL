"""trainer.algorithm.context_distillation: the guidance span leaves the first user message before the trainer
tokenizes the prompt; the served prompt rides along as a batch column with per-batch counters."""

from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.metric_names import (
    CONTEXT_DISTILLATION_ABSENT_METRIC,
    CONTEXT_DISTILLATION_EDITED_METRIC,
    CONTEXT_DISTILLATION_FAILED_METRIC,
    CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC,
)
from skyrl_train.trajectory_runners.context_distillation import (
    CONTEXT_EDITED_KEY,
    DEFAULT_END_MARKER,
    DEFAULT_START_MARKER,
    ROLLOUT_PROMPT_TOKEN_IDS_KEY,
    ContextDistillationConfig,
    ContextEdit,
    ContextEditStatus,
    context_distillation_batch_fields,
    strip_guidance_suffix,
)
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.utils import validate_cfg

BLOCK = "Make every source change with a short Python script.\nDo not use sed -i."
INSTRUCTION = "Fix the bug in foo().\n"
TERMINAL = f"{DEFAULT_END_MARKER}\nroot@host:/app#"
PROMPTED = f"Task Description:\n{INSTRUCTION}{DEFAULT_START_MARKER}{BLOCK}{TERMINAL}"
BARE = f"Task Description:\n{INSTRUCTION}{TERMINAL}"


# ---------------------------------------------------------------------------
# strip_guidance_suffix
# ---------------------------------------------------------------------------
def test_strip_removes_exactly_the_span():
    edit = strip_guidance_suffix(PROMPTED, DEFAULT_START_MARKER, DEFAULT_END_MARKER)
    assert edit.status is ContextEditStatus.STRIPPED
    assert edit.stripped and not edit.failed
    assert edit.text == BARE
    assert edit.removed_chars == len(DEFAULT_START_MARKER) + len(BLOCK)


def test_strip_without_marker_is_absent_and_unchanged():
    edit = strip_guidance_suffix(BARE, DEFAULT_START_MARKER, DEFAULT_END_MARKER)
    assert edit.status is ContextEditStatus.ABSENT
    assert not edit.stripped and not edit.failed
    assert edit.text == BARE
    assert edit.removed_chars == 0


def test_strip_repeated_start_marker_fails_loudly():
    content = PROMPTED + DEFAULT_START_MARKER + "again" + DEFAULT_END_MARKER
    edit = strip_guidance_suffix(content, DEFAULT_START_MARKER, DEFAULT_END_MARKER)
    assert edit.status is ContextEditStatus.START_REPEATED
    assert edit.failed
    assert edit.text == content


def test_strip_missing_end_marker_fails_loudly():
    content = f"{INSTRUCTION}{DEFAULT_START_MARKER}{BLOCK}"
    edit = strip_guidance_suffix(content, DEFAULT_START_MARKER, DEFAULT_END_MARKER)
    assert edit.status is ContextEditStatus.END_MISSING
    assert edit.failed
    assert edit.text == content


def test_strip_ignores_an_end_marker_before_the_start_marker():
    content = f"{DEFAULT_END_MARKER}x{DEFAULT_START_MARKER}{BLOCK}"
    edit = strip_guidance_suffix(content, DEFAULT_START_MARKER, DEFAULT_END_MARKER)
    assert edit.status is ContextEditStatus.END_MISSING


def test_strip_empty_end_marker_removes_to_the_end():
    edit = strip_guidance_suffix(f"{INSTRUCTION}{DEFAULT_START_MARKER}{BLOCK}", DEFAULT_START_MARKER, "")
    assert edit.stripped
    assert edit.text == INSTRUCTION


def test_strip_refuses_a_span_longer_than_the_bound():
    edit = strip_guidance_suffix(PROMPTED, DEFAULT_START_MARKER, DEFAULT_END_MARKER, max_removed_chars=10)
    assert edit.status is ContextEditStatus.SPAN_TOO_LONG
    assert edit.failed
    assert edit.text == PROMPTED
    assert edit.removed_chars == len(DEFAULT_START_MARKER) + len(BLOCK)
    assert strip_guidance_suffix(PROMPTED, DEFAULT_START_MARKER, DEFAULT_END_MARKER, max_removed_chars=4096).stripped


def test_strip_requires_a_start_marker():
    with pytest.raises(ValueError):
        strip_guidance_suffix(PROMPTED, "", DEFAULT_END_MARKER)


# ---------------------------------------------------------------------------
# ContextDistillationConfig
# ---------------------------------------------------------------------------
def test_config_absent_block_means_disabled():
    config = ContextDistillationConfig.from_algorithm_config(OmegaConf.create({"use_tis": True}))
    assert config == ContextDistillationConfig.disabled()
    assert not config.enabled


def test_config_base_defaults_are_off_and_match_the_terminus_layout():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    config = ContextDistillationConfig.from_algorithm_config(cfg.trainer.algorithm)
    assert not config.enabled
    assert config.start_marker == DEFAULT_START_MARKER == "\n\n---\nWorking guidance:\n"
    assert config.end_marker == DEFAULT_END_MARKER == "\n\nCurrent terminal state:"
    assert (config.on_failure, config.tis_reference, config.kl_reference) == ("error", "rollout", "training")
    assert config.max_removed_chars == 4096


def test_config_forward_predicates_follow_the_reference_choices():
    config = ContextDistillationConfig.from_algorithm_config(
        OmegaConf.create({"context_distillation": {"enabled": True}})
    )
    assert config.enabled
    assert config.rollout_context_forward_needed  # tis_reference=rollout
    assert not config.reference_uses_rollout_context  # kl_reference=training by default
    teacher = ContextDistillationConfig(enabled=True, kl_reference="rollout")
    assert teacher.reference_uses_rollout_context
    lean = ContextDistillationConfig(enabled=True, tis_reference="none")
    assert not lean.rollout_context_forward_needed
    # the predicates ignore `enabled`: a batch that carries stripped prompts is handled on its own evidence
    assert ContextDistillationConfig.disabled().rollout_context_forward_needed


@pytest.mark.parametrize(
    "override",
    [
        {"on_failure": "ignore"},
        {"tis_reference": "training"},
        {"kl_reference": "none"},
        {"enabled": True, "start_marker": ""},
        {"max_removed_chars": 0},
    ],
)
def test_config_rejects_unknown_choices(override):
    with pytest.raises(ValueError, match="trainer.algorithm.context_distillation"):
        ContextDistillationConfig.from_algorithm_config(OmegaConf.create({"context_distillation": override}))


ENABLED_OVERRIDE = "trainer.algorithm.context_distillation.enabled=true"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (["trainer.algorithm.context_distillation.tis_reference=x"], "tis_reference"),
        ([ENABLED_OVERRIDE, "trainer.step_wise_training=true"], "step_wise_training"),
        ([ENABLED_OVERRIDE, "trainer.strategy=megatron"], "megatron"),
        ([ENABLED_OVERRIDE, "trainer.algorithm.policy_loss_type=dppo"], "regular or dual_clip"),
        (
            [
                ENABLED_OVERRIDE,
                "trainer.algorithm.context_distillation.tis_reference=none",
                "trainer.algorithm.use_tis=true",
                "trainer.algorithm.tis_imp_ratio_cap=2.0",
                "trainer.fully_async.max_staleness_steps=2",
            ],
            "staleness",
        ),
        (
            [
                ENABLED_OVERRIDE,
                "trainer.algorithm.context_distillation.kl_reference=rollout",
                "trainer.algorithm.use_kl_in_reward=true",
            ],
            "use_kl_in_reward",
        ),
        (
            [
                ENABLED_OVERRIDE,
                "trainer.algorithm.context_distillation.kl_reference=rollout",
                "trainer.algorithm.use_kl_loss=true",
                "trainer.algorithm.kl_estimator_type=k3",
            ],
            "k2",
        ),
    ],
)
def test_validate_cfg_rejects_silently_wrong_combinations(overrides, match):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=overrides)
    with pytest.raises(ValueError, match=match):
        validate_cfg(cfg)


# ---------------------------------------------------------------------------
# Batch columns and counters
# ---------------------------------------------------------------------------
def _output(prompt, *, rollout_prompt=None, edit=None, loss_eligible=True):
    return SimpleNamespace(
        evidence=SimpleNamespace(prompt_token_ids=tuple(prompt)),
        rollout_prompt_token_ids=rollout_prompt,
        context_edit=edit,
        disposition=SimpleNamespace(loss_eligible=loss_eligible),
    )


def test_batch_fields_count_edited_failed_and_absent_rows():
    stripped = ContextEdit(ContextEditStatus.STRIPPED, BARE, removed_chars=5)
    failed = ContextEdit(ContextEditStatus.END_MISSING, PROMPTED)
    outputs = [
        _output([1, 2], rollout_prompt=[1, 2, 8, 9, 7], edit=stripped),
        _output([3], edit=ContextEdit(ContextEditStatus.ABSENT, BARE)),
        _output([4, 5], edit=failed, loss_eligible=False),
        # stripped, then invalidated by a peer failure: trains nothing, so it is not an edited row
        _output([0], rollout_prompt=[1, 2, 8], edit=stripped, loss_eligible=False),
        _output([6], edit=None),  # a failed trajectory carries no edit
    ]
    fields, counts = context_distillation_batch_fields(outputs)
    assert fields[ROLLOUT_PROMPT_TOKEN_IDS_KEY] == [[1, 2, 8, 9, 7], [3], [4, 5], [0], [6]]
    assert fields[CONTEXT_EDITED_KEY] == [True, False, False, False, False]
    assert counts == {
        CONTEXT_DISTILLATION_EDITED_METRIC: 1.0,
        CONTEXT_DISTILLATION_FAILED_METRIC: 1.0,
        CONTEXT_DISTILLATION_ABSENT_METRIC: 3.0,
        CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC: 3.0,
    }


def _trajectory_batch(prompt, rollout_prompt, edited, counts):
    return {
        "prompt_token_ids": [prompt],
        "response_ids": [[10, 11]],
        "rewards": [1.0],
        "unshaped_rewards": [1.0],
        "loss_masks": [[1, 1]],
        "stop_reasons": ["complete"],
        "rollout_logprobs": [[-0.1, -0.2]],
        "exclude_from_baseline": [False],
        "rollout_metrics": counts,
        ROLLOUT_PROMPT_TOKEN_IDS_KEY: [rollout_prompt],
        CONTEXT_EDITED_KEY: [edited],
    }


def test_concatenate_carries_the_columns_and_sums_the_counters():
    first = _trajectory_batch(
        [1, 2],
        [1, 2, 8],
        True,
        {CONTEXT_DISTILLATION_EDITED_METRIC: 1.0, CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC: 1.0},
    )
    second = _trajectory_batch(
        [3],
        [3],
        False,
        {CONTEXT_DISTILLATION_ABSENT_METRIC: 1.0, CONTEXT_DISTILLATION_EDITED_METRIC: 0.0},
    )
    merged = concatenate_trajectory_batches([first, second], tis_lcs_alert_threshold=0.1)
    assert merged[ROLLOUT_PROMPT_TOKEN_IDS_KEY] == [[1, 2, 8], [3]]
    assert merged[CONTEXT_EDITED_KEY] == [True, False]
    metrics = merged["rollout_metrics"]
    assert metrics[CONTEXT_DISTILLATION_EDITED_METRIC] == 1.0
    assert metrics[CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC] == 1.0
    assert metrics[CONTEXT_DISTILLATION_ABSENT_METRIC] == 1.0


def test_concatenate_fills_a_group_that_lacks_the_columns_in_either_order():
    with_columns = _trajectory_batch([1, 2], [1, 2, 8], True, {CONTEXT_DISTILLATION_EDITED_METRIC: 1.0})
    without = _trajectory_batch([3], [3], False, {})
    del without[ROLLOUT_PROMPT_TOKEN_IDS_KEY], without[CONTEXT_EDITED_KEY]  # an all-failed orchestrator group
    for batches, prompts, edited in (
        ([with_columns, without], [[1, 2, 8], [3]], [True, False]),
        ([without, with_columns], [[3], [1, 2, 8]], [False, True]),
    ):
        merged = concatenate_trajectory_batches(batches, tis_lcs_alert_threshold=0.1)
        assert merged[ROLLOUT_PROMPT_TOKEN_IDS_KEY] == prompts
        assert merged[CONTEXT_EDITED_KEY] == edited
        assert merged["rollout_metrics"][CONTEXT_DISTILLATION_EDITED_METRIC] == 1.0
