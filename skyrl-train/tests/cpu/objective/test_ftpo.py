"""FTPO token selection and loss checked against the scalar definition."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.training_batch import TrainingBatchIterator
from skyrl_train.config.ftpo import FTPOConfig, validate_ftpo
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.ftpo import (
    FTPOInputs,
    FTPOTargets,
    boundary_values,
    compact_boundary_logits,
    ftpo_counts,
    ftpo_loss,
    select_ftpo_candidates,
)
from skyrl_train.objective.losses import ftpo_policy_loss
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.trajectory_runners.trajectory_reward_shaping import LoopCreditConfig, tail_loop


@pytest.fixture
def config():
    config_dir = str(Path(__file__).resolve().parents[3] / "skyrl_train" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="ppo_base_config",
            overrides=[
                "+algorithm_recipe=ftpo",
                "trainer.use_sample_packing=false",
                "generator.sampling_params.logprobs=4",
                "generator.sampling_params.temperature=0",
                "trainer.logger=console",
            ],
        )


def _surface(ids):
    return {0: "Wait", 1: " So", 2: " Therefore", 3: "Done", 4: "\n", 5: "WAIT"}[ids[0]]


def test_ftpo_reuses_tail_detection_but_selects_first_repeat_and_filters_alternatives():
    response = [[3, 1] + [0, 1] * 4, [0, 1] * 4 + [3], [0, 1] * 4]
    masks = [[0, 0] + [1] * 8, [1] * 9, [1] * 8]
    ids = torch.tensor([0, 5, 4, 2, 3]).expand(3, 10, 5).clone()
    scores = torch.tensor([0.5, 0.3, 0.1, 0.09, 0.01]).log().expand_as(ids).clone()
    chosen, weights = select_ftpo_candidates(
        response,
        masks,
        ids,
        scores,
        decode=_surface,
        loop=LoopCreditConfig(),
        config=replace(FTPOConfig(), min_p=0.2, chosen_balance_strength=0),
        seed=4,
        eligible=[True, True, False],
    )
    hit = tail_loop(response[0], masks[0], LoopCreditConfig())
    assert (hit.start, hit.period, hit.repeat_start) == (2, 2, 4)
    assert chosen.nonzero().tolist() == [[0, 4, 3]]
    assert weights.nonzero().tolist() == [[0, 4]]
    assert ids[chosen].tolist() == [2]


def test_ftpo_balances_common_tokens_deterministically_without_retokenizing():
    responses = [[0] * 6] * 6 + [[1] * 6] * 2
    ids = torch.tensor([2, 3]).expand(8, 6, 2).clone()
    ids[6:, :, 1] = 0
    scores = torch.tensor([0.7, 0.3]).log().expand_as(ids).clone()
    kwargs = dict(decode=_surface, loop=LoopCreditConfig(), config=FTPOConfig(), seed=13, eligible=[True] * 8)
    chosen, weights = select_ftpo_candidates(responses, [[1] * 6] * 8, ids, scores, **kwargs)
    repeated, repeated_weights = select_ftpo_candidates(responses, [[1] * 6] * 8, ids, scores, **kwargs)
    assert torch.equal(chosen, repeated) and torch.equal(weights, repeated_weights)
    assert chosen.sum() < 16
    common = weights[:6].sum(-1)
    rare = weights[6:].sum(-1)
    assert common[common > 0].mean() < rare[rare > 0].mean()
    assert torch.all(weights[~chosen.any(-1)] == 0)


def _scalar_reference(logits, reference, ids, masks, rejected, weights, params):
    pref = logits.sum() * 0
    other = logits.sum() * 0
    target = logits.sum() * 0
    example_count = weights.sum()
    other_count = target_count = 0.0
    for row in range(len(logits)):
        if weights[row] == 0:
            continue
        chosen = ids[row][masks[row]].tolist()
        terms = []
        for token in chosen:
            gap = params.margin - (logits[row, token] - logits[row, rejected[row]])
            terms.append(torch.logaddexp(gap, gap.new_zeros(())) * (gap / params.margin).clamp(0, 1))
        pref = pref + weights[row] * sum(terms) / len(terms)
        for token in range(logits.shape[-1]):
            delta = logits[row, token] - reference[row, token]
            if token in chosen or token == rejected[row]:
                target = target + weights[row] * (delta.abs() - params.tau_mse_target).clamp(min=0).square()
                target_count += float(weights[row])
            else:
                other = other + weights[row] * delta.square()
                other_count += float(weights[row])
    return (
        pref / example_count
        + params.lambda_mse * other / other_count
        + params.lambda_mse_target * target / target_count
    )


@pytest.mark.parametrize("weight_scale", [0.1, 1.0])
@pytest.mark.parametrize("micro_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ftpo_value_and_gradient_match_scalar_reference_across_microbatches(config, micro_size, dtype, weight_scale):
    torch.manual_seed(9)
    logits = torch.randn(4, 3, 7, dtype=dtype).requires_grad_()
    ref = torch.randn(4, 7, dtype=dtype)
    ids = torch.tensor([1, 2, 3]).expand(4, 3, 3).clone()
    chosen = torch.zeros_like(ids, dtype=torch.bool)
    chosen[0, 1, :2] = True
    chosen[1, 0, :1] = True
    chosen[3, 2, :] = True
    weights = chosen.any(-1).float() * torch.tensor([0.3, 0.7, 0, 0.8])[:, None] * weight_scale
    rejected = torch.zeros(4, dtype=torch.long)
    targets = FTPOTargets(ids, chosen, ref)
    normalization = ftpo_counts([targets], [weights], lambda x: x)
    counts = step_counts([weights], [weights], [], [torch.zeros_like(weights)], 3, lambda x: x)
    boundary = boundary_values(logits, chosen)
    expected = _scalar_reference(
        boundary.float(),
        ref.float(),
        boundary_values(ids, chosen),
        boundary_values(chosen, chosen),
        rejected,
        weights.sum(-1),
        FTPOConfig(),
    )
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, 4, micro_size):
        chunk = slice(start, start + micro_size)
        target = FTPOTargets(ids[chunk], chosen[chunk], ref[chunk])
        inputs = FTPOInputs(boundary[chunk], target, rejected[chunk], normalization)
        batch = build_objective_micro_batch(
            action_log_probs=torch.zeros_like(weights[chunk]),
            old_action_log_probs=torch.zeros_like(weights[chunk]),
            base_action_log_probs=None,
            advantages=torch.zeros_like(weights[chunk]),
            loss_mask=weights[chunk],
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(weights[chunk]),
            think_token_weight=1,
            teacher=None,
            ftpo=inputs,
        )
        actual = (
            actual
            + compute_policy_objective(
                batch,
                loss=ftpo_policy_loss,
                counts=counts,
                config=config.trainer.algorithm,
                loss_scale=1,
                report_scale=1,
            ).optimization_loss
        )
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual_gradient[~chosen.any(-1)]) == 0


def test_ftpo_saturated_preferences_stop_and_reference_tether_penalizes_other_tokens():
    chosen = torch.tensor([[[False, True]]])
    ids = torch.tensor([[[0, 1]]])
    logits = torch.tensor([[0.0, 3.0, 2.0]], requires_grad=True)
    target = FTPOTargets(ids, chosen, torch.tensor([[0.0, 3.0, 0.0]]))
    counts = ftpo_counts([target], [torch.ones(1, 1)], lambda x: x)
    values, metrics = ftpo_loss(FTPOInputs(logits, target, torch.tensor([0]), counts), FTPOConfig())
    values.sum().backward()
    assert values.item() == pytest.approx(1.6)
    torch.testing.assert_close(logits.grad, torch.tensor([[0.0, 0.0, 1.6]]))
    assert metrics["ftpo/chosen_win_sum"] == 1


def test_full_weight_ftpo_update_and_resume_match_uninterrupted_training(tmp_path):
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0,
        )
    )
    reference = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sequences = torch.tensor([[4, 5, 0, 0, 0, 0]])
    chosen = torch.zeros(1, 4, 2, dtype=torch.bool)
    chosen[0, 1, :] = True
    ids = torch.tensor([1, 2]).expand(1, 4, 2)
    with torch.no_grad():
        ref_logits = boundary_values(reference(sequences).logits[:, -5:-1], chosen)
    targets = FTPOTargets(ids, chosen, ref_logits)
    counts = ftpo_counts([targets], [chosen.any(-1).float()], lambda x: x)

    def update(policy, optim):
        optim.zero_grad()
        logits = boundary_values(policy(sequences).logits[:, -5:-1], chosen)
        values, _ = ftpo_loss(FTPOInputs(logits, targets, torch.tensor([0]), counts), FTPOConfig())
        values.sum().backward()
        optim.step()
        return logits.detach()

    before = update(model, optimizer)
    checkpoint = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    update(model, optimizer)
    resumed = deepcopy(reference)
    resumed.requires_grad_(True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    state = torch.load(checkpoint, weights_only=True)
    resumed.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    update(resumed, resumed_optimizer)
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        after = boundary_values(model(sequences).logits[:, -5:-1], chosen)
    assert (after[0, 1:3] - after[0, 0]).mean() > (before[0, 1:3] - before[0, 0]).mean()
    assert not torch.equal(model.model.embed_tokens.weight, reference.model.embed_tokens.weight)


def test_ftpo_recipe_accepts_greedy_capture_and_rejects_unsupported_geometry(config):
    validate_cfg(config)
    assert config.generator.engine_init_kwargs.logprobs_mode == "raw_logprobs"
    for path in (
        "trainer.policy.megatron_config.tensor_model_parallel_size",
        "trainer.ref.megatron_config.context_parallel_size",
        "trainer.use_sample_packing",
    ):
        changed = deepcopy(config)
        OmegaConf.update(changed, path, True if path.endswith("packing") else 2)
        with pytest.raises(ValueError, match="FTPO"):
            validate_ftpo(changed)


def test_ftpo_rollout_payload_survives_padding_serialization_and_microbatching(config, tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"Wait": 0, "So": 1, "Therefore": 2, "Done": 3, "[PAD]": 4})),
        pad_token="[PAD]",
    )
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    trainer.ref_model = trainer.critic_model = None
    responses = [[0, 1] * 4, [2, 3], [0] * 5]
    trajectory = {
        "prompt_token_ids": [[2, 3], [3], [1, 2, 3]],
        "response_ids": responses,
        "loss_masks": [[1] * len(row) for row in responses],
        "rewards": [[0.0] * len(row) for row in responses],
        "student_topk_indices": [np.tile([0, 1, 2, 3], (len(row), 1)) for row in responses],
        "behavior_topk_logprobs": [
            np.tile(np.log([0.5, 0.3, 0.15, 0.05]), (len(row), 1)).astype(np.float32) for row in responses
        ],
        "is_last_step": [True, True, True],
        "exclude_from_baseline": [False, False, True],
    }
    batch = trainer.convert_to_training_input(trajectory, ["a", "b", "c"])
    assert batch.batch_size == 4
    assert batch["loss_mask"].nonzero().tolist() == [[0, 2]]
    # The model sees compacted tokens; boundary logits must still refer to the
    # prefix just before the second copy of the loop, despite unequal prompts.
    compact_logits = torch.arange(4 * 10 * 5).reshape(4, 10, 5).float()
    boundary = compact_boundary_logits(compact_logits, batch["attention_mask"], batch["ftpo_chosen_mask"])
    torch.testing.assert_close(boundary[0], compact_logits[0, 3])
    batch["ftpo_reference_logits"] = boundary
    for key in ("action_log_probs", "base_action_log_probs", "values", "returns", "advantages"):
        batch[key] = torch.zeros_like(batch["loss_mask"])
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    restored = torch.load(path, weights_only=False)
    experiences = list(TrainingBatchIterator(restored, sample_batch_size=1))
    assert [bool(experience.ftpo.chosen_mask.any()) for experience in experiences] == [True, False, False, False]
    torch.testing.assert_close(experiences[0].ftpo.reference_logits, boundary[:1])
    assert experiences[0].ftpo.candidate_ids[experiences[0].ftpo.chosen_mask].tolist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_ftpo_empty_rollout_batch_skips_training(config):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.all_timings = {}
    trainer.all_metrics = {}
    # No actors exist: an empty batch must not dispatch a forward or optimizer step.
    status = await trainer._run_training(
        {
            "ftpo_chosen_mask": torch.zeros((2, 3, 2), dtype=torch.bool),
            "loss_mask": torch.zeros((2, 3)),
        }
    )
    assert status["policy_update_steps"] == 0
    assert trainer.all_metrics["ftpo/empty_batch"] == 1
