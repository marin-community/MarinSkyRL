"""Selected behavior-head IDs must follow Megatron's left-padding compaction."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.losses import ppo_policy_loss

from skyrl_train.distributed.dispatch import ActorInfo, MeshRank, concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingOutputBatch
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402


@pytest.fixture
def megatron_wrapper(monkeypatch, single_rank_group):
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_first_stage", lambda **kwargs: True, raising=False)
    wrapper = object.__new__(mmw.MegatronModelWrapper)
    wrapper.actor_module = [SimpleNamespace(training=True)]
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = 2
    return wrapper


def _model_logits(sequences):
    return torch.nn.functional.one_hot(sequences, num_classes=7).float() * 2 + (
        sequences[:, :1].float().unsqueeze(-1) * torch.arange(7) * 0.1
    )


class _MegatronModel:
    training = True

    def __call__(self, sequences, *args, **kwargs):
        return _model_logits(sequences)


@pytest.mark.parametrize("micro_size", [1, 2])
@pytest.mark.parametrize("capture_width", [0, 2])
def test_pipeline_outputs_construct_before_collection_and_preserve_rows(
    monkeypatch, megatron_wrapper, micro_size, capture_width
):
    # Only the external Megatron scheduler/model and rank predicates are faked.
    # Real scoring, output-container checks and mesh collection run on CPU.
    def schedule(forward_step_func, data_iterator, model, num_microbatches, **kwargs):
        if not mmw.mpu.is_pipeline_last_stage(ignore_virtual=True):
            return []
        outputs = []
        for _ in range(num_microbatches):
            logits, collect = forward_step_func(data_iterator, model[0])
            _, result = collect(logits)
            outputs.append(result)
        return outputs

    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: schedule)
    megatron_wrapper.actor_module = [_MegatronModel()]
    actors, outputs, expected_scores, expected_candidates = [], [], [], []
    for dp in (1, 0):
        sequences = torch.tensor([[2 * dp + 1, 2, 3, 4], [2 * dp + 2, 3, 4, 5]])
        ids = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [1, 6]]]) if capture_width else None
        micros = [
            mmw.MegatronForwardMicroBatch(
                sequences=sequences[start : start + micro_size],
                attention_mask=torch.ones((micro_size, 4), dtype=torch.bool),
                position_ids=torch.arange(4).expand(micro_size, 4),
                num_actions=2,
                score_topk_indices=None if ids is None else ids[start : start + micro_size],
            )
            for start in range(0, 2, micro_size)
        ]
        for pp in (0, 1):
            monkeypatch.setattr(
                mmw.mpu, "is_pipeline_last_stage", lambda pp_rank=pp, **kwargs: pp_rank == 1, raising=False
            )
            result = megatron_wrapper.forward(micros, seq_len=4, micro_batch_size=micro_size)
            output = TrainingOutputBatch({"output": result.scores})
            if result.selected_logprobs is not None:
                output["score_old_logprobs"] = result.selected_logprobs
            outputs.append(output)
            actors.append(ActorInfo(handle=None, rank=MeshRank(dp, 0, 0, pp, 4, 2, 2)))
        normalized = _model_logits(sequences).log_softmax(-1)[:, 1:3]
        expected_scores.insert(0, normalized.gather(-1, sequences[:, -2:].unsqueeze(-1)).squeeze(-1))
        if ids is not None:
            expected_candidates.insert(0, normalized.gather(-1, ids))
    collected = concatenate_outputs_after_mesh_dispatch(actors, outputs)
    torch.testing.assert_close(collected["output"], torch.cat(expected_scores))
    if capture_width:
        torch.testing.assert_close(collected["score_old_logprobs"], torch.cat(expected_candidates))
    else:
        assert "score_old_logprobs" not in collected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_response_reuses_sampled_normalizer_without_full_float_logits(dtype, megatron_wrapper):
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
    sequences = torch.tensor([[0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 0, 0]])
    selected_ids = torch.tensor([[[1, 3], [2, 4], [0, 5]], [[6, 1], [-1, -1], [-1, -1]]])
    sampled_ids = torch.tensor([[3, 4, 5], [2, 0, 0]])
    torch.manual_seed(91)
    base_logits = torch.randn((2, 6, 7), dtype=dtype, requires_grad=True)
    logits = base_logits / 0.7
    normalized = logits.float().log_softmax(dim=-1)
    compact_positions = torch.tensor([[2, 3, 4], [1, 0, 0]])
    batch_positions = torch.arange(2)[:, None]
    expected_chosen = normalized[batch_positions, compact_positions, sampled_ids]

    saved_tensors = []

    def save(tensor):
        saved_tensors.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        response = megatron_wrapper._response_logprobs(logits, sequences, attention_mask, 3, None, selected_ids)
    selected = response.selected_logprobs

    valid = selected_ids >= 0
    expected = normalized[torch.arange(2)[:, None, None], compact_positions[:, :, None], selected_ids.clamp_min(0)]
    torch.testing.assert_close(selected[valid], expected[valid], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(response.scores[valid.any(-1)], expected_chosen[valid.any(-1)], atol=1e-5, rtol=1e-5)
    assert torch.isnan(selected[~valid]).all()
    if dtype == torch.bfloat16:
        assert (logits.shape, torch.float32) not in saved_tensors

    selected_gradient = torch.autograd.grad(selected[valid].sum(), base_logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected[valid].sum(), base_logits)[0]
    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(selected_gradient, expected_gradient, atol=tolerance, rtol=tolerance)


def test_training_scores_and_centers_constant_advantage_without_masked_nan_gradients(
    monkeypatch, megatron_wrapper, single_rank_group
):
    # Enumerate all sampled actions under q using policy-data weights. A full
    # head must cancel their expected score gradient, for both advantage signs.
    q = torch.tensor([0.30, 0.05, 0.25, 0.10, 0.10, 0.10, 0.10])
    old = torch.tensor([1e-15, 0.20, 0.25, 0.15, 0.15, 0.10, 0.15])
    sequences = torch.tensor([[1, 2, action, action] for action in range(7)] + [[1, 2, 0, 0]])
    mask = torch.cat((q[:, None].expand(7, 2), torch.zeros(1, 2)))
    advantages = torch.tensor([1.7, -2.3]).expand(8, 2).clone()
    advantages[-1] = torch.nan
    candidates = torch.arange(7).expand(8, 2, 7).clone()
    candidates[-1] = -1
    old_head, behavior_head = old.log().expand(8, 2, 7).clone(), q.log().expand(8, 2, 7).clone()
    old_head[-1], behavior_head[-1] = torch.nan, torch.nan
    old_chosen = old.log()[sequences[:, -2:]]
    behavior_chosen = q.log()[sequences[:, -2:]]
    torch.manual_seed(93)
    parameters = torch.randn(4, 7, requires_grad=True)

    def model(sequences, *args, **kwargs):
        return parameters.expand(len(sequences), 4, 7).clone()

    model.training = True

    def schedule(forward_step_func, data_iterator, model, **kwargs):
        logits, closure = forward_step_func(data_iterator, model[0])
        loss, metrics = closure(logits)
        loss.backward()
        return [metrics]

    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: schedule)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_data_parallel_group", lambda **kwargs: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_last_rank", lambda: 0, raising=False)
    megatron_wrapper.actor_module = [model]
    megatron_wrapper.policy_loss_fn = ppo_policy_loss
    megatron_wrapper.cfg = OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "score_centering_topk": 7,
                    "max_seq_len": 4,
                    "think_token_weight": 1,
                    "off_policy_correction": "custom",
                    "off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": 1.05}],
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "loss_reduction": "token_mean",
                    "use_entropy_loss": False,
                    "use_kl_loss": False,
                    "kl_loss_coef": 0,
                }
            }
        }
    )
    micro = mmw.MegatronPolicyMicroBatch(
        sequences=sequences,
        attention_mask=torch.ones_like(sequences),
        position_ids=torch.arange(4).expand(8, 4),
        num_actions=2,
        old_action_log_probs=old_chosen,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=mask,
        rollout_action_logprobs=behavior_chosen,
        response_span_tags=None,
        correction_weights=(old_chosen - behavior_chosen).exp().clamp(max=1.05),
        score_topk_indices=candidates,
        score_old_logprobs=old_head,
        score_behavior_logprobs=behavior_head,
    )
    # Exercise the actual CPU math without requiring a native compiler for telemetry.
    with torch.compiler.set_stance("force_eager"):
        megatron_wrapper.forward_backward_mini_batch([micro], seq_len=4, micro_batch_size=8, temperature=0.7)
    torch.testing.assert_close(parameters.grad, torch.zeros_like(parameters), atol=1e-7, rtol=0)
