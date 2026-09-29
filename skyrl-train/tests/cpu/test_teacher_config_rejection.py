from hydra import compose, initialize_config_dir
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.objective.losses import PolicyLossInputs
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.config.objective_spec import TopKLossParams
from skyrl_train.distillation import StudentTopKInput
from skyrl_train.objective.teacher import topk_teacher_loss


def test_packaged_entrypoints_reject_ad_hoc_teacher_configuration():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+teacher.model_path=Qwen/Qwen3-4B"])

    with pytest.raises(ValueError, match="legacy teacher configuration is not supported"):
        validate_cfg(cfg)


def test_packaged_entrypoints_accept_distillation_only_replace_mode():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                    }
                }
            },
            "teachers": {
                "primary": {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "endpoints": [{"url": "https://teacher.example/v1", "max_concurrency": 8}],
                    "tokenizer_fingerprint": f"sha256:{'a' * 64}",
                    "max_sequence_length": 32768,
                    "request_timeout_seconds": 120,
                    "evidence": "chosen_token",
                }
            },
            "teacher_routing": {
                "opd": {
                    "revision": "route-revision",
                    "routes": {"default": {"teacher": "primary", "weight": 1.0}},
                }
            },
        },
    )
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.advantage_estimator = "uniform"

    validate_cfg(cfg)

    current = torch.tensor([[-1.0]], requires_grad=True)
    teacher_advantage = torch.tensor([[0.5]])
    values = PolicyLossRegistry.get(cfg.trainer.algorithm.policy_loss_type)(
        PolicyLossInputs(current, current.detach(), None, teacher_advantage, torch.ones_like(current)),
        cfg.trainer.algorithm,
    ).values
    values.sum().backward()
    torch.testing.assert_close(values, torch.tensor([[-0.5]]))
    torch.testing.assert_close(current.grad, torch.tensor([[-0.5]]))


def selected_topk_config() -> DictConfig:
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "student_topk_policy_surrogate",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                    }
                }
            },
            "teachers": {
                "primary": {
                    "source": "local_inference",
                    "placement": "pinned",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "backend": "vllm",
                    "evidence": "student_selected_topk",
                    "top_k": 16,
                    "resources": {
                        "num_nodes": 1,
                        "gpus_per_node": 1,
                        "tensor_parallel_size": 1,
                        "colocation_group": "teacher",
                    },
                }
            },
            "teacher_routing": {
                "opd": {"revision": "route-revision", "routes": {"default": {"teacher": "primary", "weight": 1.0}}}
            },
        },
    )
    cfg.trainer.logger = "console"
    cfg.generator.sampling_params.logprobs = 16
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.use_sample_packing = False
    return cfg


def test_selected_topk_rollouts_require_matching_teacher_width():
    cfg = selected_topk_config()

    validate_cfg(cfg)

    transported = OmegaConf.create(OmegaConf.to_yaml(cfg))
    current = torch.full((1, 1, 16), 1 / 32).log().requires_grad_()
    evidence = StudentTopKInput(
        torch.arange(16).reshape(1, 1, 16),
        current.detach(),
        torch.full_like(current, 1 / 16).log(),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1),
    )
    result = topk_teacher_loss(
        evidence, current, TopKLossParams.from_config(transported.trainer.algorithm.resolved_topk_loss_params)
    )
    result.values.sum().backward()
    expected = -torch.tensor(2.0).log()
    torch.testing.assert_close(result.values, expected.reshape(1, 1))
    torch.testing.assert_close(current.grad, torch.full_like(current, expected / 16))

    cfg.generator.sampling_params.logprobs = 8
    with pytest.raises(ValueError, match="matching teacher top_k"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("trainer.use_sample_packing", True),
        ("trainer.policy.sequence_parallel_size", 2),
        ("trainer.policy.megatron_config.context_parallel_size", 2),
        ("trainer.policy.megatron_config.tensor_model_parallel_size", 2),
    ],
)
def test_selected_topk_rejects_unsupported_policy_geometry_before_allocation(path, value):
    cfg = selected_topk_config()
    OmegaConf.update(cfg, path, value)

    with pytest.raises(ValueError, match="top-K teacher objectives require use_sample_packing=false"):
        validate_cfg(cfg)
