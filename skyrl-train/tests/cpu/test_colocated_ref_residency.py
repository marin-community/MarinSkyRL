import pytest

from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import example_dummy_config


def _single_gpu_config():
    cfg = example_dummy_config()
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.num_inference_engines = 1
    return cfg


@pytest.mark.parametrize(
    ("colocate_all", "colocate_policy_ref"),
    [(False, True), (True, False)],
)
def test_fsdp2_colocated_reference_uses_persistent_cpu_offload(colocate_all, colocate_policy_ref):
    cfg = _single_gpu_config()
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.trainer.placement.colocate_policy_ref = colocate_policy_ref
    cfg.trainer.ref.fsdp_config.cpu_offload = False
    cfg.trainer.policy.fsdp_config.cpu_offload = False

    validate_cfg(cfg)

    assert cfg.trainer.ref.fsdp_config.cpu_offload is True
    assert cfg.trainer.policy.fsdp_config.cpu_offload is False


@pytest.mark.parametrize(
    ("strategy", "colocate_policy_ref"),
    [("fsdp2", False), ("megatron", True)],
)
def test_non_colocated_or_non_fsdp2_reference_residency_is_unchanged(strategy, colocate_policy_ref):
    cfg = _single_gpu_config()
    cfg.trainer.strategy = strategy
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = colocate_policy_ref
    cfg.trainer.ref.fsdp_config.cpu_offload = False

    validate_cfg(cfg)

    assert cfg.trainer.ref.fsdp_config.cpu_offload is False
