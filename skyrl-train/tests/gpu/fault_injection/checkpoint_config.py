from tests.gpu.gpu_ci.test_trainer_full_checkpointing import get_test_trainer_config


CHECKPOINT_S3_PREFIX = "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def checkpoint_config(root: str, *, resume: bool = False):
    cfg = get_test_trainer_config("megatron", optimizer_checkpoint_sharding_type="dp_reshardable")
    cfg.trainer.policy.model.revision = MODEL_REVISION
    cfg.trainer.ckpt_path = f"{root}/checkpoints"
    cfg.trainer.export_path = f"{root}/exports"
    cfg.trainer.max_ckpts_to_keep = -1
    cfg.trainer.policy.megatron_config.checkpoint_plan_cache = True
    cfg.trainer.resume_mode = "latest" if resume else "none"
    return cfg
