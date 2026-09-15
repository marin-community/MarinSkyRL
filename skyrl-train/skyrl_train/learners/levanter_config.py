"""Cheap configuration lowering for the Levanter Snowball learner."""

from __future__ import annotations

import re
from dataclasses import dataclass

from omegaconf import DictConfig

from skyrl_train.learner import UnsupportedLearnerConfiguration


@dataclass(frozen=True)
class LevanterSnowballRuntimeConfig:
    """Runtime settings that are fixed before the learner reserves a GPU."""

    model_path: str
    seed: int
    training_nodes: int
    training_gpus_per_node: int
    training_gpus: int
    inference_world_size: int
    train_batch_size: int
    micro_train_batch_size_per_gpu: int
    micro_forward_batch_size_per_gpu: int
    num_train_steps: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float
    parameter_dtype: str
    compute_dtype: str
    output_dtype: str
    attention_implementation: str
    moe_implementation: str
    publication_backend: str
    publication_max_chunk_bytes: int
    publication_timeout_seconds: float
    generator_dtype: str
    require_accelerator: bool
    log_dir: str
    model_revision: str | None = None
    offload_opt_state: bool = False
    publication_scatter_experts: bool = False
    initial_weights_already_loaded: bool = False
    model_source_identity: str | None = None

    @classmethod
    def from_msrl(cls, cfg: DictConfig) -> LevanterSnowballRuntimeConfig:
        """Validate the one supported workload without importing JAX or Torch."""

        trainer = cfg.trainer
        algorithm = trainer.algorithm
        policy = trainer.policy
        placement = trainer.placement
        generator = cfg.generator
        levanter = policy.levanter
        optimizer = policy.optimizer_config
        offpolicy_mask = algorithm.offpolicy_mask
        regular_mask_enabled = bool(offpolicy_mask.enabled)
        expected_learning_rate = 1e-6 if regular_mask_enabled else 1e-5
        expected_learning_rate_text = "1e-6" if regular_mask_enabled else "1e-5"
        expected_max_grad_norm = 1.0 if regular_mask_enabled else 0.5
        initial_weights_already_loaded = bool(levanter.get("initial_weights_already_loaded", False))
        publication_scatter_experts = bool(levanter.get("publication_scatter_experts", False))
        model_revision = str(policy.model.revision) if policy.model.get("revision") else None
        model_source_identity = str(policy.model.source_identity) if policy.model.get("source_identity") else None
        immutable_model_identity = model_source_identity
        if (
            immutable_model_identity is None
            and model_revision is not None
            and re.fullmatch(r"[0-9a-f]{40}", model_revision)
        ):
            immutable_model_identity = f"hf-revision:{model_revision}"

        unsupported: list[str] = []
        expected_values = (
            (trainer.strategy == "fsdp2", "trainer.strategy=fsdp2"),
            (not placement.colocate_all, "separate learner and inference GPUs"),
            (generator.run_engines_locally, "local inference engines"),
            (generator.backend == "vllm", "generator.backend=vllm"),
            (generator.async_engine, "generator.async_engine=true"),
            (not generator.fuse_weights, "generator.fuse_weights=false"),
            (not trainer.use_sample_packing, "trainer.use_sample_packing=false"),
            (not trainer.step_wise_training, "trainer.step_wise_training=false"),
            (not algorithm.use_kl_loss and not algorithm.use_kl_in_reward, "no KL or reference policy"),
            (not algorithm.use_entropy_loss, "trainer.algorithm.use_entropy_loss=false"),
            (not algorithm.use_tis, "trainer.algorithm.use_tis=false"),
            (algorithm.advantage_estimator == "grpo", "trainer.algorithm.advantage_estimator=grpo"),
            (
                int(algorithm.resolved_group_advantage.physical_group_size) == int(generator.n_samples_per_prompt),
                "a resolved GRPO physical group size matching generator.n_samples_per_prompt",
            ),
            (algorithm.policy_loss_type == "regular", "trainer.algorithm.policy_loss_type=regular"),
            (algorithm.loss_reduction == "token_mean", "trainer.algorithm.loss_reduction=token_mean"),
            (
                not regular_mask_enabled or bool(algorithm.require_rollout_logprobs),
                "strict rollout log probabilities for regular_mask",
            ),
            (
                not regular_mask_enabled or offpolicy_mask.ratio == "mismatch",
                "trainer.algorithm.offpolicy_mask.ratio=mismatch",
            ),
            (
                not regular_mask_enabled
                or (
                    float(offpolicy_mask.low) == 0.5
                    and float(offpolicy_mask.high) == 5.0
                    and float(offpolicy_mask.veto_ratio) == 1.0e-5
                ),
                "regular_mask bounds low=0.5, high=5.0, veto_ratio=1e-5",
            ),
            (
                not regular_mask_enabled or not bool(offpolicy_mask.renormalize),
                "regular_mask with its original loss denominator",
            ),
            (float(algorithm.eps_clip_low) == 0.2, "trainer.algorithm.eps_clip_low=0.2"),
            (float(algorithm.eps_clip_high) == 0.2, "trainer.algorithm.eps_clip_high=0.2"),
            (bool(algorithm.grpo_norm_by_std), "trainer.algorithm.grpo_norm_by_std=true"),
            (not algorithm.advantage_batch_normalize, "trainer.algorithm.advantage_batch_normalize=false"),
            (trainer.update_epochs_per_batch == 1, "trainer.update_epochs_per_batch=1"),
            (bool(trainer.restore_dataloader_state), "trainer.restore_dataloader_state=true"),
            (policy.grug_query_bias_update_mode == "frozen", "trainer.policy.grug_query_bias_update_mode=frozen"),
            (int(policy.model.lora.rank) == 0, "trainer.policy.model.lora.rank=0"),
            (policy.fsdp_config.expert_model_parallel_size == 1, "expert parallel size 1"),
            (optimizer.optimizer == "AdamW", "trainer.policy.optimizer_config.optimizer=AdamW"),
            (optimizer.scheduler == "constant_with_warmup", "a constant-with-warmup optimizer schedule"),
            (optimizer.num_warmup_steps == 0, "zero optimizer warmup steps"),
            (
                float(optimizer.lr) == expected_learning_rate,
                f"trainer.policy.optimizer_config.lr={expected_learning_rate_text}",
            ),
            (list(optimizer.adam_betas) == [0.9, 0.999], "trainer.policy.optimizer_config.adam_betas=[0.9,0.999]"),
            (float(optimizer.weight_decay) == 0.01, "trainer.policy.optimizer_config.weight_decay=0.01"),
            (
                float(optimizer.max_grad_norm) == expected_max_grad_norm,
                f"trainer.policy.optimizer_config.max_grad_norm={expected_max_grad_norm:g}",
            ),
            (levanter.publication_backend == "gloo", "trainer.policy.levanter.publication_backend=gloo"),
            (generator.weight_sync_backend == "gloo", "generator.weight_sync_backend=gloo"),
            (int(generator.inference_engine_tensor_parallel_size) == 1, "inference tensor parallel size 1"),
            (int(generator.inference_engine_pipeline_parallel_size) == 1, "inference pipeline parallel size 1"),
            (float(generator.sampling_params.temperature) == 1.0, "generator.sampling_params.temperature=1.0"),
            (
                levanter.parameter_dtype in {"float32", "bfloat16"},
                "Levanter FP32 or BF16 parameter storage",
            ),
            (levanter.compute_dtype == "bfloat16", "Levanter BF16 compute"),
            (levanter.output_dtype == "float32", "Levanter FP32 outputs"),
            (generator.model_dtype == "bfloat16", "generator.model_dtype=bfloat16"),
            (
                not initial_weights_already_loaded or generator.model_name == policy.model.path,
                "the same initial model path for Levanter and vLLM",
            ),
            (
                not initial_weights_already_loaded or immutable_model_identity is not None,
                "an immutable model source identity when adopting already-loaded initial weights",
            ),
            (
                not initial_weights_already_loaded or trainer.resume_mode == "none",
                "resume_mode=none when adopting already-loaded initial weights",
            ),
            (
                not publication_scatter_experts
                or (
                    int(generator.num_inference_engines) == 1
                    and int(generator.inference_engine_expert_parallel_size)
                    == int(generator.inference_engine_data_parallel_size)
                ),
                "expert scatter across one full inference data-parallel group",
            ),
        )
        unsupported.extend(description for accepted, description in expected_values if not accepted)

        training_nodes = int(placement.policy_num_nodes)
        training_gpus_per_node = int(placement.policy_num_gpus_per_node)
        training_gpus = training_nodes * training_gpus_per_node
        prompt_batch_size = int(trainer.train_batch_size)
        train_batch_size = prompt_batch_size * int(generator.n_samples_per_prompt)
        micro_train = int(trainer.micro_train_batch_size_per_gpu)
        micro_forward = int(trainer.micro_forward_batch_size_per_gpu)
        if training_nodes < 1:
            unsupported.append("at least one policy node")
        if training_gpus_per_node < 1:
            unsupported.append("at least one policy GPU per node")
        if prompt_batch_size != int(trainer.policy_mini_batch_size):
            unsupported.append("train_batch_size equal to policy_mini_batch_size")
        if micro_train != 1:
            unsupported.append("trainer.micro_train_batch_size_per_gpu=1")
        if micro_forward != 1:
            unsupported.append("trainer.micro_forward_batch_size_per_gpu=1")
        if train_batch_size % training_gpus:
            unsupported.append("a generated trajectory batch divisible by policy GPUs")
        if int(levanter.publication_max_chunk_bytes) < 1:
            unsupported.append("a positive Levanter publication chunk size")
        if float(levanter.publication_timeout_seconds) <= 0:
            unsupported.append("a positive Levanter publication timeout")
        if str(levanter.attention_implementation) not in {"reference", "gpu_fa4_cute"}:
            unsupported.append("Levanter reference or gpu_fa4_cute attention")
        if str(levanter.moe_implementation) != "ring":
            unsupported.append("Levanter ring MoE dispatch")
        if unsupported:
            raise UnsupportedLearnerConfiguration(
                "the Levanter Snowball learner currently requires " + ", ".join(unsupported)
            )

        optimizer_kwargs = dict(optimizer.optimizer_kwargs)
        unknown_optimizer_kwargs = set(optimizer_kwargs).difference({"eps"})
        if unknown_optimizer_kwargs:
            raise UnsupportedLearnerConfiguration(
                "the Levanter Snowball learner does not support AdamW optimizer kwargs "
                + ", ".join(sorted(unknown_optimizer_kwargs))
            )
        if float(optimizer_kwargs.get("eps", 1e-8)) != 1e-8:
            raise UnsupportedLearnerConfiguration(
                "the Levanter Snowball learner currently requires trainer.policy.optimizer_config.optimizer_kwargs.eps=1e-8"
            )

        return cls(
            model_path=str(policy.model.path),
            seed=int(trainer.seed),
            training_nodes=training_nodes,
            training_gpus_per_node=training_gpus_per_node,
            training_gpus=training_gpus,
            inference_world_size=(
                int(generator.num_inference_engines)
                * int(generator.inference_engine_tensor_parallel_size)
                * int(generator.inference_engine_pipeline_parallel_size)
                * int(generator.inference_engine_data_parallel_size)
            ),
            train_batch_size=train_batch_size,
            micro_train_batch_size_per_gpu=micro_train,
            micro_forward_batch_size_per_gpu=micro_forward,
            num_train_steps=int(trainer.max_steps),
            learning_rate=float(optimizer.lr),
            adam_beta1=float(optimizer.adam_betas[0]),
            adam_beta2=float(optimizer.adam_betas[1]),
            adam_epsilon=float(optimizer_kwargs.get("eps", 1e-8)),
            weight_decay=float(optimizer.weight_decay),
            max_grad_norm=float(optimizer.max_grad_norm),
            parameter_dtype=str(levanter.parameter_dtype),
            compute_dtype=str(levanter.compute_dtype),
            output_dtype=str(levanter.output_dtype),
            attention_implementation=str(levanter.attention_implementation),
            moe_implementation=str(levanter.moe_implementation),
            publication_backend=str(levanter.publication_backend),
            publication_max_chunk_bytes=int(levanter.publication_max_chunk_bytes),
            publication_timeout_seconds=float(levanter.publication_timeout_seconds),
            generator_dtype=str(generator.model_dtype),
            require_accelerator=bool(levanter.require_accelerator),
            log_dir=str(levanter.log_dir),
            model_revision=model_revision,
            offload_opt_state=bool(levanter.get("offload_opt_state", False)),
            publication_scatter_experts=publication_scatter_experts,
            initial_weights_already_loaded=initial_weights_already_loaded,
            model_source_identity=immutable_model_identity,
        )
