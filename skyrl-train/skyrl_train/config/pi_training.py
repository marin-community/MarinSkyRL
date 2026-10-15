"""Fail-closed Pi preset, installed-agent and training-objective binding."""

from omegaconf import DictConfig, OmegaConf


def validate_pi_objective_config(cfg: DictConfig) -> None:
    """Validate a reviewed Pi preset against its mode and training objective."""
    scenario = OmegaConf.select(cfg, "scenario")
    if scenario is None:
        if OmegaConf.select(cfg, "terminal_bench_config.installed_agent.agent.pi.mode") == "teacher_trace":
            raise ValueError("Pi teacher trace requires a reviewed scenario.preset")
        return
    if not isinstance(scenario, DictConfig) or set(scenario) != {"preset"}:
        raise ValueError("Pi training requires exactly scenario.preset")
    if not OmegaConf.select(cfg, "data.terminal_bench_data"):
        raise ValueError("Pi training requires Harbor task data in data.terminal_bench_data")
    preset = scenario.preset
    modes = {"verifier_grpo_base": "literal", "teacher_demo_pi_base": "teacher_trace"}
    if preset not in modes:
        raise ValueError(f"Pi training preset {preset!r} is not wired to a production objective")
    agent = OmegaConf.select(cfg, "terminal_bench_config.installed_agent.agent")
    if agent is None or agent.get("name") != "pi" or OmegaConf.select(agent, "pi.mode") != modes[preset]:
        raise ValueError(f"{preset} requires an installed Pi agent with pi.mode={modes[preset]}")
    if cfg.trainer.step_wise_training:
        raise ValueError("Pi scenarios require whole-trial training, not step-wise training")
    algorithm = cfg.trainer.algorithm
    if preset == "verifier_grpo_base":
        if algorithm.advantage_estimator != "grpo" or algorithm.policy_loss_type != "regular":
            raise ValueError("verifier_grpo_base requires GRPO advantages and regular clipped policy loss")
        if cfg.generator.n_samples_per_prompt < 2:
            raise ValueError("verifier_grpo_base requires at least two physical trials per prompt")
    elif (
        algorithm.policy_loss_type != "sft"
        or algorithm.advantage_estimator != "uniform"
        or cfg.generator.n_samples_per_prompt != 1
    ):
        raise ValueError("teacher_demo_pi_base requires SFT, uniform advantages and one trial per prompt")
    if algorithm.dynamic_sampling.type not in (None, "none"):
        raise ValueError("Pi scenarios do not support dynamic reward-based sampling")


def validate_pi_training_config(cfg: DictConfig) -> None:
    """Reject unavailable sources before allocating rollout or policy workers."""
    validate_pi_objective_config(cfg)
    if OmegaConf.select(cfg, "scenario.preset") != "teacher_demo_pi_base":
        return
    from skyrl_train.rollouts.pi_teacher_binding import PiTeacherBinding

    binding = OmegaConf.select(cfg, "terminal_bench_config.installed_agent.teacher")
    if binding is None:
        raise ValueError("Pi teacher trace needs a distinct teacher model source")
    teacher = PiTeacherBinding.model_validate(OmegaConf.to_container(binding, resolve=True))
    if (
        OmegaConf.select(cfg, "terminal_bench_config.installed_agent.agent.model_name")
        != f"hosted_vllm/{teacher.model}"
    ):
        raise ValueError("Pi teacher agent must address the bound teacher alias")
    if teacher.student_model != cfg.trainer.policy.model.path:
        raise ValueError("Pi teacher binding must identify the configured student policy")
    if cfg.trainer.evaluation_runner != "harbor" or cfg.trainer.eval_interval <= 0 or not cfg.data.val_data:
        raise ValueError("Pi teacher training requires scheduled student-only Harbor evaluation data")
    if teacher.source == "skyrl_managed" and teacher.resources is None:
        raise ValueError("SkyRL-managed Pi teacher requires independent GPU resource allocation")
