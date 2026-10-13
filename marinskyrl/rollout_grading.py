"""Config rules for training without per-rollout verification."""

from omegaconf import DictConfig, OmegaConf
from skyrl_gym.source_task import session_parameters
from taskcompendium.models import SessionGrader, TaskSpec

from marinskyrl.distillation import DistillationPlan, DistillationRewardMode
from marinskyrl.runtime_options import TRAJECTORY_SELECTOR_TYPE_PATH, AdvantageEstimator, RolloutGrading


def validate_task_grading(task: TaskSpec, cfg: DictConfig, distillation_plan: DistillationPlan | None) -> None:
    """Reject skipped task grading when training or evaluation uses the reward."""
    if not isinstance(task.grader, SessionGrader):
        return
    grading = RolloutGrading(session_parameters(task).config.get("grading", RolloutGrading.VERIFY))
    if grading is RolloutGrading.VERIFY:
        return
    prefix = f"Task {task.id!r} with skipped grading requires"
    if distillation_plan is None or distillation_plan.reward_mode is not DistillationRewardMode.REPLACE:
        raise ValueError(f"{prefix} trainer.algorithm.distillation.reward_mode=replace")
    if cfg.trainer.algorithm.advantage_estimator != AdvantageEstimator.UNIFORM:
        raise ValueError(
            f"{prefix} trainer.algorithm.advantage_estimator=uniform; got {cfg.trainer.algorithm.advantage_estimator}"
        )
    if cfg.trainer.algorithm.dynamic_sampling.type is not None:
        raise ValueError(f"{prefix} trainer.algorithm.dynamic_sampling.type=null")
    if OmegaConf.select(cfg, TRAJECTORY_SELECTOR_TYPE_PATH, default=None) is not None:
        raise ValueError(f"{prefix} no trajectory selector")
    if cfg.trainer.eval_before_train or cfg.trainer.eval_interval > 0:
        raise ValueError(f"{prefix} trainer.eval_before_train=false and trainer.eval_interval<=0")
