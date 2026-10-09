"""
Main entrypoint for the LLM-as-a-judge example.
"""

import ray
import hydra
from functools import partial
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.task_sessions import AnswerTaskSession
from examples.llm_as_a_judge.task_grading import grade_judged_answer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = BasePPOExp(cfg, sessions={"llm_as_a_judge": partial(AnswerTaskSession, grader=grade_judged_answer)})
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
