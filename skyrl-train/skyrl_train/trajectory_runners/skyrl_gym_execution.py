"""Construction of SkyRL-Gym trajectory runners inside rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from skyrl_gym.envs.registration import EnvSpec, registry
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.workers import WorkerShard, detached_config
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner, TrajectoryPipeline
from skyrl_train.trajectory_runners.step_wise import StepWiseRolloutCollector


@dataclass(frozen=True)
class GymRunnerSpec:
    """Serializable inputs that build one rollout worker's SkyRL-Gym runner, with its own client for the engines."""

    config: DictConfig
    engines: list[InferenceEngineInterface]
    # The trainer process's environment registry. An entrypoint may register environments at runtime, and a worker
    # process otherwise knows only the ones its imports register.
    environments: dict[str, EnvSpec]

    @classmethod
    def from_config(cls, config: DictConfig, engines: Sequence[InferenceEngineInterface]) -> GymRunnerSpec:
        return cls(
            config=detached_config(config),
            engines=list(engines),
            environments=dict(registry),
        )

    def build(self, tokenizer: PreTrainedTokenizerBase, shard: WorkerShard) -> TrajectoryRunner:
        """Construct this worker's runner, collecting step-wise trajectories when step-wise training is enabled."""
        del shard
        for env_id, env_spec in self.environments.items():
            registry.setdefault(env_id, env_spec)
        # Only the trainer's client serves the HTTP endpoint.
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        pipeline = None
        if self.config.trainer.step_wise_training:
            pipeline = TrajectoryPipeline(
                StepWiseRolloutCollector,
                StepWiseTrajectoryProjection(self.config.generator, tokenizer),
            )
        return SkyRLGymTrajectoryRunner(
            trajectory_runner_cfg=self.config.generator,
            skyrl_gym_cfg=self.config.environment.skyrl_gym,
            inference_engine_client=InferenceEngineClient(self.engines, tokenizer, client_config),
            tokenizer=tokenizer,
            pipeline=pipeline,
        )
