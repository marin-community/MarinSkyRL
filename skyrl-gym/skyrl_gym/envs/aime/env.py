from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.aime import utils
from typing import Dict, Any, Optional
from omegaconf import DictConfig


class AIMEEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.

    Supports an optional, tunable LENGTH-PENALTY reward (cosine length-scaled,
    gated on correctness). All length-penalty knobs are read from `env_config`
    (i.e. hydra `environment.skyrl_gym.aime.*`). The single grid axis is
    `length_penalty_weight`; with weight=0 (the default) the reward is byte-for-byte
    the legacy +1.0/-1.0 reward, so it is a clean A/B superset.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = extras["reward_model"]["ground_truth"]

        # ---- Tunable length-penalty config (hydra: environment.skyrl_gym.aime.*) ----
        # weight=0.0 -> legacy reward (backward compatible).
        self.length_penalty_weight: float = float(env_config.get("length_penalty_weight", 0.0))
        self.target_length: int = int(env_config.get("target_length", 0))
        self.truncated_penalty: float = float(env_config.get("truncated_penalty", -2.0))
        self.min_response_length: int = int(env_config.get("min_response_length", 16))
        self.end_think_token: str = str(env_config.get("end_think_token", "<|end_think|>"))
        # max_gen_length: the generation cap (token budget). 0 -> length shaping
        # of correct answers is skipped (falls back to +1.0). The generator sets
        # this at step time via set_generation_metadata() so it stays in sync with
        # generator.sampling_params.max_generate_length, but a config override is
        # honored if provided.
        self.max_gen_length: int = int(env_config.get("max_gen_length", 0))

        # ---- Generation metadata, populated by the generator before step() ----
        self._response_length: Optional[int] = None
        self._stop_reason: Optional[str] = None

    def set_generation_metadata(
        self,
        response_length: Optional[int] = None,
        stop_reason: Optional[str] = None,
        max_gen_length: Optional[int] = None,
    ) -> None:
        """Optional hook called by the generator just before `step()` to expose
        generation-time signals (token length, stop_reason, the generation cap)
        to the reward function. Safe no-op contract: envs that don't define this
        are simply not called; this env tolerates None for every field and falls
        back to the legacy reward path.
        """
        self._response_length = response_length
        self._stop_reason = stop_reason
        if max_gen_length is not None and max_gen_length > 0:
            self.max_gen_length = max_gen_length

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step

        score_info = utils.compute_score(
            action,
            self.ground_truth,
            response_length=self._response_length,
            stop_reason=self._stop_reason,
            length_penalty_weight=self.length_penalty_weight,
            target_length=self.target_length,
            max_gen_length=self.max_gen_length,
            truncated_penalty=self.truncated_penalty,
            min_response_length=self.min_response_length,
            end_think_token=self.end_think_token,
        )
        reward = score_info["score"]
        metadata = {
            "acc": score_info["acc"],
            "pred": score_info["pred"],
            "truncated": score_info.get("truncated"),
            "length_shaped": score_info.get("length_shaped"),
            "response_length": score_info.get("response_length"),
            "length_frac": score_info.get("length_frac"),
        }

        # No observation in aime, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
