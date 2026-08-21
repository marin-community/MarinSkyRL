import asyncio
from typing import Dict, Optional, Any
from omegaconf import DictConfig
import yaml
import traceback
import ray
from pathlib import Path
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj
from minisweagent.config import get_config_path
from .environment import evaluate_trajectory, get_sb_environment

from skyrl_train.trajectory_runners.base import (
    BatchMetadata,
    TrajectoryBatch,
    TrajectoryID,
    TrajectoryRequestBatch,
    TrajectoryRunner,
    TrainingPhase,
)
from skyrl_train.trajectory_runners.types import AgentLoopOutput
from skyrl_train.trajectory_runners.projections import attach_unshaped_rewards
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.trajectory_runners.trajectory_processing import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)


class DefaultAgentWithReminder(DefaultAgent):
    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the output."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."

        self.add_message("user", observation)
        return output


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    trajectory_runner_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        exit_status, result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(trajectory_runner_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            save_traj(
                agent,
                path,
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                reward=reward,
                eval_error=eval_error,
            )  # type: ignore[arg-type]

    return (agent.messages if agent is not None else [], reward, error)


class MiniSweTrajectoryRunner(TrajectoryRunner):
    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        tokenizer,
        model_name: str,
    ):
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.trajectory_runner_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSweTrajectoryRunner doesn't support a custom chat template")

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Optional[AgentLoopOutput]:
        sweagent_config = yaml.safe_load(get_config_path(self.trajectory_runner_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.trajectory_runner_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        if not len(messages):
            return None

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=True)
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        while response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return AgentLoopOutput(
            evidence=RolloutEvidence(
                messages=tuple(response_messages),
                stop_reason=stop_reason,
                generated_token_count=sum(bool(value) for value in loss_mask),
                prompt_token_ids=tuple(prompt_ids),
                response_token_ids=tuple(response_ids),
            ),
            verification=VerificationResult.verified(reward),
            reward=RewardResult(unshaped_reward=reward, optimization_reward=reward),
            disposition=TrainingDisposition.train(),
            loss_mask=loss_mask,
            env_metrics={},
        )

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: TrajectoryRequestBatch
        Returns:
            TrajectoryBatch
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.trajectory_runner_cfg.sampling_params.max_generate_length
        max_input_length = self.trajectory_runner_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.trajectory_runner_cfg.backend, self.trajectory_runner_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        # Filter out the `None` entries, which means that trajectory generation failed
        valid_outputs = [output for output in all_outputs if output is not None]
        responses = [list(output.evidence.response_token_ids) for output in valid_outputs]
        rewards = [output.reward.optimization_reward for output in valid_outputs]
        unshaped_rewards = [output.reward.unshaped_reward for output in valid_outputs]
        stop_reasons = [output.evidence.stop_reason for output in valid_outputs]
        loss_masks = [output.loss_mask for output in valid_outputs]
        prompt_token_ids = [list(output.evidence.prompt_token_ids) for output in valid_outputs]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards, loss_masks=loss_masks)

        trajectory_batch: TrajectoryBatch = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            "exclude_from_baseline": [not output.disposition.baseline_eligible for output in valid_outputs],
        }
        attach_unshaped_rewards(trajectory_batch, unshaped_rewards)

        return trajectory_batch
