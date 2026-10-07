"""Static trajectory runner that emits dataset-supplied preference pairs.

DPO reads chosen and rejected completions from the training parquet instead of
generating rollouts. Each prompt is leased with ``n_samples_per_prompt=2``; repetition
0 becomes the chosen row (+1 role) and repetition 1 the rejected row (-1 role), so the
trainer's group machinery delivers adjacent rows that the DPO loss pairs in order.
"""

from loguru import logger

from skyrl_train.dataset.preference_pairs import completion_text
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.types import (
    BatchMetadata,
    TrajectoryBatch,
    TrajectoryRequestBatch,
)


def _token_ids(tokenizer, prompt) -> list[int]:
    """Apply the chat template with a generation prompt, normalizing Transformers 5 returns."""
    encoded = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    if hasattr(encoded, "get"):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list):
        raise TypeError(f"tokenizer chat template returned unsupported token IDs: {type(encoded).__name__}")
    return [int(token) for token in encoded]


class PreferencePairTrajectoryRunner(TrajectoryRunner):
    """Emit tokenized dataset completions without touching any inference engine."""

    def __init__(self, tokenizer, *, max_generate_length: int, max_input_length: int):
        self.tokenizer = tokenizer
        self.max_generate_length = max_generate_length
        self.max_input_length = max_input_length

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"] or []
        if len(prompts) % 2 or len(env_extras) != len(prompts):
            raise ValueError(
                f"preference pairs arrive as chosen/rejected row pairs; got {len(prompts)} rows "
                f"with {len(env_extras)} extras"
            )
        trajectory_ids = input_batch.get("trajectory_ids")
        if trajectory_ids is not None:
            for pair_index in range(len(prompts) // 2):
                chosen_id, rejected_id = trajectory_ids[2 * pair_index : 2 * pair_index + 2]
                if chosen_id.instance_id != rejected_id.instance_id or (
                    chosen_id.repetition_id, rejected_id.repetition_id
                ) != (0, 1):
                    raise ValueError("preference pairs require trajectory IDs ordered as repetition 0 then 1")

        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        pair_roles: list[int] = []
        for pair_index in range(len(prompts) // 2):
            prompt = prompts[2 * pair_index]
            extras = env_extras[2 * pair_index]
            if prompts[2 * pair_index + 1] != prompt or env_extras[2 * pair_index + 1] != extras:
                raise ValueError("both rows of a preference pair must carry the same prompt and extras")
            chosen = completion_text(extras.get("chosen"), "chosen")
            rejected = completion_text(extras.get("rejected"), "rejected")
            prompt_ids = _token_ids(self.tokenizer, prompt)
            completions = []
            for field, text in (("chosen", chosen), ("rejected", rejected)):
                completion_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                if not completion_ids:
                    raise ValueError(f"preference-pair {field} completion tokenized to zero tokens")
                if len(completion_ids) > self.max_generate_length:
                    raise ValueError(
                        f"preference-pair {field} completion needs {len(completion_ids)} tokens but "
                        f"generator.sampling_params.max_generate_length is {self.max_generate_length}"
                    )
                if len(prompt_ids) + len(completion_ids) > self.max_input_length + self.max_generate_length:
                    raise ValueError(
                        f"preference-pair {field} completion exceeds the "
                        f"{self.max_input_length + self.max_generate_length}-token sequence budget; "
                        "filter the pair at dataset load instead"
                    )
                completions.append([int(token) for token in completion_ids])
            prompt_token_ids.extend((prompt_ids, prompt_ids))
            response_ids.extend(completions)
            rewards.extend((1.0, 0.0))
            loss_masks.extend(([1] * len(completions[0]), [1] * len(completions[1])))
            pair_roles.extend((1, -1))

        batch: TrajectoryBatch = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "data_sources": None,
            "rewards": rewards,
            "unshaped_rewards": None,
            "unshaped_reward_available": None,
            "reward_shaping_components": None,
            "reward_shaping_loop_spans": None,
            "loop_advantages": None,
            "reward_shaping_versions": None,
            "verification_results": [None] * len(response_ids),
            "evidence_messages": [None] * len(response_ids),
            "verifier_tests": None,
            "loss_masks": loss_masks,
            "stop_reasons": ["preference_pair"] * len(response_ids),
            "exception_types": None,
            "error_treatments": None,
            "server_errors": None,
            "rollout_metrics": {},
            "rollout_logprobs": None,
            "student_topk_indices": None,
            "behavior_topk_logprobs": None,
            "rollout_routed_experts": None,
            "teacher_evidence": None,
            "distillation": None,
            "token_level_shaping": None,
            "response_span_tags": None,
            "teacher_route_keys": None,
            "is_last_step": None,
            "exclude_from_baseline": None,
            "pair_roles": pair_roles,
        }
        metadata: BatchMetadata | None = input_batch.get("batch_metadata")
        if metadata is not None and metadata.training_phase == "eval":
            logger.warning("preference-pair evaluation replays the fixed dataset completions without generation")
        return batch
