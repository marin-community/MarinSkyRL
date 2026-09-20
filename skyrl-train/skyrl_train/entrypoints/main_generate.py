"""
Main entrypoint for evaluation-only.
"""

import asyncio
from pathlib import Path
from typing import Any, Protocol

import hydra
from omegaconf import OmegaConf
import ray
from loguru import logger
from omegaconf import DictConfig

from marinskyrl.checkpoint_paths import DRAFT_CHECKPOINT_SUBDIRECTORY
from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
)
from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from marinskyrl.speculative_decoding import (
    EVALUATION_ENTRYPOINT,
    SpeculativeDecodingConfig,
    parse_speculative_decoding_config,
)
from skyrl_train.draft_trainer import DraftUpdateRequest, create_draft_trainer
from skyrl_train.eagle_replay import (
    EagleReplaySelection,
    load_replay_rows,
    replay_batches,
    select_replay_sequences,
)
from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest, lora_disk_load_request
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    RayWrappedInferenceEngine,
    release_owned_placement_groups,
)
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    OnlineEagleCaptureConfig,
    OnlineEagleUpdateResult,
    active_capture_results,
    replay_session_id,
)
from skyrl_train.io import io
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray
from skyrl_train.evaluate import evaluate
from skyrl_train.utils.trainer_utils import build_dataloader


class PolicyAdapterClient(Protocol):
    async def update_named_weights(self, request: NamedWeightsUpdateRequest) -> None: ...


async def load_initial_policy_adapter(inference_engine_client: PolicyAdapterClient, cfg: DictConfig) -> None:
    """Load a configured policy LoRA before an evaluation-only rollout."""
    adapter_path = cfg.trainer.policy.model.lora.adapter_path
    if adapter_path is None:
        return
    if not cfg.generator.run_engines_locally:
        raise ValueError("evaluation-only LoRA loading requires local inference engines")
    if cfg.generator.backend != "vllm":
        raise ValueError("evaluation-only LoRA loading currently requires the vLLM backend")
    path = Path(adapter_path)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("evaluation-only LoRA adapter_path must be an existing absolute directory")
    await inference_engine_client.update_named_weights(lora_disk_load_request(str(path)))


def _offline_speculative_decoding_config(cfg: DictConfig) -> SpeculativeDecodingConfig | None:
    raw = cfg.generator.get("speculative_decoding")
    return parse_speculative_decoding_config(
        None if raw is None else OmegaConf.to_container(raw, resolve=True),
        backend=cfg.generator.backend,
        run_engines_locally=cfg.generator.run_engines_locally,
        entrypoint=EVALUATION_ENTRYPOINT,
        colocate_all=cfg.trainer.placement.colocate_all,
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        async_engine=cfg.generator.async_engine,
        engine_init_kwargs=OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True),
    )


async def _begin_offline_eagle_capture(
    inference_engine_client: InferenceEngineClient,
    cfg: DictConfig,
    speculative_decoding: SpeculativeDecodingConfig | None,
) -> str | None:
    if speculative_decoding is None or speculative_decoding.training is None:
        return None
    if not is_cloud_uri(cfg.trainer.ckpt_path):
        raise ValueError("Offline EAGLE distillation requires trainer.ckpt_path to be cloud-backed")
    capture_uri = join_resource_path(cfg.trainer.ckpt_path, DRAFT_CHECKPOINT_SUBDIRECTORY, "captures", "offline-step-1")
    if await asyncio.to_thread(io.exists, capture_uri):
        await asyncio.to_thread(io.remove, capture_uri)
    source_identity = cfg.trainer.policy.model.get("source_identity")
    if not source_identity:
        raise ValueError("Offline EAGLE distillation requires trainer.policy.model.source_identity")
    capture = OnlineEagleCaptureConfig(
        step=1,
        max_tokens=speculative_decoding.training.max_tokens_per_update,
        max_window_tokens=speculative_decoding.training.max_window_tokens,
        target_revision=str(source_identity),
        draft_revision=speculative_decoding.model.source_identity,
        reserved_gpu_memory_gib=speculative_decoding.training.reserved_gpu_memory_gib,
    )
    results = await inference_engine_client.begin_online_eagle_capture(capture.to_mapping())
    if not active_capture_results(results):
        raise RuntimeError("Offline EAGLE capture did not start on any rollout rank")
    return capture_uri


async def _seal_offline_eagle_capture(
    inference_engine_client: InferenceEngineClient,
    capture_uri: str,
) -> int:
    manifests = await inference_engine_client.seal_online_eagle_capture(capture_uri)
    active = active_capture_results(manifests)
    if not active:
        raise RuntimeError("Offline EAGLE capture did not publish any rank manifests")
    return sum(int(item.get("captured_rows", 0)) for item in active)


async def _release_inference_engines(inference_engine_client: InferenceEngineClient) -> None:
    await inference_engine_client.teardown()
    local_engines = []
    for engine in inference_engine_client.engines:
        if isinstance(engine, RayWrappedInferenceEngine):
            local_engines.append(engine)
            ray.kill(engine.inference_engine_actor, no_restart=True)
    release_owned_placement_groups(local_engines)


async def _train_offline_eagle_draft(
    cfg: DictConfig,
    capture_uri: str,
    speculative_decoding: SpeculativeDecodingConfig,
) -> OnlineEagleUpdateResult:
    assert speculative_decoding.training is not None
    checkpoint_root = join_resource_path(cfg.trainer.ckpt_path, DRAFT_CHECKPOINT_SUBDIRECTORY)
    draft_trainer = create_draft_trainer(
        initial_model=speculative_decoding.model,
        checkpoint_root=checkpoint_root,
    )
    try:
        update = await draft_trainer.update.remote(
            DraftUpdateRequest(
                step=1,
                capture_uri=capture_uri,
                target_revision=str(cfg.trainer.policy.model.source_identity),
                num_speculative_tokens=speculative_decoding.num_speculative_tokens,
                seed=int(cfg.trainer.seed),
                training=speculative_decoding.training,
            )
        )
    finally:
        ray.kill(draft_trainer, no_restart=True)
    if not isinstance(update, OnlineEagleUpdateResult) or not update.accepted or update.candidate_uri is None:
        error = update.error if isinstance(update, OnlineEagleUpdateResult) else type(update).__name__
        raise RuntimeError(f"Offline EAGLE draft update was not accepted: {error}")
    return update


class EvalOnlyEntrypoint(BasePPOExp):
    entrypoint_name = EVALUATION_ENTRYPOINT

    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    def get_eval_dataset(self):
        """Keep replay sources raw; ordinary generation uses PromptDataset."""
        if self.cfg.data.eagle_replay:
            return tuple(self.cfg.data.val_data)
        return super().get_eval_dataset()

    async def _run_eagle_replay(
        self,
        inference_engine_client: InferenceEngineClient,
        speculative_decoding: SpeculativeDecodingConfig,
        selection: EagleReplaySelection,
    ) -> dict[str, Any]:
        training = speculative_decoding.training
        assert training is not None
        capture_uri = await _begin_offline_eagle_capture(inference_engine_client, self.cfg, speculative_decoding)
        assert capture_uri is not None
        try:
            for batch in replay_batches(selection, int(self.cfg.data.eagle_replay_batch_size)):
                await inference_engine_client.generate(
                    {
                        "prompts": None,
                        "prompt_token_ids": [sequence.token_ids for sequence in batch],
                        "sampling_params": {"max_tokens": 1, "temperature": 0},
                        "session_ids": [
                            replay_session_id(sequence.group_id, sequence.loss_start) for sequence in batch
                        ],
                    }
                )
            sealed_rows = await _seal_offline_eagle_capture(inference_engine_client, capture_uri)
        finally:
            await _release_inference_engines(inference_engine_client)

        update = await _train_offline_eagle_draft(self.cfg, capture_uri, speculative_decoding)
        return {
            "speculator/replay_sequences": float(len(selection.sequences)),
            "speculator/replay_tokens": float(selection.charged_tokens),
            "speculator/replay_skipped_rows": float(selection.skipped_rows),
            "speculator/sealed_rows": float(sealed_rows),
            "speculator/candidate_holdout_agreement": update.candidate_holdout_agreement,
            "speculator/candidate_holdout_loss": update.candidate_holdout_loss,
            "speculator/train_loss": update.train_loss,
            "speculator/train_duration_seconds": update.duration_seconds,
        }

    async def run(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        speculative_decoding = _offline_speculative_decoding_config(self.cfg)
        replay_selection = None
        if self.cfg.data.eagle_replay:
            if speculative_decoding is None or speculative_decoding.training is None:
                raise ValueError("data.eagle_replay requires speculative_decoding.training")
            training = speculative_decoding.training
            replay_selection = select_replay_sequences(
                load_replay_rows(self.eval_dataset),
                self.tokenizer,
                max_tokens=training.max_tokens_per_update,
                max_window_tokens=training.max_window_tokens,
                max_sequences_per_group=training.max_sequences_per_prompt_group,
            )
            required_sequences = training.min_train_sequences + training.min_holdout_sequences
            if len(replay_selection.sequences) < required_sequences:
                raise ValueError(
                    f"EAGLE replay selected {len(replay_selection.sequences)} sequences; "
                    f"at least {required_sequences} are required"
                )

        inference_engine_client = self.create_inference_engine_client()
        await inference_engine_client.wake_up()
        await load_initial_policy_adapter(inference_engine_client, self.cfg)

        if replay_selection is not None:
            assert speculative_decoding is not None
            results = await self._run_eagle_replay(inference_engine_client, speculative_decoding, replay_selection)
            tracker = self.get_tracker()
            tracker.log(results, step=0, commit=True)
            return results

        trajectory_runner = self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)
        capture_uri = await _begin_offline_eagle_capture(inference_engine_client, self.cfg, speculative_decoding)

        try:
            results: dict[str, Any] = await evaluate(
                eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
                trajectory_runner=trajectory_runner,
                cfg=self.cfg,
                global_step=None,
                tokenizer=self.tokenizer,
            )
            if capture_uri is not None:
                results["speculator/sealed_rows"] = float(
                    await _seal_offline_eagle_capture(inference_engine_client, capture_uri)
                )
        finally:
            if capture_uri is not None:
                await _release_inference_engines(inference_engine_client)

        if capture_uri is not None:
            assert speculative_decoding is not None and speculative_decoding.training is not None
            update = await _train_offline_eagle_draft(self.cfg, capture_uri, speculative_decoding)
            results.update(
                {
                    "speculator/candidate_holdout_agreement": update.candidate_holdout_agreement,
                    "speculator/candidate_holdout_loss": update.candidate_holdout_loss,
                    "speculator/train_loss": update.train_loss,
                    "speculator/train_duration_seconds": update.duration_seconds,
                }
            )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: DictConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
