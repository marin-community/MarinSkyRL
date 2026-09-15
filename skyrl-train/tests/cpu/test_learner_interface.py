import asyncio
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from marinskyrl.checkpoint_paths import (
    CHECKPOINT_COMPLETE_FILENAME,
    LATEST_CHECKPOINT_FILE,
    POLICY_CHECKPOINT_SUBDIRECTORY,
)
from skyrl_train.callbacks.base import TrainerState
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.fully_async_trainer import (
    BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY,
    FullyAsyncRayPPOTrainer,
    _GenerationQueues,
    _validated_behavior_policy_version_segments,
)
from skyrl_train.learner import LearnerPublicationIncomplete, PublicationStatus, UnsupportedLearnerConfiguration
from skyrl_train.learner_bridge import (
    BEHAVIOR_POLICY_VERSIONS_METADATA_KEY,
    BEHAVIOR_POLICY_VERSION_SEGMENTS_METADATA_KEY,
)
from skyrl_train.policy_version import PublicationVersionHistory, expand_policy_version_segments
from skyrl_train.testing.stateful_fake_learner import (
    FakeLearnerError,
    FakeLearnerOperation,
    StatefulFakeLearner,
)
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.trainer_utils import ResumeMode
from tests.cpu.util import example_dummy_config


class _Dataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"uid": f"row-{index}"}

    def collate_fn(self, batch):
        return batch


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class _TrajectoryRunner:
    def set_trajectory_sink(self, sink) -> None:
        self.sink = sink

    async def shutdown(self) -> None:
        pass


_BASE_CONFIG = example_dummy_config()


def _config(checkpoint_path: Path, *, use_reference: bool = True):
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg.trainer.ckpt_path = str(checkpoint_path)
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.critic.model.path = None
    cfg.trainer.use_sample_packing = False
    cfg.trainer.step_wise_training = False
    with open_dict(cfg.trainer.algorithm):
        cfg.trainer.algorithm.max_seq_len = 8
    cfg.trainer.algorithm.use_kl_loss = use_reference
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.use_tis = False
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.train_batch_size = 2
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.trainer.fully_async.max_staleness_steps = 3
    cfg.generator.async_engine = True
    cfg.generator.enable_http_endpoint = True
    cfg.generator.sampling_params.logprobs = 1
    return cfg


def _trainer(
    checkpoint_path: Path,
    learner: StatefulFakeLearner,
    *,
    use_reference: bool = True,
    fully_async: bool = False,
    changes: dict[str, object] | None = None,
    trajectory_runner=None,
):
    trainer_type = FullyAsyncRayPPOTrainer if fully_async else RayPPOTrainer
    cfg = _config(checkpoint_path, use_reference=use_reference)
    for path, value in (changes or {}).items():
        OmegaConf.update(cfg, path, value)
    return trainer_type(
        cfg=cfg,
        tracker=None,
        tokenizer=_Tokenizer(),
        train_dataset=_Dataset(),
        inference_engine_client=None,
        trajectory_runner=trajectory_runner or _TrajectoryRunner(),
        learner=learner,
    )


def _training_input(*, loss_mask: torch.Tensor | None = None, behavior_versions: list[int] | None = None):
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[4, 5, 11, 12, 0], [0, 7, 13, 0, 0]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.int64),
            "response_mask": response_mask,
            "loss_mask": loss_mask if loss_mask is not None else response_mask.float(),
            "rollout_logprobs": torch.tensor([[-1.1, -1.2, 0.0], [-1.3, 0.0, 0.0]]),
        }
    )
    batch.metadata = {
        "response_length": 3,
        BEHAVIOR_POLICY_VERSIONS_METADATA_KEY: behavior_versions or [0, 0],
    }
    return batch


def _publish(trainer: RayPPOTrainer) -> None:
    trainer.init_weight_sync_state()
    asyncio.run(trainer._sync_policy_for_rollouts(reason="test"))


def _run_update(trainer: RayPPOTrainer, *, behavior_versions: list[int] | None = None):
    batch = trainer.fwd_logprobs_values_reward(_training_input(behavior_versions=behavior_versions))
    batch["advantages"] = torch.tensor([[1.0, -0.5, 0.0], [0.25, 0.0, 0.0]])
    return trainer.train_critic_and_policy(batch)


def test_msrl_learner_round_trip_restores_the_next_update(tmp_path):
    learner = StatefulFakeLearner()
    trainer = _trainer(tmp_path, learner)
    trainer.build_models(None, None, None)
    _publish(trainer)

    batch = trainer.fwd_logprobs_values_reward(_training_input())
    assert batch["action_log_probs"][0, 2] == 0
    assert not torch.equal(batch["action_log_probs"], batch["base_action_log_probs"])
    before_update = batch["action_log_probs"].clone()
    batch["advantages"] = torch.tensor([[1.0, -0.5, 0.0], [0.25, 0.0, 0.0]])
    assert trainer.train_critic_and_policy(batch)["update_status"] == "succeeded"
    assert not learner.state.ready_for_rollouts

    asyncio.run(trainer._sync_policy_for_rollouts(reason="updated"))
    assert learner.state.ready_for_rollouts
    assert not torch.equal(trainer.fwd_logprobs_values_reward(_training_input())["action_log_probs"], before_update)

    trainer.global_step = 1
    trainer.save_checkpoints()
    checkpoint_path = tmp_path / "global_step_1"
    assert (checkpoint_path / POLICY_CHECKPOINT_SUBDIRECTORY / "fake_learner_state.json").is_file()
    assert (checkpoint_path / CHECKPOINT_COMPLETE_FILENAME).read_text() == "1"
    assert (tmp_path / LATEST_CHECKPOINT_FILE).read_text() == "1"

    restored = StatefulFakeLearner(initial_parameter=99.0)
    restored_trainer = _trainer(tmp_path, restored)
    restored_trainer.resume_mode = ResumeMode.FROM_PATH
    restored_trainer.cfg.trainer.resume_path = str(checkpoint_path)
    restored_step, _ = restored_trainer.load_checkpoints()
    restored_trainer.global_step = restored_step
    assert not restored.state.ready_for_rollouts
    asyncio.run(restored_trainer._sync_policy_for_rollouts(reason="restored"))

    assert _run_update(trainer) == _run_update(restored_trainer)
    torch.testing.assert_close(
        trainer.fwd_logprobs_values_reward(_training_input())["action_log_probs"],
        restored_trainer.fwd_logprobs_values_reward(_training_input())["action_log_probs"],
        rtol=0,
        atol=0,
    )


def test_learner_failures_never_report_completion(tmp_path):
    learner = StatefulFakeLearner()
    trainer = _trainer(tmp_path, learner, use_reference=False)
    _publish(trainer)

    initial_state = learner.state
    for operation, action in (
        (FakeLearnerOperation.LOG_PROBS, lambda: trainer.fwd_logprobs_values_reward(_training_input())),
        (FakeLearnerOperation.UPDATE, lambda: _run_update(trainer)),
    ):
        learner.fail_next(operation)
        with pytest.raises(FakeLearnerError):
            action()
        assert learner.state == initial_state

    _run_update(trainer)
    learner.defer_next_publication()
    with pytest.raises(LearnerPublicationIncomplete):
        asyncio.run(trainer._sync_policy_for_rollouts(reason="pending"))
    assert learner.state.publication_status is PublicationStatus.PENDING
    assert not learner.state.ready_for_rollouts
    learner.complete_pending_publication()

    _run_update(trainer)
    learner.fail_next(FakeLearnerOperation.PUBLISH)
    with pytest.raises(FakeLearnerError):
        asyncio.run(trainer._sync_policy_for_rollouts(reason="failed"))
    assert learner.state.publication_status is PublicationStatus.FAILED
    assert not learner.state.ready_for_rollouts

    trainer.global_step = 2
    learner.fail_next(FakeLearnerOperation.SAVE)
    with pytest.raises(FakeLearnerError):
        trainer.save_checkpoints()
    assert not (tmp_path / LATEST_CHECKPOINT_FILE).exists()
    assert not (tmp_path / "global_step_2" / CHECKPOINT_COMPLETE_FILENAME).exists()
    incomplete = StatefulFakeLearner()
    incomplete_trainer = _trainer(tmp_path, incomplete, use_reference=False)
    incomplete_trainer.resume_mode = ResumeMode.FROM_PATH
    incomplete_trainer.cfg.trainer.resume_path = str(tmp_path / "global_step_2")
    with pytest.raises(RuntimeError, match="completion marker is missing"):
        incomplete_trainer.load_checkpoints()
    incomplete_trainer.global_step = 2
    with pytest.raises(RuntimeError, match="completed checkpoint marker is missing"):
        incomplete_trainer._handle_hf_export()
    trainer.save_checkpoints()

    fresh = StatefulFakeLearner()
    fresh_trainer = _trainer(tmp_path, fresh, use_reference=False)
    fresh_trainer.resume_mode = ResumeMode.FROM_PATH
    fresh_trainer.cfg.trainer.resume_path = str(tmp_path / "global_step_2")
    fresh.fail_next(FakeLearnerOperation.LOAD)
    with pytest.raises(FakeLearnerError):
        fresh_trainer.load_checkpoints()
    assert fresh.state.policy_version == fresh.state.update_count == 0


def test_stateful_fake_preserves_regular_mask_denominator_and_rollout_probabilities(tmp_path):
    trainer = _trainer(
        tmp_path,
        StatefulFakeLearner(),
        use_reference=False,
        changes={
            "trainer.algorithm.offpolicy_mask.enabled": True,
            "trainer.algorithm.offpolicy_mask.ratio": "mismatch",
            "trainer.algorithm.offpolicy_mask.low": 0.5,
            "trainer.algorithm.offpolicy_mask.high": 5.0,
            "trainer.algorithm.offpolicy_mask.veto_ratio": 1.0e-5,
            "trainer.algorithm.offpolicy_mask.renormalize": False,
        },
    )
    _publish(trainer)
    missing = _training_input()
    missing["rollout_logprobs"] = None
    missing = trainer.fwd_logprobs_values_reward(missing)
    missing["advantages"] = torch.ones((2, 3))
    with pytest.raises(ValueError, match="requires rollout log probabilities"):
        trainer.train_critic_and_policy(missing)

    batch = trainer.fwd_logprobs_values_reward(_training_input())
    rollout_logprobs = batch["action_log_probs"].clone()
    rollout_logprobs[0, 0] -= torch.log(torch.tensor(6.0))
    rollout_logprobs[1, 0] -= torch.log(torch.tensor(1.0e-6))
    batch["rollout_logprobs"] = rollout_logprobs
    batch["advantages"] = torch.tensor([[1.0, -0.5, 0.0], [0.25, 0.0, 0.0]])

    metrics = trainer.train_critic_and_policy(batch)

    # One token survives. Its fake term is -0.5 * (1 + 12 % 7 / 10)=-0.75,
    # still divided by all three originally selected tokens.
    assert metrics["fake/update_signal"] == pytest.approx(-0.25)
    assert metrics["offpolicy_mask/masked_fraction"] == pytest.approx(2 / 3)
    assert metrics["offpolicy_mask/masked_fraction_low"] == pytest.approx(1 / 3)
    assert metrics["offpolicy_mask/masked_fraction_high"] == pytest.approx(1 / 3)
    assert metrics["offpolicy_mask/vetoed_sequence_fraction"] == pytest.approx(0.5)


def test_skipped_and_non_finite_updates_preserve_state(tmp_path):
    learner = StatefulFakeLearner()
    trainer = _trainer(tmp_path, learner, use_reference=False)
    _publish(trainer)
    initial_state = learner.state

    batch = trainer.fwd_logprobs_values_reward(_training_input(loss_mask=torch.zeros((2, 3))))
    batch["advantages"] = torch.ones((2, 3))
    assert trainer.train_critic_and_policy(batch)["update_status"] == "skipped"
    assert learner.state == initial_state

    batch = trainer.fwd_logprobs_values_reward(_training_input())
    batch["advantages"] = torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        trainer.train_critic_and_policy(batch)
    assert learner.state == initial_state


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"trainer.use_sample_packing": True}, "sample packing"),
        ({"trainer.update_ref_every_epoch": True}, "reference-policy updates"),
        ({"trainer.algorithm.think_token_weight": 0.5}, "think-token loss weighting"),
        ({"trainer.policy.fsdp_config.moe_router_replay": True}, "MoE router replay"),
        ({"trainer.algorithm.z_clip.enabled": True}, "z-clip"),
        ({"trainer.algorithm.stale_clip.enabled": True}, "stale-clip"),
        (
            {"trainer.algorithm.use_tis": True, "trainer.algorithm.tis_imp_ratio_cap": -1.0},
            "importance ratio cap",
        ),
        (
            {"trainer.algorithm.offpolicy_mask.enabled": True, "trainer.algorithm.offpolicy_mask.low": 0.0},
            "off-policy mask requires",
        ),
        (
            {
                "trainer.algorithm.offpolicy_mask.enabled": True,
                "trainer.algorithm.policy_loss_type": "behavior_clip",
            },
            "off-policy mask currently requires regular",
        ),
    ],
)
def test_learner_preflight_rejects_worker_only_semantics_before_allocation(tmp_path, changes, match):
    class AllocationMustNotStart(BasePPOExp):
        def get_tokenizer(self, padding_side="left"):
            raise AssertionError("allocation started before learner preflight")

    cfg = _config(tmp_path, use_reference=False)
    for path, value in changes.items():
        OmegaConf.update(cfg, path, value)
    learner = StatefulFakeLearner()

    with pytest.raises(UnsupportedLearnerConfiguration, match=match):
        AllocationMustNotStart(cfg, learner=learner)
    assert learner.state.lifecycle.value == "uninitialized"


@pytest.mark.parametrize("mode", ["behavior_clip", "tis"])
def test_behavior_probability_channel_changes_the_update(tmp_path, mode):
    def update_signal(name: str, rollout_shift: float) -> float:
        changes = (
            {"trainer.algorithm.policy_loss_type": "behavior_clip"}
            if mode == "behavior_clip"
            else {"trainer.algorithm.use_tis": True, "trainer.algorithm.tis_imp_ratio_cap": 10.0}
        )
        trainer = _trainer(
            tmp_path / name,
            StatefulFakeLearner(),
            use_reference=False,
            changes=changes,
        )
        _publish(trainer)
        batch = _training_input()
        batch["rollout_logprobs"] += rollout_shift
        batch = trainer.fwd_logprobs_values_reward(batch)
        batch["advantages"] = torch.tensor([[1.0, -0.5, 0.0], [0.25, 0.0, 0.0]])
        return trainer.train_critic_and_policy(batch)["fake/update_signal"]

    assert update_signal("baseline", 0.0) != update_signal("shifted", -1.0)


def test_global_normalization_and_authoritative_metrics_cross_the_boundary(tmp_path):
    class CollidingMetricsLearner(StatefulFakeLearner):
        def update(self, request):
            result = super().update(request)
            return replace(result, metrics={**result.metrics, "update_status": -1.0, "policy_version": -1.0})

    learner = CollidingMetricsLearner()
    trainer = _trainer(
        tmp_path,
        learner,
        use_reference=False,
        changes={"trainer.algorithm.loss_reduction": "seq_mean_token_sum_norm_global"},
    )
    _publish(trainer)

    result = _run_update(trainer)

    assert result["fake/global_loss_denominator"] > 0
    assert result["update_status"] == "succeeded"
    assert result["policy_version"] == 1


def test_async_serving_identity_is_per_row_and_independent_of_global_step(tmp_path):
    class OnePromptDataloader:
        returned = False

        async def get_next_non_consumed_data(self):
            if self.returned:
                return None
            self.returned = True
            return [{"uid": "row", "prompt": [], "env_class": None, "env_extras": {}}]

    class VersionedRunner(_TrajectoryRunner):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def run(self, _request, disable_tqdm):
            self.started.set()
            await self.release.wait()
            return {
                "prompt_token_ids": [[1, 2], [1, 2]],
                "response_ids": [[11, 12], [13]],
                "rewards": [1.0, 1.0],
                "unshaped_rewards": [1.0, 1.0],
                "loss_masks": [[1, 1], [1]],
                "stop_reasons": ["stop", "stop"],
                "rollout_metrics": {},
                "rollout_logprobs": None,
                "is_last_step": [True, True],
                "exclude_from_baseline": [False, False],
                "actual_global_step": 10,
                BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY: [
                    [
                        {"start": 0, "token_count": 1, "policy_version": 0},
                        {"start": 1, "token_count": 1, "policy_version": 1},
                    ],
                    [{"start": 0, "token_count": 1, "policy_version": 1}],
                ],
            }

    async def capture_group():
        runner = VersionedRunner()
        trainer = _trainer(
            tmp_path,
            StatefulFakeLearner(),
            use_reference=False,
            fully_async=True,
            trajectory_runner=runner,
            changes={
                "generator.n_samples_per_prompt": 2,
                "trainer.policy_mini_batch_size": 1,
                "trainer.train_batch_size": 1,
                "trainer.algorithm.resolved_group_advantage.physical_group_size": 2,
            },
        )
        # Deliberately disagree with the runner's actual_global_step=10. Serving
        # provenance, not the trainer counter, determines this group's age.
        trainer.global_step = 1
        trainer.async_train_dataloader = OnePromptDataloader()
        trainer.init_weight_sync_state()
        await trainer.async_sync_policy_weights_to_inference_engines()
        queues = _GenerationQueues(
            completed=asyncio.Queue(maxsize=2),
            retries=asyncio.Queue(),
            condition=asyncio.Condition(),
            active_producers=1,
        )
        producer = asyncio.create_task(trainer._run_generate_for_a_group_loop(queues))
        await runner.started.wait()
        _run_update(trainer)
        await trainer.async_sync_policy_weights_to_inference_engines()
        runner.release.set()
        group = await asyncio.wait_for(queues.completed.get(), timeout=1)
        producer.cancel()
        await producer
        return trainer, group

    trainer, group = asyncio.run(capture_group())
    assert group.earliest_model_step == 1
    assert group.behavior_policy_version_segments == [
        [
            {"start": 0, "token_count": 1, "policy_version": 0},
            {"start": 1, "token_count": 1, "policy_version": 1},
        ],
        [{"start": 0, "token_count": 1, "policy_version": 1}],
    ]
    batch = trainer.convert_generation_group_mini_batch_to_training_input([group])
    assert batch.metadata[BEHAVIOR_POLICY_VERSION_SEGMENTS_METADATA_KEY] == group.behavior_policy_version_segments
    batch = trainer.fwd_logprobs_values_reward(batch)
    batch["advantages"] = torch.tensor([[1.0, -0.5], [-0.25, 0.0]])
    result = trainer.train_critic_and_policy(batch)
    assert (result["fake/oldest_behavior_version"], result["fake/newest_behavior_version"]) == (0.0, 1.0)

    with pytest.raises(RuntimeError):
        _validated_behavior_policy_version_segments(
            {
                "response_ids": [[11, 12]],
                "loss_masks": [[1, 1]],
                BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY: [[{"start": 0, "token_count": 1, "policy_version": 0}]],
            }
        )


def test_policy_version_segments_expand_only_at_the_learner_boundary():
    rows = [
        [
            {"start": 0, "token_count": 1, "policy_version": 3},
            {"start": 1, "token_count": 1, "policy_version": 4},
            {"start": 2, "token_count": 1, "policy_version": None},
        ],
        [{"start": 0, "token_count": 1, "policy_version": 4}],
    ]
    response_mask = torch.tensor([[1, 1, 1], [1, 0, 0]]).numpy()
    required_mask = torch.tensor([[1, 1, 0], [1, 0, 0]]).numpy()

    dense = expand_policy_version_segments(rows, response_mask, required_mask=required_mask)

    assert dense.tolist() == [[3, 4, -1], [4, -1, -1]]


def test_publication_version_history_uses_engine_first_token_clock():
    history = PublicationVersionHistory()
    history.record_resume(10.0, 0)
    history.record_resume(20.0, 1)

    assert history.at_first_token(9.9) is None
    assert history.at_first_token(10.0) == 0
    assert history.at_first_token(19.9) == 0
    assert history.at_first_token(20.0) == 1


def test_async_checkpoint_commits_after_required_callbacks(tmp_path):
    previous_checkpoint = tmp_path / "global_step_0"
    previous_checkpoint.mkdir()
    (tmp_path / LATEST_CHECKPOINT_FILE).write_text("0")
    trainer = _trainer(tmp_path, StatefulFakeLearner(), use_reference=False, fully_async=True)
    trainer.global_step = 1
    trainer._last_saved_step = 0
    state = TrainerState(global_step=1, epoch=0, total_steps=1, num_steps_per_epoch=1)

    with pytest.raises(RuntimeError):
        trainer.save_checkpoints()
    with pytest.raises(RuntimeError):
        asyncio.run(trainer._save_intermediate_checkpoint(state))

    assert (tmp_path / LATEST_CHECKPOINT_FILE).read_text() == "0"
    assert not (tmp_path / "global_step_1" / CHECKPOINT_COMPLETE_FILENAME).exists()
    assert trainer._last_saved_step == 0
    assert previous_checkpoint.is_dir()

    incomplete = _trainer(tmp_path, StatefulFakeLearner(), use_reference=False)
    incomplete.resume_mode = ResumeMode.FROM_PATH
    incomplete.cfg.trainer.resume_path = str(tmp_path / "global_step_1")
    with pytest.raises(RuntimeError, match="completion marker is missing"):
        incomplete.load_checkpoints()
    incomplete.global_step = 1
    with pytest.raises(RuntimeError, match="completed checkpoint marker is missing"):
        incomplete._handle_hf_export()


def test_levanter_hf_export_is_idempotent_within_one_run(tmp_path):
    learner = StatefulFakeLearner()
    trainer = _trainer(tmp_path, learner, use_reference=False)
    trainer.global_step = 1
    checkpoint = tmp_path / "global_step_1"
    checkpoint.mkdir()
    (checkpoint / CHECKPOINT_COMPLETE_FILENAME).write_text("1")
    exports = []
    learner.export_policy = lambda path: exports.append(path)

    trainer._handle_hf_export()
    trainer._handle_hf_export()

    assert len(exports) == 1


def test_learner_close_failure_is_reported_after_later_cleanup(tmp_path):
    learner = StatefulFakeLearner()
    trainer = _trainer(tmp_path, learner, use_reference=False)
    cleanup_events = []
    trainer._kill_ray_actors = lambda: cleanup_events.append("ray")
    trainer._start_exit_watchdog = lambda timeout: cleanup_events.append("watchdog")
    learner.fail_next(FakeLearnerOperation.CLOSE)

    with pytest.raises(ExceptionGroup):
        asyncio.run(trainer.shutdown())

    assert cleanup_events == ["ray", "watchdog"]
    assert not trainer._shutdown_complete
