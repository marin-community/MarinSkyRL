"""RayPPOTrainer driver behavior: checkpoint durability, resume, advantages, collation, and batch sizing."""

import asyncio
import copy
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray
import torch
from omegaconf import OmegaConf

import skyrl_train.trainer as trainer_module
from skyrl_train.batch_assembly import assemble_slice, plan_batch
from skyrl_train.batch_metrics import zero_std_group_fraction
from skyrl_train.callbacks.base import TrainerCallback
from skyrl_train.rollouts.buffer import RolloutGroup, RowFacts
from skyrl_train.utils.advantage_estimators import apply_loop_advantages, compute_advantages_and_returns
from skyrl_train.config.utils import get_default_config
from skyrl_train.distillation import ChosenTokenTeacherInput, TeacherTopKInput, TopKTeacherEvidence
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import CheckpointSnapshot
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.trajectory_runners.types import BatchFields, TrajectoryID
from skyrl_train.objective.losses import PolicyLossInputs, ppo_policy_loss
from skyrl_train.config.objective_spec import LossReduction
from skyrl_train.objective.reduction import reduce_to_step, step_counts
from skyrl_train.training_batch import TrainingBatchIterator
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.utils.utils import validate_batch_sizes
from skyrl_train.workers.worker import CriticWorkerBase, PolicyWorkerBase
from tests.cpu.util import example_dummy_config
from tests.rollout_fixtures import FixedPromptDataset, FixedRolloutRunner, RecordingTracker


@pytest.fixture
def dummy_config():
    return example_dummy_config()


class _CapturingPolicyGroup:
    def __init__(self):
        self.actor_infos = [SimpleNamespace(rank=SimpleNamespace(dp_size=2)) for _ in range(4)]
        self.training_batch = None

    def async_run_ray_method(self, dispatch_type, method_name, *args):
        del dispatch_type
        if method_name == "ppo_train":
            self.training_batch = args[0]
            return [object()]
        if method_name == "empty_cache":
            return []
        raise AssertionError(f"Unexpected policy method: {method_name}")


class _ForwardPolicyGroup:
    actor_infos = [SimpleNamespace(rank=MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=1, dp_size=1, pp_size=1))]

    def async_run_ray_method(self, dispatch_type, method_name, **kwargs):
        if method_name == "barrier_all":
            return [ray.put(None)]
        if method_name == "empty_cache":
            return []
        if method_name == "forward":
            data = kwargs["data"]
            output = torch.zeros(len(data["sequences"]), data.metadata["response_length"])
            return [ray.put(TrainingOutputBatch({"output": output}))]
        raise AssertionError(f"Unexpected policy method: {method_name}")


class _ResidencyPolicyGroup:
    def __init__(self):
        self.model_on_gpu = False
        self.optimizer_on_gpu = False

    def backload_to_gpu(self, backload_optimizer=True, backload_model=True):
        self.optimizer_on_gpu |= backload_optimizer
        self.model_on_gpu |= backload_model

    def offload_to_cpu(self, offload_optimizer=True, offload_model=True):
        if offload_optimizer:
            self.optimizer_on_gpu = False
        if offload_model:
            self.model_on_gpu = False


class _CheckpointResidencyPolicyGroup(_ResidencyPolicyGroup):
    def __init__(self):
        super().__init__()
        self.restore_residencies = []
        self.restored_dirs = []

    def async_run_ray_method(self, dispatch_type, method_name, **kwargs):
        assert dispatch_type == "pass_through"
        assert method_name == "load_checkpoint"
        assert kwargs["load_training_state"]
        self.restore_residencies.append((self.model_on_gpu, self.optimizer_on_gpu))
        self.restored_dirs.append(kwargs["ckpt_dir"])
        return []


class _ResidencyInferenceClient:
    def __init__(self):
        self.awake = True
        self.wake_tags = []

    async def sleep(self):
        self.awake = False

    async def wake_up(self, tags):
        self.wake_tags.append(tags)
        self.awake = True


@pytest.mark.parametrize("save_error", [None, RuntimeError("storage of size 0")])
def test_colocated_checkpoint_temporarily_backloads_policy_and_restores_rollout_residency(
    save_error, dummy_config, driver_trainer_factory
):
    trainer = driver_trainer_factory(dummy_config)
    trainer.colocate_all = True
    trainer.policy_model = _ResidencyPolicyGroup()
    trainer.inference_engine_client = _ResidencyInferenceClient()
    trainer.sync_policy_weights_to_inference_engines = AsyncMock()
    trainer.global_step = 3
    save_observations = []

    def snapshot_checkpoint(rollout_state):
        save_observations.append(
            (
                trainer.policy_model.model_on_gpu,
                trainer.policy_model.optimizer_on_gpu,
                trainer.inference_engine_client.awake,
            )
        )
        if save_error is not None:
            raise save_error

    trainer._snapshot_checkpoint = snapshot_checkpoint

    if save_error is None:
        asyncio.run(trainer._save_checkpoints_with_residency())
    else:
        with pytest.raises(RuntimeError, match="storage of size 0"):
            asyncio.run(trainer._save_checkpoints_with_residency())

    assert save_observations == [(True, True, False)]
    assert not trainer.policy_model.model_on_gpu
    assert not trainer.policy_model.optimizer_on_gpu
    assert trainer.inference_engine_client.awake
    assert trainer.inference_engine_client.wake_tags == [["weights"], ["kv_cache"]]


def test_intermediate_checkpoint_failure_is_recorded_and_later_save_can_succeed(
    monkeypatch, tmp_path, dummy_config, training_trainer_factory
):
    dummy_config.trainer.epochs = 2
    dummy_config.trainer.max_steps = 2
    dummy_config.trainer.ckpt_path = str(tmp_path / "checkpoints")
    dummy_config.trainer.ckpt_interval = 1
    dummy_config.trainer.max_ckpts_to_keep = -1
    dummy_config.trainer.eval_interval = dummy_config.trainer.hf_save_interval = -1
    dummy_config.trainer.eval_before_train = False
    dummy_config.trainer.algorithm.advantage_estimator = "uniform"
    dummy_config.trainer.algorithm.use_kl_loss = False
    dummy_config.trainer.algorithm.off_policy_correction = "none"
    groups = [
        RolloutGroup(
            {
                "prompt_token_ids": [[1]],
                "response_ids": [[2, 3]],
                "rewards": [1.0],
                "loss_masks": [[1, 1]],
                "stop_reasons": ["stop"],
                "rollout_metrics": {},
            },
            uid=uid,
            policy_step=0,
            prompt={"uid": uid},
        )
        for uid in ("a", "b")
    ]
    saved_steps = []

    class SaveObserver(TrainerCallback):
        def on_save(self, state, control, **kwargs):
            saved_steps.append(state.global_step)
            return control

    tracker = RecordingTracker()
    trainer = training_trainer_factory(
        dummy_config,
        dataset=FixedPromptDataset([group.uid for group in groups]),
        runner=FixedRolloutRunner(groups),
        tokenizer=SimpleNamespace(decode=str, pad_token_id=0),
        tracker=tracker,
        callbacks=[SaveObserver()],
    )
    makedirs = trainer_module.io.makedirs
    failed_directory = Path(dummy_config.trainer.ckpt_path) / "global_step_1"

    def fail_first_checkpoint(path, **kwargs):
        if Path(path) == failed_directory:
            raise OSError("AccessDenied")
        return makedirs(path, **kwargs)

    monkeypatch.setattr(trainer_module.io, "makedirs", fail_first_checkpoint)
    asyncio.run(trainer.train())

    logged = {step: metrics for metrics, step in tracker.logs}
    assert logged[1]["trainer/checkpoint_save_failures"] == 1.0
    assert saved_steps == [2]
    root = Path(dummy_config.trainer.ckpt_path)
    assert (root / "latest_ckpt_global_step.txt").read_text() == "2"
    assert torch.load(root / "global_step_2/trainer_state.pt", weights_only=False)["global_step"] == 2
    assert isinstance(torch.load(root / "global_step_2/data.pt", weights_only=False), TrainingContextState)


def test_intermediate_checkpoint_does_not_suppress_non_storage_failure(dummy_config, driver_trainer_factory):
    trainer = driver_trainer_factory(dummy_config)

    async def fail_save():
        raise ValueError("invalid checkpoint state")

    trainer._save_checkpoints_with_residency = fail_save
    state = SimpleNamespace(global_step=6)

    with pytest.raises(ValueError, match="invalid checkpoint state"):
        asyncio.run(trainer._save_intermediate_checkpoint(state))

    assert trainer.all_metrics == {}


def test_checkpoint_marker_waits_for_rank_uploads(monkeypatch, tmp_path, dummy_config, driver_trainer_factory):
    dummy_config.trainer.ckpt_path = str(tmp_path)
    dummy_config.trainer.max_ckpts_to_keep = 1
    trainer = driver_trainer_factory(dummy_config)
    trainer.global_step = 6
    old_checkpoint = tmp_path / "global_step_5"
    old_checkpoint.mkdir()
    (old_checkpoint / "data.pt").write_bytes(b"old data")
    marker = tmp_path / "latest_ckpt_global_step.txt"
    marker.write_text("5")
    upload_ref = object()

    def checkpoint_rpc(_dispatch, method, **kwargs):
        if method == "wait_checkpoint_upload":
            return [upload_ref]
        if method in {"save_checkpoint", "get_ray_node_id"}:
            return []
        raise AssertionError(method)

    trainer.policy_model = SimpleNamespace(async_run_ray_method=checkpoint_rpc)
    upload_started = threading.Event()
    release_upload = threading.Event()
    ray_get = ray.get

    def wait_for_upload(refs, *args, **kwargs):
        if refs == [upload_ref]:
            upload_started.set()
            assert release_upload.wait(timeout=5)
            return [None]
        return ray_get(refs, *args, **kwargs)

    monkeypatch.setattr(trainer_module.ray, "get", wait_for_upload)

    async def finish_upload():
        expected_state = await trainer.context.state_dict()
        task = asyncio.create_task(trainer.save_checkpoints())
        try:
            assert await asyncio.to_thread(upload_started.wait, 5)
            assert marker.read_text() == "5"
            assert old_checkpoint.is_dir()
        finally:
            release_upload.set()
            await task
        return expected_state

    expected_state = asyncio.run(finish_upload())

    assert torch.load(tmp_path / "global_step_6/data.pt", weights_only=False) == expected_state
    assert torch.load(tmp_path / "global_step_6/trainer_state.pt", weights_only=False)["global_step"] == 6
    assert marker.read_text() == "6"
    assert not old_checkpoint.exists()


def test_background_checkpoint_failure_does_not_advance_marker(
    monkeypatch, tmp_path, dummy_config, driver_trainer_factory
):
    trainer = driver_trainer_factory(dummy_config)
    trainer.policy_model = SimpleNamespace(async_run_ray_method=lambda *_args: [object()])

    def fail_upload(_refs):
        raise OSError("AccessDenied")

    monkeypatch.setattr(trainer_module.ray, "get", fail_upload)
    snapshot = CheckpointSnapshot(
        step=6,
        upload_started_at=0.0,
        rollout_state_path=str(tmp_path / "global_step_6" / "data.pt"),
        rollout_state_payload=b"data",
        trainer_state_path=str(tmp_path / "global_step_6" / "trainer_state.pt"),
        trainer_state_payload=b"trainer",
        marker_path=str(tmp_path / "latest_ckpt_global_step.txt"),
    )
    state = SimpleNamespace(global_step=6)

    async def drain_failure():
        trainer._start_checkpoint_upload(snapshot, state)
        await trainer._drain_checkpoint_upload()

    asyncio.run(drain_failure())

    assert not Path(snapshot.marker_path).exists()
    assert trainer.all_metrics["trainer/checkpoint_save_failures"] == 1.0


@pytest.mark.parametrize(
    ("checkpoint_committed", "expected_events"),
    [(True, ["checkpoint_committed", "hf_export"]), (False, ["checkpoint_failed"])],
)
def test_step_end_exports_hf_model_only_after_checkpoint_upload_commits(
    checkpoint_committed, expected_events, dummy_config, driver_trainer_factory
):
    trainer = driver_trainer_factory(dummy_config)

    async def request_hf_export(event, state, control, **_kwargs):
        control.should_save_hf_model = True
        return control

    trainer.callback_handler = SimpleNamespace(call_event_async=request_hf_export)
    events = []

    async def drain_upload():
        events.append("checkpoint_committed" if checkpoint_committed else "checkpoint_failed")
        return checkpoint_committed

    trainer._drain_checkpoint_upload = drain_upload
    trainer.handle_hf_export = lambda: events.append("hf_export")

    asyncio.run(trainer._run_step_end_callbacks(SimpleNamespace()))

    assert events == expected_events


def test_consumed_staleness_counts_selected_masked_sequences_by_group(
    monkeypatch, dummy_config, driver_trainer_factory
):
    events = []
    monkeypatch.setattr(
        trainer_module,
        "record_event",
        lambda name, values, attributes: events.append((name, values, attributes)),
    )
    trainer = driver_trainer_factory(dummy_config)
    trainer.global_step = 7

    trainer._record_consumed_staleness(["a", "b", "a"], [2, 0, 2], [2, 0, 3])

    assert [event[1] for event in events] == [
        {"staleness": 2, "groups": 1, "sequences": 2, "response_tokens": 5},
        {"staleness": 0, "groups": 1, "sequences": 1, "response_tokens": 0},
    ]
    assert all(event[2]["step"] == "7" for event in events)


@pytest.fixture
def dummy_tokenizer():
    # convert_to_training_input only pads with pad_token_id; any other tokenizer use should fail loudly.
    return SimpleNamespace(pad_token_id=0)


@pytest.mark.parametrize(
    "resume_path",
    [
        "s3://marin-us-east-02a/iris/test/checkpoints/global_step_12",
        "s3://marin-us-east-02a/iris/test/checkpoints/global_step_12/",
    ],
)
def test_load_checkpoints_probes_cloud_resume_uri_without_trailing_slash(
    dummy_config, monkeypatch, resume_path, driver_trainer_factory
):
    probed = []

    def exists(uri):
        probed.append(uri)
        return False

    monkeypatch.setattr(trainer_module.io, "exists", exists)
    dummy_config.trainer.resume_path = resume_path
    dummy_config.trainer.resume_mode = ResumeMode.FROM_PATH.value
    trainer = driver_trainer_factory(dummy_config)

    with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
        trainer.load_checkpoints()

    assert probed == ["s3://marin-us-east-02a/iris/test/checkpoints/global_step_12"]


def test_load_checkpoints_offloads_disaggregated_optimizer_before_policy_restore(
    dummy_config, tmp_path, monkeypatch, driver_trainer_factory
):
    checkpoint_path = tmp_path / "global_step_12"
    (checkpoint_path / trainer_module.POLICY_CHECKPOINT_SUBDIRECTORY).mkdir(parents=True)
    torch.save({"global_step": 12}, checkpoint_path / trainer_module.TRAINER_STATE_FILENAME)
    dummy_config.trainer.resume_path = str(checkpoint_path)
    dummy_config.trainer.offload_optimizer_during_rollouts = True

    dummy_config.trainer.resume_mode = ResumeMode.FROM_PATH.value
    trainer = driver_trainer_factory(dummy_config)
    trainer.policy_model = _CheckpointResidencyPolicyGroup()
    trainer.policy_model.model_on_gpu = True
    trainer.policy_model.optimizer_on_gpu = True
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)

    global_step, restored_path = trainer.load_checkpoints()

    assert global_step == 12
    assert restored_path == str(checkpoint_path)
    assert trainer.policy_model.restore_residencies == [(True, False)]
    assert "offload_policy_optimizer_before_checkpoint_load" in trainer.all_startup_timings


@pytest.mark.parametrize("restore_dataloader_state", [False, True])
def test_load_checkpoints_restores_rollout_state_only_when_requested(
    tmp_path, dummy_config, restore_dataloader_state, training_trainer_factory
):
    dataset = FixedPromptDataset([f"prompt-{index}" for index in range(8)])
    checkpoint_path = tmp_path / "global_step_12"
    checkpoint_path.mkdir()
    torch.save({"global_step": 12}, checkpoint_path / "trainer_state.pt")
    rollout_state = TrainingContextState(
        loader=PromptLoaderState(order={"epoch": 0, "position": 7}, retries=dataset.collate_fn(["prompt-3"])),
        ready=[],
        object_store_root=None,
    )
    torch.save(rollout_state, checkpoint_path / "data.pt")

    dummy_config.trainer.resume_path = str(checkpoint_path)
    dummy_config.trainer.restore_dataloader_state = restore_dataloader_state
    dummy_config.trainer.resume_mode = ResumeMode.FROM_PATH.value
    dummy_config.trainer.eval_interval = dummy_config.trainer.ckpt_interval = dummy_config.trainer.hf_save_interval = -1
    dummy_config.trainer.eval_before_train = False
    dummy_config.trainer.algorithm.use_kl_loss = False
    restored = []

    class RestoreObserver(TrainerCallback):
        async def on_train_begin_async(self, state, control, *, trainer, **kwargs):
            restored.append(await trainer.context.state_dict())
            control.should_training_stop = True
            return control

    trainer = training_trainer_factory(
        dummy_config,
        dataset=dataset,
        runner=FixedRolloutRunner(),
        tokenizer=SimpleNamespace(decode=str, pad_token_id=0),
        callbacks=[RestoreObserver()],
    )

    async def resume():
        initial_state = await trainer.context.state_dict()
        await trainer.train()
        return initial_state

    initial_state = asyncio.run(resume())

    assert trainer.global_step == 12
    assert trainer.loaded_checkpoint_path == str(checkpoint_path)
    assert restored == [rollout_state if restore_dataloader_state else initial_state]


@pytest.mark.parametrize(
    ("reset_stage_state", "expected_step", "expected_tokens"),
    [(False, 12, 900), (True, 0, 0)],
)
def test_load_checkpoints_can_start_a_new_stage_with_continued_model_training_state(
    tmp_path,
    dummy_config,
    monkeypatch,
    reset_stage_state,
    expected_step,
    expected_tokens,
    driver_trainer_factory,
):
    checkpoint_path = tmp_path / "global_step_12"
    checkpoint_path.mkdir()
    torch.save(
        {"global_step": 12, "distillation_scored_tokens_total": 900},
        checkpoint_path / "trainer_state.pt",
    )
    dummy_config.trainer.resume_path = str(checkpoint_path)
    dummy_config.trainer.restore_dataloader_state = False
    dummy_config.trainer.reset_global_step_on_resume = reset_stage_state
    dummy_config.trainer.reset_distillation_token_count_on_resume = reset_stage_state
    dummy_config.trainer.resume_mode = ResumeMode.FROM_PATH.value
    trainer = driver_trainer_factory(dummy_config)
    trainer.policy_model = _CheckpointResidencyPolicyGroup()
    trainer.colocate_all = True

    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)

    global_step, _ = trainer.load_checkpoints()

    assert global_step == expected_step
    assert trainer.distillation_scored_tokens_total == expected_tokens
    assert trainer.policy_model.restored_dirs == [str(checkpoint_path / "policy")]


def test_reward_kl_penalty_reports_masked_kl_metrics(dummy_config, driver_trainer_factory):
    trainer = driver_trainer_factory(dummy_config)
    data = TrainingInputBatch(
        {
            "loss_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.int32),
            "base_action_log_probs": torch.log(
                torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2], [0.25, 0.25, 0.25, 0.15, 0.10]])
            ),
            "action_log_probs": torch.log(torch.tensor([[0.1, 0.3, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.1, 0.1]])),
            "rewards": torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]),
        }
    )
    data.metadata = {}

    metrics = trainer.apply_reward_kl_penalty(data).metadata["metrics"]

    assert metrics["avg_kl_max"] == pytest.approx(0.3143, abs=1e-4)
    # The raw KL mean is 0.054; masking out the unscored tokens raises it.
    assert metrics["avg_kl"] == pytest.approx(0.1249, abs=1e-4)


@pytest.mark.parametrize(
    ("loss_reduction", "expected_policy_loss"),
    [("token_mean", 0.05), ("sequence_mean", 0.04375)],
)
def test_grpo_loop_credit_is_token_local_when_every_group_member_has_the_same_outcome(
    loss_reduction, expected_policy_loss
):
    response_length = 48
    algorithm = OmegaConf.create(
        {
            "advantage_estimator": "grpo",
            "gamma": 1.0,
            "lambd": 1.0,
            "grpo_norm_by_std": True,
            "policy_loss_type": "regular",
            "loss_reduction": loss_reduction,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "max_seq_len": response_length,
        }
    )
    loop_start = 18
    response_mask = torch.ones(4, response_length)
    response_mask[2:, 24:] = 0
    loop_advantages = torch.zeros(4, response_length)
    loop_advantages[:, loop_start:] = -0.1
    loop_advantages *= response_mask
    data = TrainingInputBatch(
        {
            "rewards": torch.zeros(4, response_length),
            "response_mask": response_mask,
            "values": None,
            "loop_advantages": loop_advantages,
            "loss_mask": response_mask,
        }
    )
    data.metadata = {
        "uids": ["same-group"] * 4,
        "avg_response_length": float(response_length),
    }

    data["advantages"], data["returns"] = compute_advantages_and_returns(
        token_level_rewards=data["rewards"],
        response_mask=data["response_mask"],
        index=data.metadata["uids"],
        adv_estimator=algorithm.advantage_estimator,
        config=algorithm,
    )
    result = trainer_module.normalize_advantages_dict(data)
    result = apply_loop_advantages(result)

    assert torch.equal(result["advantages"], loop_advantages)
    assert torch.equal(result["returns"], torch.zeros(4, response_length))
    inputs = PolicyLossInputs(
        torch.zeros_like(loop_advantages), torch.zeros_like(loop_advantages), None, result["advantages"], response_mask
    )
    token_loss = ppo_policy_loss(inputs, algorithm)
    counts = step_counts(
        [response_mask], [response_mask], [], [result["advantages"]], response_length, lambda value: value
    )
    policy_loss = reduce_to_step(
        token_loss.values,
        response_mask,
        counts.policy,
        LossReduction(loss_reduction),
        max_seq_len=response_length,
        nonzero_advantage_rows=counts.nonzero_advantage_rows,
    )
    assert policy_loss.item() == pytest.approx(expected_policy_loss)


def test_replace_mode_rejects_batch_without_teacher_evidence(
    dummy_config, local_distillation_config, driver_trainer_factory
):
    cfg = local_distillation_config(dummy_config)
    cfg.trainer.algorithm.distillation.reward_mode = "replace"
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.off_policy_correction = "none"
    trainer = driver_trainer_factory(cfg)
    trainer.policy_model = _ForwardPolicyGroup()
    trainer.global_step = 1
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "response_mask": torch.ones(1, 2, dtype=torch.long),
            "rollout_staleness": torch.zeros(1, dtype=torch.int32),
            "loss_mask": torch.ones(1, 2),
            "rollout_logprobs": None,
        }
    )
    batch.metadata = {"response_length": 2, "global_step": 1, "uids": ["a"]}

    async def train_supplied_batch():
        source = trainer.driver_batch_source()
        try:
            await source.prepare_from_batch(batch, diagnostics=trainer.batch_diagnostics())
            await source.forward(step_wall=None)
            with pytest.raises(ValueError, match="replace requires teacher evidence on every training batch"):
                await source.finalize(step_wall=None)
        finally:
            await source.release()

    asyncio.run(train_supplied_batch())


@pytest.mark.parametrize("reward_mode", ["add", "replace"])
@pytest.mark.parametrize("clip", [None, 0.5])
def test_teacher_credit_follows_environment_normalization_and_loop_credit(
    dummy_config, local_distillation_config, reward_mode, clip, driver_trainer_factory
):
    config = local_distillation_config(dummy_config)
    config.trainer.algorithm.advantage_batch_normalize = reward_mode == "add"
    config.trainer.algorithm.advantage_estimator = "uniform" if reward_mode == "replace" else "grpo"
    config.trainer.algorithm.distillation.reward_mode = reward_mode
    config.trainer.algorithm.distillation.advantage_clip = clip
    trainer = driver_trainer_factory(config)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    valid = torch.tensor([[True, False, False, True]])
    if reward_mode == "replace":
        mask *= valid
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "action_log_probs": torch.tensor([[-1.0, -1.0, torch.nan, -1.0]], requires_grad=True),
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros(1, 4),
            "advantages": torch.tensor([[1.0, 3.0, torch.nan, 5.0]]),
            "rewards": torch.zeros(1, 4),
            "loop_advantages": torch.tensor([[-0.1, -0.2, 0.0, -0.3]]) if reward_mode == "add" else torch.zeros(1, 4),
            "response_mask": torch.ones(1, 4),
            "loss_mask": mask,
            "teacher_action_log_probs": torch.tensor([[-0.2, torch.nan, torch.nan, -2.5]], requires_grad=True),
            "teacher_valid_mask": valid,
            "distillation_loss_weights": torch.tensor([[0.3, torch.nan, torch.nan, 2.0]]),
        }
    )
    batch.metadata = {"uids": ["a"], "response_length": 4}

    result = trainer.finalize_advantages_for_training(batch)
    [experience] = list(TrainingBatchIterator(result, sample_batch_size=1))

    teacher = torch.tensor([[0.24, 0.0, 0.0, -3.0]]) if clip is None else torch.tensor([[0.15, 0.0, 0.0, -1.0]])
    normalized = torch.tensor([[-(1.5**0.5) - 0.1, -0.2, 0.0, 1.5**0.5 - 0.3]])
    expected = teacher if reward_mode == "replace" else normalized + teacher
    torch.testing.assert_close(experience.advantages, expected)
    assert not experience.advantages.requires_grad
    assert experience.distillation is None
    for name in (
        "teacher_action_log_probs",
        "teacher_valid_mask",
        "distillation_loss_weights",
        "loop_advantages",
        "rewards",
    ):
        assert name not in result
    assert trainer.all_metrics["distillation/teacher_advantage_mean"] == pytest.approx(teacher.sum().item() / 2)
    assert trainer.all_metrics["distillation/teacher_advantage_abs_mean"] == pytest.approx(
        teacher.abs().sum().item() / 2
    )
    assert trainer.all_metrics["distillation/teacher_advantage_clipped_fraction"] == (0 if clip is None else 1)
    assert trainer.all_metrics["distillation/valid_tokens"] == 2


def test_loop_advantages_are_collated_with_response_tokens(dummy_config, dummy_tokenizer):
    trajectory_batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 0.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": None,
        "loop_advantages": [[0.0, -0.1, -0.1], [-0.2]],
    }

    plan = plan_batch(
        batch_id=1,
        global_step=1,
        uids=["a", "b"],
        facts=RowFacts.from_batch(trajectory_batch),
        fields=BatchFields.from_batch(trajectory_batch),
        dp_size=1,
        rollout_staleness=[0, 0],
        num_experts=None,
    )
    batch = assemble_slice(
        plan,
        range(2),
        trajectory_batch,
        pad_token_id=dummy_tokenizer.pad_token_id,
        algorithm=dummy_config.trainer.algorithm,
    )

    torch.testing.assert_close(
        batch["loop_advantages"],
        torch.tensor([[0.0, -0.1, -0.1], [-0.2, 0.0, 0.0]]),
    )
    assert "teacher_action_log_probs" not in batch
    assert "teacher_valid_mask" not in batch
    assert "distillation_loss_weights" not in batch


def test_teacher_evidence_is_validated_and_collated_with_response_tokens(
    dummy_config,
    dummy_tokenizer,
    chosen_teacher_evidence,
    driver_trainer_factory,
):
    trainer = driver_trainer_factory(dummy_config, tokenizer=dummy_tokenizer)
    evidence = chosen_teacher_evidence
    distillation = ChosenTokenTeacherInput(
        teacher_action_log_probs=evidence.chosen_logprobs,
        valid_mask=evidence.valid_mask,
        loss_weights=torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]),
    )
    trajectory_batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 0.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": None,
        "trajectory_ids": [
            TrajectoryID(instance_id="math", repetition_id=0),
            TrajectoryID(instance_id="swe", repetition_id=0),
        ],
        "teacher_evidence": evidence,
        "distillation": distillation,
    }

    batch = trainer.convert_to_training_input(trajectory_batch, ["math", "swe"])

    torch.testing.assert_close(batch["teacher_action_log_probs"], evidence.chosen_logprobs, equal_nan=True)
    assert torch.equal(batch["teacher_valid_mask"], evidence.valid_mask)
    torch.testing.assert_close(batch["distillation_loss_weights"], distillation.loss_weights)


def test_topk_teacher_evidence_is_collated_without_dense_vocabulary_tensors(
    dummy_config, dummy_tokenizer, driver_trainer_factory
):
    trainer = driver_trainer_factory(dummy_config, tokenizer=dummy_tokenizer)
    valid_mask = torch.tensor([[True, True, True], [True, False, False]])
    evidence = TopKTeacherEvidence(
        trajectory_ids=("math_0", "swe_0"),
        route_ids=("math", "swe"),
        teacher_id="teacher-a",
        teacher_revision="teacher-revision",
        plan_version="mopd-v1",
        valid_mask=valid_mask,
        topk_indices=torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [-1, -1], [-1, -1]]]),
        topk_logprobs=torch.log(
            torch.tensor(
                [[[0.7, 0.2], [0.6, 0.2], [0.5, 0.1]], [[0.8, 0.15], [torch.nan, torch.nan], [torch.nan, torch.nan]]]
            )
        ),
        retained_mass=torch.tensor([[0.9, 0.8, 0.6], [0.95, torch.nan, torch.nan]]),
    )
    distillation = TeacherTopKInput(
        teacher_topk_indices=evidence.topk_indices,
        teacher_topk_logprobs=evidence.topk_logprobs,
        retained_mass=evidence.retained_mass,
        valid_mask=evidence.valid_mask,
        loss_weights=torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.0, 0.0]]),
    )
    trajectory_batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 0.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": None,
        "trajectory_ids": [TrajectoryID("math", 0), TrajectoryID("swe", 0)],
        "teacher_evidence": evidence,
        "distillation": distillation,
    }

    batch = trainer.convert_to_training_input(trajectory_batch, ["math", "swe"])

    assert "teacher_action_log_probs" not in batch
    torch.testing.assert_close(batch["teacher_topk_indices"], evidence.topk_indices)
    torch.testing.assert_close(batch["teacher_topk_logprobs"], evidence.topk_logprobs, equal_nan=True)
    torch.testing.assert_close(batch["teacher_retained_mass"], evidence.retained_mass, equal_nan=True)


@pytest.mark.parametrize(
    ("worker_class", "mini_batch_key", "mini_batch_size", "n_samples_per_prompt", "dp_size", "expected"),
    [
        (PolicyWorkerBase, "policy_mini_batch_size", 16, 2, 4, 8),
        (PolicyWorkerBase, "policy_mini_batch_size", 8, 1, 1, 8),
        (CriticWorkerBase, "critic_mini_batch_size", 8, 2, 4, 4),
        (CriticWorkerBase, "critic_mini_batch_size", 32, 4, 2, 64),
    ],
)
def test_normalize_mini_batch_size_splits_prompt_samples_across_data_parallel_ranks(
    worker_class, mini_batch_key, mini_batch_size, n_samples_per_prompt, dp_size, expected
):
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "trainer": {mini_batch_key: mini_batch_size},
                "generator": {"n_samples_per_prompt": n_samples_per_prompt},
            }
        ),
        mesh_rank=MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1),
    )

    worker_class._normalize_mini_batch_size(worker)

    assert getattr(worker, f"{mini_batch_key}_per_gpu") == expected


@pytest.fixture(scope="module")
def default_config():
    return get_default_config()


def _batch_size_config(
    default_config,
    *,
    train_batch_size=128,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=2,
    n_samples_per_prompt=2,
    policy_dp=4,
    ref_dp=None,
):
    cfg = copy.deepcopy(default_config)
    cfg.trainer.train_batch_size = train_batch_size
    cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = policy_dp
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = ref_dp or 1
    cfg.trainer.algorithm.use_kl_loss = ref_dp is not None
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.generator.n_samples_per_prompt = n_samples_per_prompt
    return cfg


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({}, None),
        # A mini-batch larger than the train batch fails the first (unmessaged) assertion.
        ({"train_batch_size": 8, "policy_mini_batch_size": 16}, "^$"),
        ({"train_batch_size": 100}, "train_batch_size .* should be divisible by policy_mini_batch_size"),
        (
            {
                "policy_mini_batch_size": 8,
                "n_samples_per_prompt": 1,
                "policy_dp": 1,
                "micro_train_batch_size_per_gpu": 3,
            },
            "policy_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
        ),
        (
            {
                "train_batch_size": 10,
                "policy_mini_batch_size": 5,
                "policy_dp": 2,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
            },
            "policy_train_batch_size_per_gpu .* should be divisible by policy_mini_batch_size_per_gpu",
        ),
        # With a reference model, the batch must cover lcm(policy_dp, ref_dp) ranks; without one, only policy_dp.
        (
            {
                "train_batch_size": 5,
                "policy_mini_batch_size": 5,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
                "policy_dp": 2,
                "ref_dp": 3,
            },
            "least common multiple of the data parallel sizes",
        ),
        (
            {
                "train_batch_size": 6,
                "policy_mini_batch_size": 6,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
                "policy_dp": 2,
                "ref_dp": 3,
            },
            None,
        ),
        (
            {
                "train_batch_size": 2,
                "policy_mini_batch_size": 2,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
                "policy_dp": 2,
            },
            None,
        ),
    ],
)
def test_validate_batch_sizes_requires_even_division_across_ranks(default_config, overrides, error):
    cfg = _batch_size_config(default_config, **overrides)

    if error is None:
        validate_batch_sizes(cfg)
        return
    with pytest.raises(AssertionError, match=error):
        validate_batch_sizes(cfg)


def test_grpo_reports_one_flat_and_one_varied_reward_group():
    rewards = torch.tensor([[1.0], [1.0], [0.0], [2.0]])
    uids = ["easy", "easy", "hard", "hard"]
    advantages, _ = compute_advantages_and_returns(
        token_level_rewards=rewards,
        response_mask=torch.ones(4, 1),
        index=uids,
        adv_estimator="grpo",
        config=OmegaConf.create({}),
    )

    assert zero_std_group_fraction(uids, rewards.sum(-1)) == pytest.approx(0.5)
    assert torch.equal(advantages[:2], torch.zeros(2, 1))
    assert torch.isfinite(advantages).all()
