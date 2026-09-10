import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.utils import measurement_guard
from tests.cpu.test_data_order import SyncSourceOrderDriver, resolve_cpu_actor_results
from tests.cpu.test_fully_async_publication_cadence import DriverWithCpuLearner, StartupLearnerService, make_driver
from tests.cpu.test_fully_async_n2 import n2_driver


URI = "s3://marin-us-east-02a/qualification/measurement-start.json"


class ConditionalStore:
    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()
        self.corrupt_read = False

    def call_s3(self, method, **kwargs):
        assert method == "put_object"
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["Bucket"] == "marin-us-east-02a"
        with self.lock:
            if kwargs["Key"] in self.objects:
                raise FileExistsError("Measurement already claimed")
            self.objects[kwargs["Key"]] = kwargs["Body"]

    def cat_file(self, path):
        return b"changed" if self.corrupt_read else self.objects[path]


@pytest.fixture
def store(monkeypatch):
    result = ConditionalStore()
    monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/fixture/0")
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "original-attempt")
    monkeypatch.setattr(measurement_guard, "fs_and_path", lambda uri: (result, "qualification/measurement-start.json"))
    return result


def test_duplicate_attempt_cannot_replace_first_claim(store, monkeypatch):
    first = measurement_guard.claim_measurement(URI)
    original = dict(store.objects)
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "retry-attempt")
    with pytest.raises(FileExistsError):
        measurement_guard.claim_measurement(URI)
    assert store.objects == original
    assert first["attempt_uid"] == "original-attempt"
    assert json.loads(next(iter(original.values())))["boundary"] == "before_initial_evaluation_or_training"


def test_atomic_creation_has_one_winner(store):
    def claim():
        try:
            measurement_guard.claim_measurement(URI)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(outcomes) == [False, True]


def test_readback_mismatch_is_fatal(store):
    store.corrupt_read = True
    with pytest.raises(RuntimeError, match="readback differs"):
        measurement_guard.claim_measurement(URI)


@pytest.mark.parametrize("uri", ["", "/tmp/marker", "gs://bucket/key", "s3://bucket", "s3://bucket/key?version=x"])
def test_unsupported_uri_is_rejected_before_storage(store, uri):
    with pytest.raises(ValueError, match="explicit S3"):
        measurement_guard.claim_measurement(uri)
    assert store.objects == {}


def test_native_identity_is_required(store, monkeypatch):
    monkeypatch.delenv("IRIS_ATTEMPT_UID")
    with pytest.raises(ValueError, match="native task and attempt"):
        measurement_guard.claim_measurement(URI)
    assert store.objects == {}


@pytest.mark.parametrize("driver_type", [SyncSourceOrderDriver, DriverWithCpuLearner])
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_driver_claim_precedes_initial_eval_and_training(store, monkeypatch, driver_type, enabled):
    trainer = make_driver(steps=1, interval=1, age=0, driver_type=driver_type)
    trainer.policy_model = StartupLearnerService()
    resolve_cpu_actor_results(monkeypatch)
    if enabled:
        trainer.cfg.trainer.measurement_guard_uri = URI
    original = measurement_guard.claim_measurement
    observations = []

    def claim(uri):
        observations.append(True)
        assert trainer.inference_engine_client.publications == [0]
        assert trainer.trajectory_runner.evaluations == []
        assert trainer.trajectory_runner.generations == []
        assert trainer.policy_model.completed_update == 0
        return original(uri)

    monkeypatch.setattr(measurement_guard, "claim_measurement", claim)
    asyncio.run(asyncio.wait_for(trainer._train_loop(), timeout=10))
    assert observations == ([True] if enabled else [])
    assert bool(store.objects) is enabled
    assert trainer.policy_model.completed_update == 1
    assert trainer.trajectory_runner.evaluations[0] == (0, 0)
    assert trainer.trajectory_runner.generations


@pytest.mark.parametrize("driver_type", [SyncSourceOrderDriver, DriverWithCpuLearner])
def test_actual_retry_after_claim_cannot_evaluate_or_train(store, monkeypatch, driver_type):
    measurement_guard.claim_measurement(URI)
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "new-attempt")
    trainer = make_driver(steps=1, interval=1, age=0, driver_type=driver_type)
    trainer.policy_model = StartupLearnerService()
    trainer.cfg.trainer.measurement_guard_uri = URI
    resolve_cpu_actor_results(monkeypatch)
    with pytest.raises(FileExistsError):
        asyncio.run(asyncio.wait_for(trainer._train_loop(), timeout=10))
    assert trainer.trajectory_runner.evaluations == trainer.trajectory_runner.generations == []
    assert trainer.policy_model.completed_update == 0


@pytest.mark.parametrize("driver_type", [SyncSourceOrderDriver, DriverWithCpuLearner])
def test_actual_preboundary_startup_failure_does_not_claim(store, monkeypatch, driver_type):
    trainer = make_driver(steps=1, interval=1, age=0, driver_type=driver_type)
    trainer.cfg.trainer.measurement_guard_uri = URI

    def fail():
        raise RuntimeError("startup failed")

    monkeypatch.setattr(trainer, "init_weight_sync_state", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        asyncio.run(asyncio.wait_for(trainer._train_loop(), timeout=10))
    assert store.objects == {}
    assert trainer.trajectory_runner.evaluations == trainer.trajectory_runner.generations == []


@pytest.mark.parametrize("step", [None, 0, -1, True, 1.0, "1", 2])
def test_continuation_guard_rejects_missing_or_wrong_resume_step_before_claim(store, step):
    trainer = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "trainer": {
                    "measurement_guard_uri": URI,
                    "measurement_guard_resume_step": step,
                    "resume_path": "/bound/global_step_1",
                }
            }
        ),
        resume_mode=ResumeMode.FROM_PATH,
        global_step=1,
    )
    with pytest.raises(ValueError):
        RayPPOTrainer._claim_measurement_boundary(trainer)
    assert store.objects == {}


@pytest.mark.parametrize(
    "mode,path,uri",
    [
        (ResumeMode.LATEST, "/bound/global_step_1", URI),
        (ResumeMode.NONE, "/bound/global_step_1", URI),
        (ResumeMode.FROM_PATH, None, URI),
        (ResumeMode.FROM_PATH, "", URI),
        (ResumeMode.FROM_PATH, "/bound/global_step_1", None),
    ],
)
def test_continuation_requires_explicit_path_mode_and_own_marker(store, mode, path, uri):
    trainer = SimpleNamespace(
        cfg=OmegaConf.create(
            {"trainer": {"measurement_guard_uri": uri, "measurement_guard_resume_step": 1, "resume_path": path}}
        ),
        resume_mode=mode,
        global_step=1,
    )
    with pytest.raises(ValueError):
        RayPPOTrainer._claim_measurement_boundary(trainer)
    assert store.objects == {}


@pytest.mark.asyncio
async def test_actual_n2_continuation_claim_keeps_saved_partition_and_blocks_duplicate(store, tmp_path, monkeypatch):
    trainer = n2_driver(stop_step=1, save_step=1)
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    try:
        await asyncio.wait_for(trainer._train_loop(), timeout=15)
    finally:
        await trainer._cancel_trajectory_tasks()

    def continuation():
        resumed = n2_driver()
        resumed.resume_mode = ResumeMode.FROM_PATH
        resumed.cfg.trainer.resume_path = str(tmp_path / "global_step_1")
        resumed.cfg.trainer.measurement_guard_uri = URI
        resumed.cfg.trainer.measurement_guard_resume_step = 1
        return resumed

    resumed = continuation()
    original_claim = measurement_guard.claim_measurement
    claims = []

    def claim(uri):
        assert resumed.global_step == 1
        assert resumed.inference_engine_client.publications == [1]
        assert resumed.trajectory_runner.evaluations == resumed.trajectory_runner.generations == []
        claims.append(uri)
        return original_claim(uri)

    monkeypatch.setattr(measurement_guard, "claim_measurement", claim)
    try:
        await asyncio.wait_for(resumed._train_loop(), timeout=15)
    finally:
        await resumed._cancel_trajectory_tasks()
    assert claims == [URI]
    assert [step for step, _ in resumed.preparations] == [3]
    assert resumed.inputs[0]["action_log_probs"].unique().tolist() == pytest.approx([-0.1])
    assert resumed.inputs[0].metadata["async_cohort_update_index"] == 1
    monkeypatch.setattr(measurement_guard, "claim_measurement", original_claim)
    duplicate = continuation()
    try:
        with pytest.raises(FileExistsError):
            await asyncio.wait_for(duplicate._train_loop(), timeout=15)
    finally:
        await duplicate._cancel_trajectory_tasks()
    assert duplicate.trajectory_runner.evaluations == duplicate.trajectory_runner.generations == []
    assert duplicate.inputs == []


def test_native_checkpoint_loader_selects_explicit_seven_instead_of_latest_eight(tmp_path, monkeypatch):
    checkpoint = tmp_path / "global_step_7"
    checkpoint.mkdir()
    (tmp_path / "latest_ckpt_global_step.txt").write_text("8")
    trainer = SimpleNamespace(
        cfg=OmegaConf.create({"trainer": {"resume_path": str(checkpoint), "ckpt_path": str(tmp_path)}}),
        resume_mode=ResumeMode.FROM_PATH,
    )
    observed = []

    def exists(path):
        observed.append(path)
        return path == str(checkpoint)

    monkeypatch.setattr("skyrl_train.trainer.io.exists", exists)
    with pytest.raises(FileNotFoundError, match="Trainer state file not found"):
        RayPPOTrainer.load_checkpoints(trainer)
    assert observed == [str(checkpoint), str(checkpoint / "trainer_state.pt")]
