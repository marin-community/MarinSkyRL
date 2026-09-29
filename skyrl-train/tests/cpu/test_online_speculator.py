"""Online EAGLE speculator lifecycle driven through RayPPOTrainer's step hooks.

The trainer talks to three remote parties: the inference engines (capture, seal, draft refresh),
the DraftTrainer actor (update requests), and the draft checkpoint store. Each is replaced by an
in-memory fake so the tests observe what the trainer sends and the metrics it reports.
"""

import asyncio
from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

import skyrl_train.trainer as trainer_module
from marinskyrl.speculative_decoding import SpeculativeDecodingConfig
from skyrl_train.draft_trainer import DraftCheckpoint
from skyrl_train.inference_engines.vllm.online_eagle_trainer import OnlineEagleUpdateResult
from skyrl_train.trainer import RayPPOTrainer

DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
CHECKPOINT_ROOT = "s3://bucket/checkpoints/drafts"

TWO_RANK_MANIFESTS = [
    [
        {"active": True, "worker_rank": 0, "captured_rows": 41, "dropped_windows": 2, "windows": [{"path": "w0"}]},
        {"active": True, "worker_rank": 1, "captured_rows": 43, "dropped_windows": 1, "windows": [{"path": "w0"}]},
    ]
]


def draft_checkpoint(uri: str) -> DraftCheckpoint:
    return DraftCheckpoint(
        step=2,
        revision="draft-step-2",
        uri=uri,
        weights_uri=f"{uri}/model.safetensors",
        weights_size=5,
        completion_uri=f"{uri}/complete.json",
        source_identity=DRAFT_REVISION,
    )


class FakeInferenceClient:
    def __init__(self):
        self.begins = []
        self.seals = []
        self.refreshes = []
        self.seal_result = TWO_RANK_MANIFESTS
        self.refresh_result = [{"active": True}]

    async def begin_online_eagle_capture(self, config):
        self.begins.append(config)
        return [[{"active": True, "worker_rank": 0}, {"active": True, "worker_rank": 1}]]

    async def seal_online_eagle_capture(self, output_root):
        self.seals.append(output_root)
        return self.seal_result

    async def update_draft_weights(self, weights_path):
        self.refreshes.append(weights_path)
        return self.refresh_result


@dataclass
class FakeRef:
    """Stands in for a Ray ObjectRef; the trainer's poll helper is patched to read these fields."""

    ready: bool
    value: object


class FakeDraftTrainer:
    def __init__(self):
        self.requests = []
        self.update = self

    def remote(self, request):
        self.requests.append(request)
        return FakeRef(
            ready=True,
            value=OnlineEagleUpdateResult(
                accepted=True,
                step=request.step,
                draft_revision=f"draft-step-{request.step}",
                candidate_uri=f"{CHECKPOINT_ROOT}/draft-step-{request.step}",
                parent_draft_revision="draft-step-1",
                trained_against_target_revision=request.target_revision,
                train_loss=0.5,
                incumbent_holdout_loss=0.4,
                candidate_holdout_loss=0.3,
            ),
        )


class FakeDraftStore:
    def __init__(self):
        self.latest: DraftCheckpoint | None = None
        self.error: Exception | None = None

    def read_latest(self, _root, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.latest


@pytest.fixture
def draft_store(monkeypatch) -> FakeDraftStore:
    store = FakeDraftStore()
    monkeypatch.setattr(trainer_module, "read_latest_draft_checkpoint", store.read_latest)
    monkeypatch.setattr(trainer_module, "_poll_object_ref", lambda ref: (ref.ready, ref.value if ref.ready else None))
    monkeypatch.setattr(trainer_module.io, "exists", lambda _uri: False)
    return store


def make_trainer(interval_steps: int = 1) -> RayPPOTrainer:
    """Build the trainer's speculator state as ``RayPPOTrainer.__init__`` does, without starting Ray."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.speculative_decoding = SpeculativeDecodingConfig.from_mapping(
        {
            "method": "eagle3",
            "model": {"source_uri": "hf://laion/snowball-64k-eagle3-draft-r2egym", "source_identity": DRAFT_REVISION},
            "num_speculative_tokens": 3,
            "training": {"interval_steps": interval_steps},
        }
    )
    trainer.global_step = 2
    trainer._speculator_capture_active = False
    trainer._speculator_revision = "draft-step-1"
    trainer._sealed_speculator_capture_uri = None
    trainer._speculator_checkpoint_root = CHECKPOINT_ROOT
    trainer._draft_trainer = FakeDraftTrainer()
    trainer._draft_trainer_update_ref = None
    trainer._draft_trainer_submitted_at = None
    trainer._speculator_refresh_task = None
    trainer._speculator_requested_revision = None
    trainer._speculator_update_failures = 0
    trainer._speculator_install_count = 0
    trainer._speculator_install_failures = 0
    trainer._speculator_checkpoint_poll_failures = 0
    trainer.inference_engine_client = FakeInferenceClient()
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.cfg = OmegaConf.create({"trainer": {"seed": 17}})
    return trainer


def test_capture_update_and_refresh_serve_the_accepted_draft(draft_store):
    trainer = make_trainer(interval_steps=2)
    client = trainer.inference_engine_client

    async def scenario():
        await trainer._begin_speculator_capture(2)
        await trainer._begin_speculator_capture(2)  # a repeated hook at the same step must not re-arm capture
        await trainer._seal_speculator_capture()
        await trainer._start_speculator_update()
        draft_store.latest = draft_checkpoint(f"{CHECKPOINT_ROOT}/draft-step-2")
        await trainer._poll_speculator_lifecycle()
        await trainer._refresh_latest_speculator(wait=True)
        trainer.global_step = 3
        await trainer._begin_speculator_capture(3)
        trainer.global_step = 4
        await trainer._begin_speculator_capture(4)

    asyncio.run(scenario())

    assert client.begins[0] == {
        "step": 2,
        "max_tokens": 16_384,
        "max_window_tokens": 16_384,
        "target_revision": "policy-step-1",
        "draft_revision": "draft-step-1",
        "reserved_gpu_memory_gib": 8,
    }
    assert client.seals == [f"{CHECKPOINT_ROOT}/captures/step-2"]
    [request] = trainer._draft_trainer.requests
    assert (request.capture_uri, request.target_revision, request.seed) == (
        f"{CHECKPOINT_ROOT}/captures/step-2",
        "policy-step-1",
        17,
    )
    assert client.refreshes == [f"{CHECKPOINT_ROOT}/draft-step-2/model.safetensors"]
    # Step 3 is off-cadence; the step-4 capture trains against the newly installed draft.
    assert [begin["step"] for begin in client.begins] == [2, 4]
    assert client.begins[1]["draft_revision"] == "draft-step-2"
    assert trainer.all_metrics["speculator/sealed_rows"] == 84.0
    assert trainer.all_metrics["speculator/sealed_windows"] == 2.0
    assert trainer.all_metrics["speculator/capture_dropped_windows"] == 3.0
    assert trainer.all_metrics["speculator/candidate_accepted"] == 1.0
    assert trainer.all_metrics["speculator/install_count"] == 1.0


def test_partial_capture_is_still_handed_to_the_draft_trainer(draft_store):
    trainer = make_trainer()
    trainer.inference_engine_client.seal_result = [
        [{"active": True, "worker_rank": 0, "captured_rows": 41, "windows": []}, {"active": False, "worker_rank": 1}]
    ]

    async def scenario():
        await trainer._begin_speculator_capture(2)
        await trainer._seal_speculator_capture()
        await trainer._start_speculator_update()

    asyncio.run(scenario())

    assert [request.capture_uri for request in trainer._draft_trainer.requests] == [
        f"{CHECKPOINT_ROOT}/captures/step-2"
    ]
    assert trainer.all_metrics["speculator/sealed_rows"] == 41.0


@pytest.mark.parametrize(
    ("update_result", "expected_pending", "expected_failures", "expected_begins"),
    [
        # A pending update keeps the incumbent draft and blocks a new capture.
        (FakeRef(ready=False, value=None), 1.0, 0.0, 0),
        # A malformed update result counts as a failure and frees the pipeline for the next capture.
        (FakeRef(ready=True, value=None), 0.0, 1.0, 1),
    ],
)
def test_outstanding_draft_update_gates_the_next_capture(
    draft_store, update_result, expected_pending, expected_failures, expected_begins
):
    trainer = make_trainer()
    trainer._draft_trainer_update_ref = update_result

    asyncio.run(trainer._begin_speculator_capture(2))

    assert len(trainer.inference_engine_client.begins) == expected_begins
    assert trainer.inference_engine_client.refreshes == []
    assert trainer.all_metrics["speculator/update_pending"] == expected_pending
    assert trainer.all_metrics["speculator/update_failures"] == expected_failures


def test_failed_draft_install_is_counted_and_retried(draft_store):
    trainer = make_trainer()
    # A checkpoint written for an earlier step still refreshes serving once it is the latest accepted draft.
    trainer.global_step = 5
    draft_store.latest = draft_checkpoint("gcs://bucket/checkpoints/drafts/draft-step-2")
    client = trainer.inference_engine_client
    client.refresh_result = [{"active": False, "error": "RuntimeError: draft load failed"}]

    asyncio.run(trainer._refresh_latest_speculator(wait=True))

    assert trainer.all_metrics["speculator/install_successful_engines"] == 0.0
    assert trainer.all_metrics["speculator/install_failed_engines"] == 1.0
    assert trainer.all_metrics["speculator/install_failures"] == 1.0
    assert trainer.all_metrics["speculator/install_count"] == 0.0

    client.refresh_result = [{"active": True}]
    asyncio.run(trainer._refresh_latest_speculator(wait=True))

    # vLLM loads through runai, which only understands the gs:// scheme.
    assert client.refreshes == ["gs://bucket/checkpoints/drafts/draft-step-2/model.safetensors"] * 2
    assert trainer.all_metrics["speculator/install_count"] == 1.0


@pytest.mark.parametrize(
    ("latest", "error", "expected_poll_failures"),
    [(None, None, None), (None, OSError("object store unavailable"), 1.0)],
)
def test_missing_or_unreadable_draft_checkpoint_keeps_the_initial_draft(
    draft_store, latest, error, expected_poll_failures
):
    trainer = make_trainer()
    draft_store.latest = latest
    draft_store.error = error

    asyncio.run(trainer._refresh_latest_speculator(wait=True))

    assert trainer.inference_engine_client.refreshes == []
    assert trainer.all_metrics.get("speculator/checkpoint_poll_failures") == expected_poll_failures
