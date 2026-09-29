import argparse
import hashlib
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import ray
import skyrl_gym
import skyrl_train
import torch
import zstandard
from examples.cat_count.cpu_canary import pretrain
from skyrl_train.config.trajectory_runner_capabilities import (
    TrajectoryRunnerMode,
    validate_trajectory_runner_capabilities,
)
from skyrl_train.entrypoints.main_base import EntrypointOperation
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.algorithm_registry import AdvantageEstimatorRegistry

from tests.cpu.tiny_training.cat_count import FAST_STEPS, cat_count_config, flipped_grpo
from tests.cpu.tiny_training.experiment import (
    LOGICAL_CPUS,
    LOGICAL_GPUS,
    WORKER_ENV_VARS,
    TinyTrainingExp,
    read_metrics,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def cat_count_policy(pytestconfig) -> Path:
    # pytest's cache is portable and reusable across CI invocations; this fixture does no RL.
    parameters = argparse.Namespace(steps=3000, lr=3e-4, width=128, layers=2, seed=0)
    identity = hashlib.sha256(json.dumps(vars(parameters), sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(pytestconfig.cache.mkdir("cat_count_policy")) / f"llama-{identity}"
    parameters.out = directory
    if not all((directory / name).exists() for name in ("model.safetensors", "config.json", "tokenizer.json")):
        torch.set_num_threads(1)
        pretrain(parameters)
    return directory


@pytest.fixture(scope="module")
def cat_count_session(monkeypatch_module):
    repository = Path(__file__).resolve().parents[3]
    assert Path(skyrl_train.__file__).resolve().is_relative_to(repository / "skyrl-train")
    assert Path(skyrl_gym.__file__).resolve().is_relative_to(repository / "skyrl-gym")
    rows = []

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "zstd":
                payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=1 << 28)
            batch = json.loads(payload)
            rows.extend(batch["records"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"batch_id": batch["batch_id"], "status": "accepted"}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    telemetry_env = {
        "SKYRL_TELEMETRY_ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1/ingest",
        "SKYRL_RUN_ID": "cat-count-cpu",
        "SKYRL_EXECUTION_UID": "cat-count-cpu-attempt",
    }
    for key, value in telemetry_env.items():
        monkeypatch_module.setenv(key, value)
    ray.init(
        num_cpus=LOGICAL_CPUS,
        num_gpus=LOGICAL_GPUS,
        include_dashboard=False,
        runtime_env={"env_vars": {**WORKER_ENV_VARS, **telemetry_env}},
    )
    AdvantageEstimatorRegistry.register(
        "cat_count_flipped_grpo", flipped_grpo, group_contract=AdvantageEstimatorRegistry.group_contract("grpo")
    )
    try:
        yield rows
    finally:
        AdvantageEstimatorRegistry.unregister("cat_count_flipped_grpo")
        ray.shutdown()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def train(
    root: Path,
    model: Path,
    *,
    flipped: bool = False,
    steps: int = FAST_STEPS,
    staleness: int = 0,
    resume: bool = False,
    eval_interval: int | None = None,
):
    cfg = cat_count_config(root, model, steps=steps, staleness=staleness, resume=resume, eval_interval=eval_interval)
    if flipped:
        cfg.trainer.algorithm.advantage_estimator = "cat_count_flipped_grpo"
    validate_cfg(cfg)
    validate_trajectory_runner_capabilities(cfg, TrajectoryRunnerMode.SKYRL_GYM, EntrypointOperation.TRAIN)
    start = time.perf_counter()
    TinyTrainingExp(cfg).run()
    print(f"CAT_COUNT_CPU run={root.name} seconds={time.perf_counter() - start:.3f}")
    return read_metrics(root)


def scores(records):
    evaluations = [row for row in records if "eval/train/avg_score" in row]
    return [(row["eval/train/avg_score"], row["eval/heldout/avg_score"]) for row in evaluations]


def test_cat_count_cpu_learns_and_flipped_advantage_fails(tmp_path, cat_count_policy, cat_count_session):
    positive = train(tmp_path / "positive", cat_count_policy)
    negative = train(tmp_path / "negative", cat_count_policy, flipped=True)
    before, after = scores(positive)
    negative_before, negative_after = scores(negative)
    assert before == negative_before
    assert sum(after) / 2 >= sum(before) / 2 + 0.1
    assert all(final > initial for initial, final in zip(before, after, strict=True))
    assert sum(negative_after) <= sum(negative_before)
    assert (sum(after) - sum(negative_after)) / 2 >= 0.4
    training = [row for row in positive if "policy/raw_grad_norm" in row]
    assert len(training) == FAST_STEPS
    assert all(any(name.startswith("policy/tis/") for name in row) for row in training)
    assert all(math.isfinite(row["policy/policy_loss"]) for row in training)
    assert all(row["tis/skipped_fraction"] == 0 for row in training)
    assert any(row["policy/ppo_clip_ratio"] > 0 for row in training)
    assert all("environment/exact" in row for row in training)
    assert any("environment/exact_n20" in row for row in training)
    checkpoint = torch.load(tmp_path / f"positive/ckpts/global_step_{FAST_STEPS}/policy/rank_0.pt", weights_only=False)
    optimizer_steps = [state["step"].item() for state in checkpoint["optimizer"]["state"].values()]
    assert optimizer_steps and set(optimizer_steps) == {2 * FAST_STEPS}
    names = {row["name"] for row in cat_count_session}
    assert {"policy_step", "work_completed", "phase_duration_seconds", "weight_sync_completed"} <= names
    assert any(
        row["name"] == "training_metric_value" and row["attributes"].get("metric") == "environment/exact"
        for row in cat_count_session
    )
    print(f"CAT_COUNT_CPU paired_scores positive={before}->{after} negative={negative_before}->{negative_after}")


@pytest.mark.nightly
def test_cat_count_async_resumes_and_converges(tmp_path, cat_count_policy, cat_count_session):
    root = tmp_path / "async"
    first = train(root, cat_count_policy, steps=4, staleness=1)
    resumed = train(root, cat_count_policy, steps=100, staleness=1, resume=True, eval_interval=2)
    assert scores(resumed)[len(scores(first))] == scores(first)[-1]
    assert len([row for row in first if "policy/raw_grad_norm" in row]) == 4
    training = [row for row in resumed if "policy/raw_grad_norm" in row]
    assert [row["trainer/global_step"] for row in training] == list(range(1, 101))
    assert any(row["async/staleness_mean"] > 0 for row in training)
    assert any(train_score >= 0.9 and heldout_score >= 0.9 for train_score, heldout_score in scores(resumed))
    checkpoint = torch.load(root / "ckpts/global_step_100/policy/rank_0.pt", weights_only=False)
    assert {state["step"].item() for state in checkpoint["optimizer"]["state"].values()} == {200}
