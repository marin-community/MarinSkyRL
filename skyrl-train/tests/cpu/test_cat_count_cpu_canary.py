import argparse
import asyncio
import hashlib
import json
import math
import multiprocessing
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from multiprocessing.context import ForkServerContext

import pytest
import skyrl_gym
import skyrl_train
import torch
import zstandard
from examples.cat_count.cpu_canary import PROMPT, pretrain

from tests.cpu.tiny_training.cat_count import FAST_STEPS, cat_count_config, run_cat_count
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.experiment import read_metrics

pytestmark = pytest.mark.slow
RUN_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="session")
def cat_count_policy(pytestconfig) -> Path:
    # pytest's cache is portable and reusable across CI invocations; this fixture does no RL.
    parameters = argparse.Namespace(steps=3000, lr=3e-4, width=128, layers=2, seed=0)
    repository = Path(__file__).resolve().parents[3]
    sources = (
        "skyrl-train/examples/cat_count/cpu_canary.py",
        "skyrl-gym/skyrl_gym/envs/cat_count/reward.py",
        "skyrl-train/tests/cpu/test_cat_count_cpu_canary.py",
        "pyproject.toml",
        "skyrl-gym/pyproject.toml",
        "uv.lock",
    )
    digest = hashlib.sha256(json.dumps(vars(parameters), sort_keys=True).encode())
    for source in sources:
        digest.update(source.encode())
        digest.update((repository / source).read_bytes())
    identity = digest.hexdigest()[:16]
    directory = Path(pytestconfig.cache.mkdir("cat_count_policy")) / f"llama-{identity}"
    parameters.out = directory
    if not all((directory / name).exists() for name in ("model.safetensors", "config.json", "tokenizer.json")):
        torch.set_num_threads(1)
        started = time.perf_counter()
        pretrain(parameters)
        print(f"CAT_COUNT_PRETRAIN cache=cold seconds={time.perf_counter() - started:.3f} identity={identity}")
    else:
        print(f"CAT_COUNT_PRETRAIN cache=warm identity={identity}")
    return directory


@pytest.fixture
def cat_count_session(monkeypatch):
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
        monkeypatch.setenv(key, value)
    try:
        yield rows
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(scope="module")
def runs() -> ForkServerContext:
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(["tests.cpu.tiny_training.cat_count"])
    return context


def train(
    runs: ForkServerContext,
    root: Path,
    model: Path,
    *,
    flipped: bool = False,
    steps: int = FAST_STEPS,
    staleness: int = 0,
    resume: bool = False,
    eval_interval: int | None = None,
    seed: int = 0,
):
    cfg = cat_count_config(
        root, model, steps=steps, staleness=staleness, resume=resume, eval_interval=eval_interval, seed=seed
    )
    if flipped:
        cfg.trainer.algorithm.advantage_estimator = "cat_count_flipped_grpo"
    worker_env = {key: os.environ[key] for key in ("SKYRL_TELEMETRY_ENDPOINT", "SKYRL_RUN_ID", "SKYRL_EXECUTION_UID")}
    start = time.perf_counter()
    process = runs.Process(target=run_cat_count, args=(cfg, worker_env))
    process.start()
    process.join(RUN_TIMEOUT_SECONDS)
    if process.exitcode is None:
        process.kill()
        process.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert process.exitcode == 0
    print(f"CAT_COUNT_CPU run={root.name} seconds={time.perf_counter() - start:.3f}")
    return read_metrics(root)


def scores(records):
    evaluations = [row for row in records if "eval/train/avg_score" in row]
    return [(row["eval/train/avg_score"], row["eval/heldout/avg_score"]) for row in evaluations]


@pytest.mark.asyncio
async def test_cpu_sampling_preserves_trajectory_rng_and_minimum_tokens(cat_count_policy):
    engine = CPUInferenceEngine(str(cat_count_policy), seed=0)
    tokenizer = engine.tokenizer
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": PROMPT.format(N=7),
            }
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    sampling = {"temperature": 1.0, "max_tokens": 64, "min_tokens": 0, "logprobs": 0}
    try:
        batch = await engine.generate(
            {"prompt_token_ids": [prompt] * 3, "session_ids": ["a", "b", "c"], "sampling_params": sampling}
        )
        reordered = await asyncio.gather(
            *(
                engine.generate({"prompt_token_ids": [prompt], "session_ids": [identity], "sampling_params": sampling})
                for identity in ("c", "b", "a")
            )
        )
        for index, result in enumerate(reversed(reordered)):
            assert result["response_ids"][0] == batch["response_ids"][index]
            assert result["response_logprobs"][0] == pytest.approx(batch["response_logprobs"][index], abs=1e-5)
        assert len({tuple(ids) for ids in batch["response_ids"]}) > 1
        for index, identity in enumerate(("a", "b", "c")):
            singleton = await engine.generate(
                {"prompt_token_ids": [prompt], "session_ids": [identity], "sampling_params": sampling}
            )
            assert singleton["response_ids"][0] == batch["response_ids"][index]

        eos_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": PROMPT.format(N=1),
                }
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        ) + tokenizer.encode("cat", add_special_tokens=False)
        mixed = await engine.generate(
            {"prompt_token_ids": [eos_prompt, prompt], "session_ids": ["eos-short", "a"], "sampling_params": sampling}
        )
        assert mixed["response_ids"][0] == [tokenizer.eos_token_id]
        assert mixed["response_ids"][1] == batch["response_ids"][0]
        for minimum in (0, None, 3):
            params = {"temperature": 0.0, "max_tokens": 8, "logprobs": 0}
            if minimum is not None:
                params["min_tokens"] = minimum
            result = await engine.generate(
                {"prompt_token_ids": [eos_prompt], "session_ids": ["eos"], "sampling_params": params}
            )
            ids = result["response_ids"][0]
            if minimum == 0:
                assert ids == [tokenizer.eos_token_id]
            else:
                assert len(ids) >= (minimum or 1)
                assert tokenizer.eos_token_id not in ids[: minimum or 1]
            assert all(math.isfinite(value) for value in result["response_logprobs"][0])
    finally:
        await engine.teardown()


@pytest.mark.parametrize(
    "seed", [0, pytest.param(1, marks=pytest.mark.nightly), pytest.param(2, marks=pytest.mark.nightly)]
)
def test_cat_count_cpu_learns_and_flipped_advantage_fails(tmp_path, cat_count_policy, cat_count_session, runs, seed):
    positive = train(runs, tmp_path / "positive", cat_count_policy, seed=seed)
    negative = train(runs, tmp_path / "negative", cat_count_policy, flipped=True, seed=seed)
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
    print(
        f"CAT_COUNT_CPU seed={seed} paired_scores positive={before}->{after} negative={negative_before}->{negative_after}"
    )


@pytest.mark.nightly
def test_cat_count_async_resumes_and_converges(tmp_path, cat_count_policy, cat_count_session, runs):
    root = tmp_path / "async"
    first = train(runs, root, cat_count_policy, steps=4, staleness=1)
    resumed = train(runs, root, cat_count_policy, steps=100, staleness=1, resume=True, eval_interval=2)
    assert scores(resumed)[len(scores(first))] == scores(first)[-1]
    assert len([row for row in first if "policy/raw_grad_norm" in row]) == 4
    training = [row for row in resumed if "policy/raw_grad_norm" in row]
    assert [row["trainer/global_step"] for row in training] == list(range(1, 101))
    assert any(row["async/staleness_mean"] > 0 for row in training)
    assert any(train_score >= 0.9 and heldout_score >= 0.9 for train_score, heldout_score in scores(resumed))
    checkpoint = torch.load(root / "ckpts/global_step_100/policy/rank_0.pt", weights_only=False)
    assert {state["step"].item() for state in checkpoint["optimizer"]["state"].values()} == {200}
