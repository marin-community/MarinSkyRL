import asyncio
from types import SimpleNamespace
from torchdata.stateful_dataloader import StatefulDataLoader

import pytest

from skyrl_train.evaluation_pool import EvaluationPool
from skyrl_train.config.utils import get_default_config
from skyrl_train.evaluate import evaluate, evaluation_session
from tests.cpu.evaluation_fakes import EvaluationLifecycle
from tests.cpu.task_specs import lowered_task, task_input, text_task


@pytest.mark.asyncio
async def test_mixed_sources_share_slots_and_replenish_before_slow_tasks_finish():
    slow = asyncio.Event()
    ready = asyncio.Event()
    started = []
    active = set()
    maximum = 0
    sandbox_maximum = 0

    async def work(name):
        nonlocal maximum, sandbox_maximum
        active.add(name)
        started.append(name)
        maximum = max(maximum, len(active))
        sandbox_maximum = max(sandbox_maximum, sum(item.startswith("sandbox") for item in active))
        if name in ("native-slow", "sandbox-slow"):
            await slow.wait()
        else:
            ready.set()
        active.remove(name)
        return name

    async with EvaluationPool(3) as pool:
        async with asyncio.TaskGroup() as tasks:
            results = [
                tasks.create_task(pool.run(source, lambda name=name: work(name), limit=limit))
                for source, name, limit in [
                    ("native", "native-slow", None),
                    ("sandbox", "sandbox-slow", 1),
                    ("sandbox", "sandbox-next", 1),
                    ("native", "native-fast", None),
                    ("native", "native-next", None),
                ]
            ]
            async with asyncio.timeout(5):
                await ready.wait()
            assert "sandbox-next" not in started
            slow.set()
        assert sorted(result.result() for result in results) == [
            "native-fast",
            "native-next",
            "native-slow",
            "sandbox-next",
            "sandbox-slow",
        ]
    assert maximum == 3
    assert sandbox_maximum == 1


@pytest.mark.asyncio
async def test_pool_reports_failures_and_releases_capacity():
    async def fail():
        raise ConnectionError("provider failed")

    async def success():
        return "scored"

    async with EvaluationPool(1) as pool:
        with pytest.raises(ConnectionError, match="provider failed"):
            await pool.run("native", fail)
        assert await pool.run("harbor", success) == "scored"


@pytest.mark.asyncio
async def test_native_evaluation_preserves_every_row_and_repetition_without_batch_barriers():
    seen = []
    later_batch_started = asyncio.Event()

    class Runner(EvaluationLifecycle):
        async def run(self, request):
            uid = request.metadata[0].uid
            if uid == "slow":
                async with asyncio.timeout(5):
                    await later_batch_started.wait()
            else:
                later_batch_started.set()
            assert request.batch_metadata.training_phase == "eval"
            assert request.batch_metadata.global_step == 7
            seen.extend((item.instance_id, item.repetition_id) for item in request.trajectory_ids)
            count = len(request.trajectory_ids)
            return {
                "prompt_token_ids": [[1]] * count,
                "response_ids": [[2]] * count,
                "loss_masks": [[1]] * count,
                "rewards": [float(uid == "fast")] * count,
                "rollout_logprobs": None,
                "rollout_metrics": {},
                "trajectory_ids": list(request.trajectory_ids),
            }

    runner = Runner()
    cfg = get_default_config()
    cfg.generator.eval_n_samples_per_prompt = 2
    cfg.trainer.dump_eval_results = False
    cfg.trajectory_runner.max_concurrent_tasks = 4
    rows = [[{"input": task_input(lowered_task(text_task(uid)))}] for uid in ("slow", "fast")]
    loader = StatefulDataLoader([batch[0] for batch in rows], batch_size=1, collate_fn=list)
    sink = object()
    async with evaluation_session(runner, cfg, 7, sink) as pool:
        results = await asyncio.gather(
            *(
                evaluate(
                    loader,
                    runner,
                    cfg,
                    7,
                    SimpleNamespace(decode=lambda _: "response"),
                    val_set_name=name,
                    trajectory_sink=sink,
                    evaluation_pool=pool,
                )
                for name in (None, "sampled", "native")
            )
        )
    assert len(seen) == len(set(seen)) == 12
    assert set(seen) == {
        (f"eval-7-{len(name)}-{name}-{uid}", repetition)
        for name in ("", "sampled", "native")
        for uid in ("slow", "fast")
        for repetition in (0, 1)
    }
    assert all(metrics["eval/all/num_attempted"] == 4 for metrics in results)
    assert all(metrics["eval/all/pass_at_2"] == 0.5 for metrics in results)
    assert runner.events == ["started", "stopped"]
