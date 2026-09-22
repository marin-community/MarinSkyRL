import asyncio

import pytest

from skyrl_train.entrypoints import terminal_bench_generate
from skyrl_train.entrypoints.terminal_bench_generate import TerminalBenchGenerateExp


def test_terminal_bench_generate_uses_shared_evaluation_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = object.__new__(TerminalBenchGenerateExp)
    expected = {"eval/all/avg_score": 0.5}

    async def fake_run_evaluation_only(candidate):
        await asyncio.sleep(0)
        assert candidate is experiment
        return expected

    monkeypatch.setattr(terminal_bench_generate, "run_evaluation_only", fake_run_evaluation_only)

    assert experiment.get_train_dataset() is None
    assert experiment.run() == expected
