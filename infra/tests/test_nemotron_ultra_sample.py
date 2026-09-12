import json

import pytest

from infra.rl_data.nemotron_ultra_sample import BlendSampleSpec, _range_rows, sample_raw_generator_rows
from infra.rl_data.sources import NEMOTRON_ULTRA_SWE_AGENT, nemotron_ultra_rlvr1_source


class _Response:
    def __init__(self, *, status_code=206, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_range_reader_discards_partial_boundary_lines(monkeypatch):
    complete = {"agent_ref": {"name": "calendar_simple_agent"}}
    payload = b"partial\n" + json.dumps(complete).encode() + b"\npartial"
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: _Response(content=payload))

    assert _range_rows("https://example.invalid/data", 100) == [complete]


def test_range_reader_rejects_server_that_ignores_range(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: _Response(status_code=200, content=b"{}"))

    with pytest.raises(RuntimeError, match="ignored bounded JSONL range"):
        _range_rows("https://example.invalid/data", 100)


def test_sampler_selects_one_row_per_generator(monkeypatch):
    spec = BlendSampleSpec(
        name="fixture",
        filename="fixture.jsonl",
        size=2_097_152,
        agents=frozenset({"a", "b"}),
        source=nemotron_ultra_rlvr1_source(),
    )
    rows = [
        {"agent_ref": {"name": name}, "responses_create_params": {"input": "question"}, "uuid": name}
        for name in ("a", "b")
    ]
    monkeypatch.setattr("infra.rl_data.nemotron_ultra_sample._range_rows", lambda url, offset: rows)

    assert {row["uuid"] for row in sample_raw_generator_rows(spec, seed=7, max_ranges=1)} == {"a", "b"}


def test_sampler_chooses_only_swe_rows_with_exact_tasktrove_proxies(monkeypatch):
    spec = BlendSampleSpec(
        name="fixture",
        filename="fixture.jsonl",
        size=2_097_152,
        agents=frozenset({NEMOTRON_ULTRA_SWE_AGENT}),
        source=nemotron_ultra_rlvr1_source(),
    )

    def row(step):
        return {
            "agent_ref": {"name": NEMOTRON_ULTRA_SWE_AGENT},
            "trajectory_id": 7,
            "info": {"step": step, "turn": 1, "depth": step},
            "metadata": {"instance_id": "owner__repo-a", "agent_cls": "opencode"},
            "responses_create_params": {"input": "question"},
            "uuid": f"step-{step}",
        }

    monkeypatch.setattr("infra.rl_data.nemotron_ultra_sample._range_rows", lambda url, offset: [row(2), row(3)])
    proxies = {
        ("7", 3, 1, 3, "owner__repo-a", "opencode"): {
            "path": "proxy-state-7-3.tar.gz",
            "task_binary": b"unused",
        }
    }

    selected = sample_raw_generator_rows(spec, seed=7, max_ranges=1, swe_proxies=proxies)

    assert [value["uuid"] for value in selected] == ["step-3"]
    assert selected[0]["metadata"]["tasktrove_proxy_path"] == "proxy-state-7-3.tar.gz"
