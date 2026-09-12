import json

import pytest

from infra.rl_data.nemotron_ultra_sample import BlendSampleSpec, _range_rows, sample_raw_generator_rows
from infra.rl_data.sources import nemotron_ultra_rlvr1_source


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
