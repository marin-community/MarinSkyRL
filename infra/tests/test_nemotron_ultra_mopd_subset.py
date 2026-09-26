import json

import pytest

from infra.rl_data.nemotron_ultra_mopd_subset import (
    MAX_REQUEST_CHARACTERS,
    ROUTE_COLUMN,
    TEACHER_ROUTES,
    prepare_subset_rows,
    routed_agent,
    sample_raw_subset_rows,
)
from infra.rl_data.sources import NEMOTRON_ULTRA_MOPD_AGENTS


def _row(agent: str, uuid: str = "u", **extra) -> dict:
    return {"agent_ref": {"name": agent}, "responses_create_params": {"input": "question"}, "uuid": uuid, **extra}


def test_routes_only_cover_in_process_generators_of_the_blend():
    assert set(TEACHER_ROUTES) <= NEMOTRON_ULTRA_MOPD_AGENTS
    for agent in ("math_with_judge_simple_agent", "genrm_simple_agent", "ns_tools_simple_agent"):
        assert agent not in TEACHER_ROUTES
    assert "swe_pivot_single_step_tool_use_with_argument_comparison_agent" not in TEACHER_ROUTES


def test_routed_agent_excludes_placeholders_oversized_and_unrouted_rows():
    assert routed_agent(_row("calendar_simple_agent")) == "calendar_simple_agent"
    assert routed_agent(_row("calendar_simple_agent", _hf_question_placeholder={"row": 1})) is None
    assert routed_agent(_row("abstention_simple_agent")) is None
    oversized = _row("calendar_simple_agent")
    oversized["responses_create_params"]["input"] = "x" * MAX_REQUEST_CHARACTERS
    assert routed_agent(oversized) is None
    assert routed_agent({"agent_ref": "not-a-mapping"}) is None


def test_sampler_stops_once_every_generator_has_its_quota(monkeypatch):
    calls = []

    def fake_range_rows(url, offset):
        calls.append(offset)
        return [_row(agent, uuid=f"{agent}-{offset}") for agent in TEACHER_ROUTES]

    monkeypatch.setattr("infra.rl_data.nemotron_ultra_mopd_subset.range_rows", fake_range_rows)

    candidates = sample_raw_subset_rows(seed=7, rows_per_agent=2, max_ranges=200)

    assert set(candidates) == set(TEACHER_ROUTES)
    assert all(len(rows) == 2 for rows in candidates.values())
    # One 24-range batch already satisfies every quota, so the sampler reads no second batch.
    assert len(calls) == 24
    assert candidates == sample_raw_subset_rows(seed=7, rows_per_agent=2, max_ranges=200)


def test_sampler_rejects_non_positive_bounds():
    with pytest.raises(ValueError):
        sample_raw_subset_rows(seed=1, rows_per_agent=0)


def test_prepared_rows_carry_the_hardcoded_route_column():
    candidates = {
        "calendar_simple_agent": [_row("calendar_simple_agent", uuid="cal", exp_cal_state={})],
        "code_gen_simple_agent": [_row("code_gen_simple_agent", uuid="code")],
    }

    prepared = prepare_subset_rows(candidates)

    assert [row[ROUTE_COLUMN] for row in prepared] == ["terminal", "swe"]
    assert [row["env_class"] for row in prepared] == ["nemotron_ultra", "nemotron_ultra"]
    assert [row["extra_info"]["index"] for row in prepared] == [0, 1]
    ultra = prepared[0]["extra_info"]["nemotron_ultra"]
    assert (ultra["blend"], ultra["agent"], ultra["route"]) == ("mopd", "calendar_simple_agent", "skyrl_gym")
    assert json.loads(ultra["record_json"])["exp_cal_state"] == {}
