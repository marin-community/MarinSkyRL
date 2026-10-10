import json

import pytest

from infra.rl_data.nemotron_ultra_mopd_subset import (
    ROUTE_COLUMN,
    TEACHER_ROUTES,
    prepare_subset_rows,
    routed_agent,
    sample_proportional_rows,
    sample_per_agent_rows,
    subset_manifest,
    swe_proxy_paths,
)
from infra.rl_data.nemotron_ultra_swe import SWEProxyKey
from infra.rl_data.sources import NEMOTRON_ULTRA_MOPD_AGENTS, NEMOTRON_ULTRA_SWE_AGENT

LIMIT = 1_000
SAMPLER = "infra.rl_data.nemotron_ultra_mopd_subset.range_rows"


def _row(agent: str, uuid: str = "u", **extra) -> dict:
    return {"agent_ref": {"name": agent}, "responses_create_params": {"input": "question"}, "uuid": uuid, **extra}


def _swe_row(trajectory_id: str) -> dict:
    return _row(
        NEMOTRON_ULTRA_SWE_AGENT,
        uuid=trajectory_id,
        trajectory_id=trajectory_id,
        info={"step": 1, "turn": 2, "depth": 0},
        metadata={"instance_id": f"repo__{trajectory_id}", "agent_cls": "CodeActAgent"},
    )


def _proxy(trajectory_id: str) -> tuple[SWEProxyKey, dict]:
    key = SWEProxyKey(
        trajectory_id=trajectory_id,
        step=1,
        turn=2,
        depth=0,
        instance_id=f"repo__{trajectory_id}",
        agent_cls="CodeActAgent",
    )
    return key, {"path": f"proxies/{trajectory_id}"}


def test_every_blend_generator_has_a_route():
    assert set(TEACHER_ROUTES) == NEMOTRON_ULTRA_MOPD_AGENTS
    assert set(TEACHER_ROUTES.values()) == {"math", "swe", "terminal"}


def test_routed_agent_excludes_oversized_and_unrouted_rows():
    assert routed_agent(_row("abstention_simple_agent"), max_request_characters=LIMIT) == "abstention_simple_agent"
    oversized = _row("calendar_simple_agent")
    oversized["responses_create_params"]["input"] = "x" * LIMIT
    assert routed_agent(oversized, max_request_characters=LIMIT) is None
    assert routed_agent(_row("unknown_agent"), max_request_characters=LIMIT) is None
    assert routed_agent({"agent_ref": "not-a-mapping"}, max_request_characters=LIMIT) is None


def test_per_agent_sampler_skips_unbound_swe_rows_and_is_deterministic(monkeypatch):
    calls = []
    agents = [agent for agent in TEACHER_ROUTES if agent != NEMOTRON_ULTRA_SWE_AGENT]

    def fake_range_rows(url, offset):
        calls.append(offset)
        return [_row(agent, uuid=f"{agent}-{offset}") for agent in agents] + [_swe_row(f"t{offset}")]

    monkeypatch.setattr(SAMPLER, fake_range_rows)

    def sample():
        return sample_per_agent_rows(
            seed=7, rows_per_agent=2, swe_proxies={}, max_request_characters=LIMIT, max_ranges=48
        )

    rows = sample()

    # No SWE proxy exists, so the SWE quota never fills and the sampler reads every range.
    assert len(calls) == 48
    assert {row["agent_ref"]["name"] for row in rows} == set(agents)
    assert len(rows) == 2 * len(agents)
    assert rows == sample()


def test_proportional_sampler_keeps_read_order_and_caps_bound_swe_rows(monkeypatch):
    block = [
        _row("mcqa_simple_agent", uuid="a"),
        _swe_row("bound-1"),
        _swe_row("unbound"),
        _row("abstention_simple_agent", uuid="b"),
        _swe_row("bound-2"),
        _row("ns_tools_simple_agent", uuid="c"),
        _row("code_gen_simple_agent", uuid="d"),
    ]
    monkeypatch.setattr(SAMPLER, lambda url, offset: block)
    proxies = dict([_proxy("bound-1"), _proxy("bound-2")])

    rows = sample_proportional_rows(
        seed=3, rows=4, swe_rows=1, swe_proxies=proxies, max_request_characters=LIMIT, max_ranges=1
    )

    assert [row["uuid"] for row in rows] == ["a", "bound-1", "b", "c"]
    assert swe_proxy_paths(rows) == {"proxies/bound-1"}


def test_prepared_rows_carry_routes_and_harbor_task_ids():
    swe = _swe_row("t1")
    swe["metadata"]["tasktrove_proxy_path"] = "proxies/t1"
    rows = [_row("calendar_simple_agent", uuid="cal", exp_cal_state={}), swe]

    prepared = prepare_subset_rows(rows)

    assert [row[ROUTE_COLUMN] for row in prepared] == ["terminal", "swe"]
    assert [row["extra_info"]["index"] for row in prepared] == [0, 1]
    calendar, swe_ultra = (row["extra_info"]["nemotron_ultra"] for row in prepared)
    assert (calendar["blend"], calendar["route"]) == ("mopd", "task_session")
    assert json.loads(calendar["record_json"])["exp_cal_state"] == {}
    assert (swe_ultra["route"], swe_ultra["terminal_bench_instance_id"]) == ("terminal_bench", "proxies/t1")


def test_manifest_requires_every_route():
    prepared = prepare_subset_rows([_row("calendar_simple_agent", exp_cal_state={})])

    with pytest.raises(RuntimeError, match="math"):
        subset_manifest(prepared, seed=1, revision="r", sampling={"mode": "proportional"})
