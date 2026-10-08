"""opencode per-trial literal correlation in HarborTrajectoryRunner.

opencode is a CLI agent that bypasses harbor Chat, so it returns EMPTY
rollout_details even behind a co-located RecordProxy (which writes ONE shared
worker-side log for all concurrent trials). The generator recovers each trial's
token_ids/logprobs from that shared log by the per-trial correlation id harbor
stamped (x-ot-trial-id), so TIS works AND concurrent GRPO same-seed trials never
bleed into each other.

Exercises the two helpers as unbound methods on a SimpleNamespace fake self (the
CPU-test pattern from test_preserve_logprobs_on_timeout.py) so no tokenizer /
engine / config is needed.
"""

import json
import types

import pytest

# The runner pulls in the Harbor agentic-RL stack (absent under the CPU extra).
try:
    import skyrl_train.trajectory_runners.harbor.runner as harbor_runner_module
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

# The correlation builder ships in harbor; skip if this harbor predates the bridge.
try:
    import harbor.literal.rollout_build  # noqa: F401
except ImportError:
    pytest.skip("harbor without the literal rollout_build bridge", allow_module_level=True)


# The incremental reader itself is unit-tested standalone in test_literal_log_store.py
# (no harbor dependency). Here we only exercise the harbor-facing correlation +
# chat_history helpers, injecting a real store exactly as HarborTrajectoryRunner.__init__
# does — the reader is a collaborator, not a bound method on the fake self.
from skyrl_train.trajectory_runners.harbor.literal_log_store import LiteralLogStore  # noqa: E402

_correlate = HarborTrajectoryRunner._maybe_correlate_cli_rollout_details
_select_chain = harbor_runner_module._select_cli_literal_chain


def _attach_store(s):
    s._literal_log_store = LiteralLogStore()
    return s


def _fake_self(collect=True, literal_log_path=None):
    s = types.SimpleNamespace(_collect_rollout_details=collect, _literal_log_path=literal_log_path)
    return _attach_store(s)


def _result(trial_id, rollout_details=None):
    md = {"rollout_correlation_id": trial_id} if trial_id else None
    agent_result = types.SimpleNamespace(rollout_details=rollout_details, metadata=md)
    return types.SimpleNamespace(agent_result=agent_result, agent_info=types.SimpleNamespace(name="opencode"))


def _entry(trial_id, ts, pids, cids, lps):
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {
            "messages": [{"role": "user", "content": "same task"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        },
        "literal": {
            "prompt_token_ids": pids,
            "completion_token_ids": cids,
            "logprobs": lps,
        },
    }


def _write_log(tmp_path, entries):
    p = tmp_path / "literal.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(p)


def test_correlates_per_trial_no_bleed_for_identical_seed_group(tmp_path, monkeypatch):
    """n=2 rollouts of the IDENTICAL prompt, interleaved in one shared log: each
    trial gets ITS OWN turns; no cross-rollout bleed; result is mutated in place."""
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("B", 1.1, [1], [20], [-0.2]),
        _entry("A", 2.0, [1, 10], [11], [-0.3]),
        _entry("B", 2.1, [1, 20], [21], [-0.4]),
    ]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))

    ra = _result("A")
    a = _correlate(_fake_self(), ra, None)
    assert a[0]["completion_token_ids"] == [[10], [11]]
    assert a[0]["logprobs"] == [[-0.1], [-0.3]]
    assert ra.agent_result.rollout_details == a  # persisted onto the result

    rb = _result("B")
    b = _correlate(_fake_self(), rb, None)
    assert b[0]["completion_token_ids"] == [[20], [21]]
    # No bleed.
    assert 20 not in [t for turn in a[0]["completion_token_ids"] for t in turn]
    assert 10 not in [t for turn in b[0]["completion_token_ids"] for t in turn]


def test_selects_final_continuous_chain_around_auxiliary_call():
    """An auxiliary OpenCode request must not displace agent turns from TITO."""
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 1.5, [99], [90], [-0.2]),
        _entry("A", 2.0, [1, 10, 2], [11], [-0.3]),
        _entry("A", 3.0, [1, 10, 2, 11, 3], [12], [-0.4]),
    ]

    selected = _select_chain(entries, "A", "opencode")

    assert [entry["literal"]["completion_token_ids"] for entry in selected] == [[10], [11], [12]]


def test_selects_only_final_session_after_context_reset():
    """A compacted session starts a new causal sequence and excludes its prefix."""
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 2.0, [1, 10, 2], [11], [-0.2]),
        _entry("A", 3.0, [50], [60], [-0.3]),
        _entry("A", 4.0, [50, 60, 3], [61], [-0.4]),
    ]

    selected = _select_chain(entries, "A", "opencode")

    assert [entry["literal"]["completion_token_ids"] for entry in selected] == [[60], [61]]


def test_correlation_excludes_auxiliary_call_from_tito_stream(tmp_path, monkeypatch):
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 1.5, [99], [90], [-0.2]),
        _entry("A", 2.0, [1, 10, 2], [11], [-0.3]),
    ]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))

    rollout_details = _correlate(_fake_self(), _result("A"), None)

    assert rollout_details[0]["completion_token_ids"] == [[10], [11]]
    assert rollout_details[0]["prompt_token_ids"] == [[1], [1, 10, 2]]


@pytest.mark.parametrize(
    ("entries", "expected_completions", "expected_logprobs"),
    [
        pytest.param(
            [
                _entry("A", 1.0, [1], [10, 11], [-0.1, -0.1]),
                _entry("A", 2.0, [1, 10, 11], [12], [-0.2]),
                _entry("A", 3.0, [1, 10], [11, 12], [-0.3, -0.3]),
                _entry("A", 4.0, [1, 10, 11, 12], [13], [-0.4]),
            ],
            [[10, 11], [12], [13]],
            [[-0.1, -0.1], [-0.2], [-0.4]],
            id="longer-chain-before-newer-predecessor",
        ),
        pytest.param(
            [
                _entry("A", 1.0, [1], [10], [-0.1]),
                _entry("A", 2.0, [1], [10], [-0.2]),
                _entry("A", 3.0, [1, 10], [11], [-0.3]),
            ],
            [[10], [11]],
            [[-0.1], [-0.3]],
            id="earliest-predecessor-on-equal-chain-length",
        ),
    ],
)
def test_correlation_preserves_longest_chain_and_tied_logprobs(
    tmp_path, monkeypatch, entries, expected_completions, expected_logprobs
):
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))

    details = _correlate(_fake_self(), _result("A"), None)

    assert details[0]["completion_token_ids"] == expected_completions
    assert details[0]["logprobs"] == expected_logprobs


@pytest.mark.parametrize("agent_name", ["opencode", "mini-swe-agent"])
def test_correlation_keeps_task_route_when_tool_free_call_finishes_last(tmp_path, monkeypatch, agent_name):
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 2.0, [1, 10, 2], [11], [-0.2]),
        _entry("A", 3.0, [99], [90], [-0.3]),
    ]
    for entry in entries:
        if agent_name == "mini-swe-agent" or entry is entries[-1]:
            entry["request"]["tools"] = []
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))
    result = _result("A")
    result.agent_info.name = agent_name

    details = _correlate(_fake_self(), result, None)

    assert details[0]["completion_token_ids"] == ([[10], [11]] if agent_name == "opencode" else [[90]])


MISSING_LOG = object()
EXISTING_DETAILS = [{"completion_token_ids": [[99]], "logprobs": [[-0.9]]}]
CORRELATABLE_LOG = [_entry("A", 1.0, [1], [10], [-0.1])]


def _set_log_env(monkeypatch, tmp_path, entries):
    """Publish the shared log path; ``None`` unsets it and ``MISSING_LOG`` points at no file."""
    if entries is None:
        monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    elif entries is MISSING_LOG:
        monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", str(tmp_path / "nope.jsonl"))
    else:
        monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))


@pytest.mark.parametrize(
    ("collect", "log_entries", "trial_id", "existing"),
    [
        pytest.param(False, CORRELATABLE_LOG, "A", None, id="flag-off"),
        pytest.param(True, CORRELATABLE_LOG, "A", EXISTING_DETAILS, id="already-populated"),
        pytest.param(True, CORRELATABLE_LOG, None, None, id="no-correlation-id"),
        pytest.param(True, None, "A", None, id="env-unset"),
        pytest.param(True, MISSING_LOG, "A", None, id="log-missing"),
        pytest.param(True, CORRELATABLE_LOG, "Z", None, id="trial-absent"),
    ],
)
def test_correlation_leaves_rollout_details_unchanged_without_usable_evidence(
    tmp_path, monkeypatch, collect, log_entries, trial_id, existing
):
    _set_log_env(monkeypatch, tmp_path, log_entries)
    assert _correlate(_fake_self(collect=collect), _result(trial_id), existing) is existing


# --- opencode chat_history reconstruction (feeds _process_trial_result) ---------

_build_chat = HarborTrajectoryRunner._maybe_build_cli_chat_history


class _FakeTokenizer:
    """Minimal tokenizer: decode ids -> a marker string (content is opaque to the
    reconstruction; the exact-id alignment path consumes the raw ids, not this text)."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: D401
        return "assistant-final:" + ",".join(str(i) for i in ids)


def _chat_self(collect=True, tokenizer=None, literal_log_path=None):
    s = types.SimpleNamespace(
        _collect_rollout_details=collect,
        tokenizer=tokenizer or _FakeTokenizer(),
        _literal_log_path=literal_log_path,
    )
    return _attach_store(s)


def test_chat_history_resolves_log_from_cfg_path_when_env_unset(tmp_path, monkeypatch):
    """The Ray-boundary fix: the log path is threaded via cfg (self._literal_log_path);
    the helper must resolve it even when OTAGENT_LITERAL_LOG_PATH is absent from the
    worker env (which is exactly what broke fullgate14 → tis/skipped_fraction=1.0)."""
    monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    log = _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "task"}], [5, 6])])
    ch = _build_chat(_chat_self(literal_log_path=log), _result("A"))
    assert ch == [{"role": "user", "content": "task"}, {"role": "assistant", "content": "assistant-final:5,6"}]


def test_correlate_resolves_log_from_cfg_path_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    log = _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])])
    out = _correlate(_fake_self(literal_log_path=log), _result("A"), None)
    assert out[0]["completion_token_ids"] == [[10]]


def _entry_msgs(trial_id, ts, messages, cids):
    """A shared-log entry with an explicit request.messages + a completion turn."""
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {"messages": messages, "tools": [{"type": "function", "function": {"name": "bash"}}]},
        "literal": {"prompt_token_ids": [1], "completion_token_ids": cids, "logprobs": [-0.1] * len(cids)},
    }


def test_chat_history_uses_last_request_plus_decoded_final_completion(tmp_path, monkeypatch):
    # Two turns of a growing conversation; the LAST request carries the full prior
    # context, and we append the final turn's decoded completion as the last assistant.
    convo_t1 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "solve X"},
    ]
    convo_t2 = convo_t1 + [
        {"role": "assistant", "content": "turn-1 answer"},
        {"role": "user", "content": "tool result"},
    ]
    entries = [
        _entry_msgs("A", 1.0, convo_t1, [10, 11]),
        _entry_msgs("A", 2.0, convo_t2, [20, 21]),
    ]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))
    ch = _build_chat(_chat_self(), _result("A"))
    # base = last request's messages (the full convo) + appended final assistant.
    assert ch[:-1] == convo_t2
    assert ch[-1] == {"role": "assistant", "content": "assistant-final:20,21"}


def test_chat_history_single_turn(tmp_path, monkeypatch):
    msgs = [{"role": "user", "content": "one-shot task"}]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [_entry_msgs("A", 1.0, msgs, [7, 8, 9])]))
    ch = _build_chat(_chat_self(), _result("A"))
    assert ch == msgs + [{"role": "assistant", "content": "assistant-final:7,8,9"}]


def _entry_without_messages():
    entry = _entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])
    entry["request"] = {}
    return entry


CHAT_LOG = [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]


@pytest.mark.parametrize(
    ("collect", "log_entries", "trial_id"),
    [
        pytest.param(False, CHAT_LOG, "A", id="flag-off"),
        pytest.param(True, CHAT_LOG, None, id="no-correlation-id"),
        pytest.param(True, None, "A", id="env-unset"),
        pytest.param(True, CHAT_LOG, "Z", id="trial-absent"),
        # A 200 with an empty completion is not a usable turn: drop honestly.
        pytest.param(
            True,
            [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [])],
            "A",
            id="no-completion-bearing-record",
        ),
        pytest.param(True, [_entry_without_messages()], "A", id="request-without-messages"),
    ],
)
def test_chat_history_is_none_without_usable_evidence(tmp_path, monkeypatch, collect, log_entries, trial_id):
    _set_log_env(monkeypatch, tmp_path, log_entries)
    assert _build_chat(_chat_self(collect=collect), _result(trial_id)) is None


def test_chat_history_picks_latest_timestamp_regardless_of_order(tmp_path, monkeypatch):
    # Entries out of order in the file; the LATEST timestamp must be the base.
    early = _entry_msgs("A", 1.0, [{"role": "user", "content": "early"}], [1])
    late = _entry_msgs("A", 9.0, [{"role": "user", "content": "late"}], [2])
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [late, early]))
    ch = _build_chat(_chat_self(), _result("A"))
    assert ch[0] == {"role": "user", "content": "late"}


def test_chat_history_normalizes_openai_shaped_messages(tmp_path, monkeypatch):
    # opencode/ai-sdk send OpenAI-shaped messages: list-of-parts content, structured
    # tool_calls, and tool-role results. That raw shape made the downstream re-tok fallback
    # in get_response_ids_and_loss_mask_from_messages run apply_chat_template on a non-plain
    # message -> "TypeError: Can only get item pairs from a mapping" -> the trajectory was
    # classified zero-reward (this starved ~87% of the keep1-v24 batch). The reconstruction
    # must normalize every message to a plain {role, content:str} (harbor.normalize_message).
    raw = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "solve X"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "content": [{"type": "text", "text": "exit 0"}], "tool_call_id": "c1"},
    ]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [_entry_msgs("A", 1.0, raw, [5, 6])]))
    ch = _build_chat(_chat_self(), _result("A"))
    assert ch is not None
    # every kept message is a plain {role, content:str} — no list content, no tool_calls
    for m in ch:
        assert set(m.keys()) == {"role", "content"}
        assert isinstance(m["content"], str)
    assert ch[0] == {"role": "system", "content": "sys"}
    assert ch[1] == {"role": "user", "content": "solve X"}
    assert "bash" in ch[2]["content"]  # tool_calls serialized into text, not dropped
    assert ch[3] == {"role": "tool", "content": "exit 0"}
    # + the appended final assistant completion (decoded from the fake tokenizer)
    assert ch[-1] == {"role": "assistant", "content": "assistant-final:5,6"}
