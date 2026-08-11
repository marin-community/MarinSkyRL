"""S5: opencode per-trial literal correlation in TerminalBenchGenerator.

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
import os
import sys
import types

import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# The generator pulls in the harbor agentic-RL stack (absent under the CPU extra).
try:
    import terminal_bench.terminal_bench_generator as terminal_bench_generator  # noqa: E402
    from terminal_bench.terminal_bench_generator import TerminalBenchGenerator  # noqa: E402
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

# The correlation builder ships in harbor; skip if this harbor predates the bridge.
try:
    import harbor.literal.rollout_build  # noqa: F401
except ImportError:
    pytest.skip("harbor without the literal rollout_build bridge", allow_module_level=True)


# The incremental reader itself is unit-tested standalone in test_literal_log_store.py
# (no harbor dependency). Here we only exercise the harbor-facing correlation +
# chat_history helpers, injecting a real store exactly as TerminalBenchGenerator.__init__
# does — the reader is a collaborator, not a bound method on the fake self.
from terminal_bench.literal_log_store import LiteralLogStore  # noqa: E402

_correlate = TerminalBenchGenerator._maybe_correlate_opencode_rollout_details
_select_chain = terminal_bench_generator._select_opencode_literal_chain


def _attach_store(s):
    s._literal_log_store = LiteralLogStore()
    return s


def _fake_self(collect=True, literal_log_path=None):
    s = types.SimpleNamespace(_collect_rollout_details=collect, _literal_log_path=literal_log_path)
    return _attach_store(s)


def _result(trial_id, rollout_details=None):
    md = {"rollout_correlation_id": trial_id} if trial_id else None
    agent_result = types.SimpleNamespace(rollout_details=rollout_details, metadata=md)
    return types.SimpleNamespace(agent_result=agent_result)


def _entry(trial_id, ts, pids, cids, lps):
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {"messages": [{"role": "user", "content": "same task"}]},
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
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, entries))

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

    selected = _select_chain(entries, "A")

    assert [entry["literal"]["completion_token_ids"] for entry in selected] == [[10], [11], [12]]


def test_selects_only_final_session_after_context_reset():
    """A compacted session starts a new causal sequence and excludes its prefix."""
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 2.0, [1, 10, 2], [11], [-0.2]),
        _entry("A", 3.0, [50], [60], [-0.3]),
        _entry("A", 4.0, [50, 60, 3], [61], [-0.4]),
    ]

    selected = _select_chain(entries, "A")

    assert [entry["literal"]["completion_token_ids"] for entry in selected] == [[60], [61]]


def test_correlation_excludes_auxiliary_call_from_tito_stream(tmp_path, monkeypatch):
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 1.5, [99], [90], [-0.2]),
        _entry("A", 2.0, [1, 10, 2], [11], [-0.3]),
    ]
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, entries))

    rollout_details = _correlate(_fake_self(), _result("A"), None)

    assert rollout_details[0]["completion_token_ids"] == [[10], [11]]
    assert rollout_details[0]["prompt_token_ids"] == [[1], [1, 10, 2]]


def test_noop_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    assert _correlate(_fake_self(collect=False), _result("A"), None) is None


def test_noop_when_already_populated(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    existing = [{"completion_token_ids": [[99]], "logprobs": [[-0.9]]}]
    assert _correlate(_fake_self(), _result("A"), existing) is existing


def test_noop_when_no_correlation_id(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    assert _correlate(_fake_self(), _result(None), None) is None


def test_noop_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYRL_IRIS_LITERAL_LOG_PATH", raising=False)
    assert _correlate(_fake_self(), _result("A"), None) is None


def test_noop_when_log_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", str(tmp_path / "nope.jsonl"))
    assert _correlate(_fake_self(), _result("A"), None) is None


def test_noop_when_trial_absent_from_log(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    assert _correlate(_fake_self(), _result("Z"), None) is None


# --- opencode chat_history reconstruction (feeds _process_trial_result) ---------

_build_chat = TerminalBenchGenerator._maybe_build_opencode_chat_history


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
    the helper must resolve it even when SKYRL_IRIS_LITERAL_LOG_PATH is absent from the
    worker env (which is exactly what broke fullgate14 → tis/skipped_fraction=1.0)."""
    monkeypatch.delenv("SKYRL_IRIS_LITERAL_LOG_PATH", raising=False)
    log = _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "task"}], [5, 6])])
    ch = _build_chat(_chat_self(literal_log_path=log), _result("A"))
    assert ch == [{"role": "user", "content": "task"}, {"role": "assistant", "content": "assistant-final:5,6"}]


def test_correlate_resolves_log_from_cfg_path_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYRL_IRIS_LITERAL_LOG_PATH", raising=False)
    log = _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])])
    out = _correlate(_fake_self(literal_log_path=log), _result("A"), None)
    assert out[0]["completion_token_ids"] == [[10]]


def _entry_msgs(trial_id, ts, messages, cids):
    """A shared-log entry with an explicit request.messages + a completion turn."""
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {"messages": messages},
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
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, entries))
    ch = _build_chat(_chat_self(), _result("A"))
    # base = last request's messages (the full convo) + appended final assistant.
    assert ch[:-1] == convo_t2
    assert ch[-1] == {"role": "assistant", "content": "assistant-final:20,21"}


def test_chat_history_single_turn(tmp_path, monkeypatch):
    msgs = [{"role": "user", "content": "one-shot task"}]
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, [_entry_msgs("A", 1.0, msgs, [7, 8, 9])]))
    ch = _build_chat(_chat_self(), _result("A"))
    assert ch == msgs + [{"role": "assistant", "content": "assistant-final:7,8,9"}]


def test_chat_history_none_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]),
    )
    assert _build_chat(_chat_self(collect=False), _result("A")) is None


def test_chat_history_none_when_no_correlation_id(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]),
    )
    assert _build_chat(_chat_self(), _result(None)) is None


def test_chat_history_none_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYRL_IRIS_LITERAL_LOG_PATH", raising=False)
    assert _build_chat(_chat_self(), _result("A")) is None


def test_chat_history_none_when_trial_absent(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYRL_IRIS_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]),
    )
    assert _build_chat(_chat_self(), _result("Z")) is None


def test_chat_history_none_when_no_completion_bearing_record(tmp_path, monkeypatch):
    # status 200 but empty completion -> not a usable turn -> None (honest drop).
    e = _entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [])
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, [e]))
    assert _build_chat(_chat_self(), _result("A")) is None


def test_chat_history_none_when_request_has_no_messages(tmp_path, monkeypatch):
    e = _entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])
    e["request"] = {}  # malformed: no messages
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, [e]))
    assert _build_chat(_chat_self(), _result("A")) is None


def test_chat_history_picks_latest_timestamp_regardless_of_order(tmp_path, monkeypatch):
    # Entries out of order in the file; the LATEST timestamp must be the base.
    early = _entry_msgs("A", 1.0, [{"role": "user", "content": "early"}], [1])
    late = _entry_msgs("A", 9.0, [{"role": "user", "content": "late"}], [2])
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, [late, early]))
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
    monkeypatch.setenv("SKYRL_IRIS_LITERAL_LOG_PATH", _write_log(tmp_path, [_entry_msgs("A", 1.0, raw, [5, 6])]))
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
