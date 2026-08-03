"""Correlate a worker-side RecordProxy literal.jsonl with opencode trajectories.

For installed agents like ``opencode`` the LLM calls happen INSIDE a Daytona
sandbox (via ai-sdk), so harbor never sees the token IDs / logprobs. A
co-located :class:`~harbor.literal.proxy.RecordProxy` on the iris WORKER
captures them into a single job-global ``literal.jsonl`` (see
``hpc/literal_proxy_utils.py``). ai-sdk hides the upstream vLLM completion id,
so the trajectory has no ``response_id`` to join on. This module correlates by
CONTENT + COUNTS, entirely from data already logged:

  1. Reconstruct per-trial ordered call CHAINS from the global log by exact
     message-PREFIX extension: within a trial, opencode replays the full history
     each turn, so request_k.messages is an exact prefix of request_{k+1}.messages.
     Distinct trials seed distinct chains. A record that could extend two chains
     is marked ambiguous.

  2. BIND a chain to a trajectory by its exact per-step
     ``(prompt_tokens, completion_tokens)`` sequence. The ``agent_execution``
     time window only breaks ties (never forces a match).

  3. INJECT the token IDs / logprobs into the trajectory step metrics ONLY when
     the per-step counts agree exactly (verify-or-skip). A wrong guess yields
     OMISSION, never corrupt training tokens.

All functions here are pure and I/O-free except the thin ``load_*`` / ``enrich_*``
boundary helpers, so the correlation logic is unit-testable without a live job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from upath import UPath

# A record's messages is considered part of the SAME trial as a longer record
# when it is an exact element-wise prefix of the longer messages list.
Messages = list[dict[str, Any]]


@dataclass
class LiteralRecord:
    """One RecordProxy exchange, reduced to the fields the correlator needs."""

    timestamp: float
    messages: Messages
    prompt_token_ids: Optional[list[int]]
    completion_token_ids: Optional[list[int]]
    logprobs: Optional[list[float]]
    ambiguous: bool = False

    @property
    def counts(self) -> tuple[int, int]:
        """(len(prompt_token_ids), len(completion_token_ids)); -1 if absent."""
        p = len(self.prompt_token_ids) if self.prompt_token_ids is not None else -1
        c = len(self.completion_token_ids) if self.completion_token_ids is not None else -1
        return (p, c)


@dataclass
class Chain:
    """An ordered sequence of a single trial's LLM calls (oldest first)."""

    records: list[LiteralRecord] = field(default_factory=list)

    @property
    def count_sequence(self) -> list[tuple[int, int]]:
        return [r.counts for r in self.records]

    @property
    def ambiguous(self) -> bool:
        return any(r.ambiguous for r in self.records)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_literal_records(lines: Iterable[str]) -> list[LiteralRecord]:
    """Parse RecordProxy JSONL lines into usable :class:`LiteralRecord`s.

    ``lines`` may be any iterable of strings — a list, or (preferred for the
    multi-GB per-serve logs) a streaming file handle, so the whole file is never
    materialized at once.

    Keeps only status-200 records that carry a ``literal`` block with
    ``completion_token_ids`` and a ``request.messages`` list — the minimum needed
    to both correlate (messages) and inject (token ids).
    """
    out: list[LiteralRecord] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("status_code") != 200:
            continue
        literal = entry.get("literal")
        request = entry.get("request")
        if not isinstance(literal, dict) or not isinstance(request, dict):
            continue
        messages = request.get("messages")
        completion = literal.get("completion_token_ids")
        if not isinstance(messages, list) or not isinstance(completion, list) or not completion:
            continue
        prompt = literal.get("prompt_token_ids")
        logprobs = literal.get("logprobs")
        out.append(
            LiteralRecord(
                timestamp=float(entry.get("timestamp") or 0.0),
                messages=messages,
                prompt_token_ids=prompt if isinstance(prompt, list) else None,
                completion_token_ids=completion,
                logprobs=logprobs if isinstance(logprobs, list) else None,
            )
        )
    return out


def _is_strict_prefix(shorter: Messages, longer: Messages) -> bool:
    """True when ``shorter`` is an exact element-wise prefix of ``longer`` (strictly)."""
    if len(shorter) >= len(longer):
        return False
    return all(shorter[i] == longer[i] for i in range(len(shorter)))


def reconstruct_chains(records: list[LiteralRecord]) -> list[Chain]:
    """Group records into per-trial ordered chains by exact message-prefix extension.

    Records are processed shortest-history-first. Each record extends the unique
    chain whose current tip is a strict message-prefix of it; a record that could
    extend two or more chains (identical task instruction across trials) is marked
    ambiguous and seeds its own chain, so the count-sequence bind will skip it
    rather than risk a wrong join.
    """
    ordered = sorted(records, key=lambda r: (len(r.messages), r.timestamp))
    chains: list[Chain] = []
    for rec in ordered:
        candidates = [ch for ch in chains if _is_strict_prefix(ch.records[-1].messages, rec.messages)]
        if len(candidates) == 1:
            candidates[0].records.append(rec)
        elif not candidates:
            chains.append(Chain(records=[rec]))
        else:
            # Ambiguous: >1 chain could be extended (duplicate seed) -> do not
            # guess; isolate this record so any chain it touches is skippable.
            rec.ambiguous = True
            chains.append(Chain(records=[rec]))
    return chains


# --------------------------------------------------------------------------- #
# Trajectory <-> chain binding
# --------------------------------------------------------------------------- #
def trajectory_count_sequence(trajectory: dict[str, Any]) -> list[tuple[int, int]]:
    """Per-agent-step ``(prompt_tokens, completion_tokens)`` from a trajectory.

    Mirrors the agent-step selection the exporter uses: source=="agent" and not
    ``is_copied_context``. A step missing either count contributes ``-1`` so it can
    never spuriously equal a real record length.
    """
    seq: list[tuple[int, int]] = []
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        metrics = step.get("metrics") or {}
        p = metrics.get("prompt_tokens")
        c = metrics.get("completion_tokens")
        seq.append((p if isinstance(p, int) else -1, c if isinstance(c, int) else -1))
    return seq


def parse_iso(ts: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp to a unix float, or None."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _chain_in_window(chain: Chain, window: Optional[tuple[float, float]], slack: float = 120.0) -> bool:
    """True if every record timestamp falls within ``window`` (± ``slack`` seconds)."""
    if window is None:
        return True
    start, end = window
    return all(start - slack <= r.timestamp <= end + slack for r in chain.records)


def bind_chain(
    chains: list[Chain],
    count_sequence: list[tuple[int, int]],
    window: Optional[tuple[float, float]] = None,
) -> Optional[Chain]:
    """Return the single chain matching ``count_sequence`` exactly, else None.

    Verify-or-skip: a match requires the chain's ``(len(pids), len(cids))``
    sequence to equal the trajectory's ``(prompt_tokens, completion_tokens)``
    sequence exactly (and the chain to be unambiguous). If multiple chains match,
    the ``agent_execution`` window is used ONLY to break the tie; if still not
    unique, return None (omit rather than mis-join).
    """
    if not count_sequence or any(-1 in pc for pc in count_sequence):
        return None
    matches = [ch for ch in chains if not ch.ambiguous and ch.count_sequence == count_sequence]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and window is not None:
        windowed = [ch for ch in matches if _chain_in_window(ch, window)]
        if len(windowed) == 1:
            return windowed[0]
    return None


def inject_literals(trajectory: dict[str, Any], chain: Chain) -> int:
    """Inject a bound chain's token IDs / logprobs into the trajectory's agent steps.

    Verify-or-skip per step: a step is enriched only when its
    ``(prompt_tokens, completion_tokens)`` exactly equal the record's
    ``(len(pids), len(cids))``. Returns the number of steps enriched. Mutates
    ``trajectory`` in place.
    """
    agent_steps = [
        s for s in trajectory.get("steps", []) if s.get("source") == "agent" and not s.get("is_copied_context")
    ]
    enriched = 0
    for step, rec in zip(agent_steps, chain.records):
        metrics = step.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            step["metrics"] = metrics
        p_ok = rec.prompt_token_ids is not None and metrics.get("prompt_tokens") == len(rec.prompt_token_ids)
        c_ok = metrics.get("completion_tokens") == len(rec.completion_token_ids)
        if not c_ok:
            continue
        metrics["completion_token_ids"] = rec.completion_token_ids
        if p_ok:
            metrics["prompt_token_ids"] = rec.prompt_token_ids
        if rec.logprobs is not None:
            metrics["logprobs"] = rec.logprobs
        enriched += 1
    return enriched


# --------------------------------------------------------------------------- #
# I/O boundary
# --------------------------------------------------------------------------- #
def load_literal_records(literal_log_uri: "str | Sequence[str]") -> list[LiteralRecord]:
    """Read + parse one OR MANY (possibly gs://) literal.jsonl files into records.

    A preempted campaign job accumulates ONE durable ``literal.jsonl`` per serve
    attempt; each trial's records live entirely in whichever serve handled it,
    so binding is correct over the UNION of all attempts' records.
    """
    uris = [literal_log_uri] if isinstance(literal_log_uri, str) else list(literal_log_uri)
    records: list[LiteralRecord] = []
    for uri in uris:
        # Stream line-by-line — a per-serve literal.jsonl is multi-GB (token ids +
        # full request messages), so iterating the open handle keeps the transient
        # at one line while parse_literal_records accumulates only the kept records.
        with UPath(uri).open("r") as fh:
            records.extend(parse_literal_records(fh))
    return records


def _trial_window(trial_dir: UPath) -> Optional[tuple[float, float]]:
    """The trial's agent_execution [start, end] as unix floats, or None."""
    result = trial_dir / "result.json"
    try:
        data = json.loads(result.read_text())
    except (FileNotFoundError, json.JSONDecodeError, NotImplementedError):
        return None
    exec_block = data.get("agent_execution")
    if not isinstance(exec_block, dict):
        return None
    start = parse_iso(exec_block.get("started_at"))
    end = parse_iso(exec_block.get("finished_at"))
    if start is None or end is None:
        return None
    return (start, end)


_LITERAL_METRIC_KEYS = ("prompt_token_ids", "completion_token_ids", "logprobs")


def _strip_literal_metrics(trajectory: dict[str, Any]) -> int:
    """Remove any pre-existing literal token_ids/logprobs from every step's metrics.

    The correlator is the SOLE authority on literal columns when active: token IDs
    must come from the RecordProxy ``literal.jsonl`` via :func:`inject_literals`,
    never from whatever was sitting in ``trajectory.json``. Stripping first
    guarantees an unbound trial contributes NO literal columns. Returns the number
    of steps modified.
    """
    modified = 0
    for step in trajectory.get("steps", []):
        metrics = step.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if any(k in metrics for k in _LITERAL_METRIC_KEYS):
            for k in _LITERAL_METRIC_KEYS:
                metrics.pop(k, None)
            modified += 1
    return modified


@dataclass
class EnrichStats:
    trials: int = 0
    trials_enriched: int = 0
    steps_enriched: int = 0
    trials_stripped: int = 0
    chains: int = 0
    ambiguous_chains: int = 0


def enrich_trajectories_with_literals(
    job_dir: str,
    literal_log_uri: "str | Sequence[str]",
    *,
    iter_trial_dirs,
    verbose: bool = False,
) -> EnrichStats:
    """Populate opencode trajectory step metrics with proxy-captured literals.

    Reconstructs per-trial chains from ``literal_log_uri`` (one file, OR the UNION
    of every serve attempt's file for a preempted job) once, then for each trial
    under ``job_dir`` binds its trajectory to a chain (verify-or-skip) and rewrites
    ``agent/trajectory.json`` in place with the token IDs / logprobs. Idempotent.
    """
    records = load_literal_records(literal_log_uri)
    chains = reconstruct_chains(records)
    stats = EnrichStats(chains=len(chains), ambiguous_chains=sum(ch.ambiguous for ch in chains))
    if verbose:
        n_files = 1 if isinstance(literal_log_uri, str) else len(list(literal_log_uri))
        print(
            f"[literal-correlator] {len(records)} records -> {len(chains)} chains "
            f"({stats.ambiguous_chains} ambiguous) from {n_files} literal file(s)"
        )

    for trial_dir in iter_trial_dirs(job_dir):
        trial_dir = UPath(trial_dir)
        traj_path = trial_dir / "agent" / "trajectory.json"
        try:
            trajectory = json.loads(traj_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, NotImplementedError):
            continue
        stats.trials += 1
        # Strip any pre-existing literal fields FIRST so the correlator is the
        # sole source of token columns. Counts used for binding are untouched.
        stripped = _strip_literal_metrics(trajectory)
        seq = trajectory_count_sequence(trajectory)
        chain = bind_chain(chains, seq, window=_trial_window(trial_dir))
        n = inject_literals(trajectory, chain) if chain is not None else 0
        if n:
            traj_path.write_text(json.dumps(trajectory))
            stats.trials_enriched += 1
            stats.steps_enriched += n
            if verbose:
                print(f"[literal-correlator] {trial_dir.name}: enriched {n} step(s)")
        elif stripped:
            # Unbound trial that HAD stale literal fields: persist the stripped
            # trajectory so those bogus token_ids can't leak.
            traj_path.write_text(json.dumps(trajectory))
            stats.trials_stripped += 1
            if verbose:
                print(f"[literal-correlator] {trial_dir.name}: stripped stale literal fields")

    if verbose:
        print(
            f"[literal-correlator] enriched {stats.trials_enriched}/{stats.trials} trials "
            f"({stats.steps_enriched} steps)"
        )
    return stats


def discover_literal_logs(job_dir: str) -> list[str]:
    """Find ALL durable literal.jsonl files for a job dir (empty list if none).

    The proxy writes one per serve attempt to
    ``<experiments_dir>/logs/<slug>_literal.jsonl``. The export ``--job_dir`` may
    be that experiments_dir or a child; search the dir and a few parents, and
    return EVERY match from the first ``logs/`` that has any — so a preempted
    job's records are correlated over the full union. Returns them sorted.
    """
    cur = UPath(job_dir)
    for _ in range(4):
        logs = cur / "logs"
        try:
            if logs.exists():
                hits = sorted(logs.glob("*_literal.jsonl"))
                if hits:
                    return [str(h) for h in hits]
        except (FileNotFoundError, NotImplementedError):
            pass
        if cur.parent == cur:
            break
        cur = cur.parent
    return []


def discover_literal_log(job_dir: str) -> Optional[str]:
    """Back-compat single-file discovery: the first of :func:`discover_literal_logs`."""
    logs = discover_literal_logs(job_dir)
    return logs[0] if logs else None
