"""One-engine queue persistence measurement; no learner or admission changes."""

import asyncio
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir


async def measure_queue(client, prompts, sampling, receipt, *, sample_delays, request_timeout=180.0):
    """Retain native ledger identities through one original-grace pause and resume."""
    states, logical = receipt["states"], receipt["logical"]
    clock = time.monotonic
    submitted = clock()
    receipt["submitted_monotonic"] = submitted

    async def snapshot(label, **kwargs):
        before = clock()
        rows = await client.read_publication_request_state(drain_accounting=True, **kwargs)
        assert len(rows) == 1, "one-engine precursor"
        states.append({"label": label, "read_started": before, "read_finished": clock(), "state": rows[0]})
        return rows[0]

    await snapshot("initial", initial_policy_version=0)

    async def generate(index, prompt):
        started = clock()
        result = await client.generate({"prompt_token_ids": [prompt], "sampling_params": dict(sampling)})
        logical.append(
            {
                "index": index,
                "started": started,
                "finished": clock(),
                "tokens": len(result["response_ids"][0]),
                "logprobs": len(result["response_logprobs"][0]),
                "finite_logprobs": all(math.isfinite(value) for value in result["response_logprobs"][0]),
                "stop_reason": result["stop_reasons"][0],
            }
        )

    tasks = [asyncio.create_task(generate(index, prompt)) for index, prompt in enumerate(prompts)]
    receipt["requests_submitted_monotonic"] = clock()
    try:
        # Sampling starts when every logical request has crossed the native
        # ledger boundary, including queued requests with no first token yet.
        started_ids = set()
        async with asyncio.timeout(30):
            while len(started_ids) < len(prompts):
                await asyncio.sleep(0)  # Let newly submitted logical tasks enter the native RPC.
                state = await snapshot("admission")
                started_ids.update(state["request_accounting"]["started_ids"])
                for task in tasks:
                    if task.done():
                        task.result()
        assert len(started_ids) == len(prompts)
        admitted = clock()
        receipt["all_admitted_monotonic"] = admitted
        for delay in sample_delays:
            # This deliberate delay is the measured PPO-duration input, not a
            # readiness heuristic. Both samples share the same origin.
            await asyncio.sleep(max(0.0, admitted + delay - clock()))
            await snapshot(f"delay_{delay}")
        await snapshot("before_grace")
        receipt["pause_started"] = clock()
        await asyncio.wait_for(client.pause_generation(), timeout=40)
        receipt["pause_finished"] = clock()
        await snapshot("paused", terminal_timeout_seconds=30)
        async with asyncio.timeout(5):
            while client.publication_inflight_snapshot() != (0,):
                await snapshot("abort_delivery")
                await asyncio.sleep(0)
        await client.resume_generation(policy_version=0)
        receipt["resume_finished"] = clock()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=request_timeout)
        await snapshot("final", terminal_timeout_seconds=30)
    except BaseException as error:
        receipt["failure_type"] = type(error).__name__
        raise
    finally:
        try:
            if client.generation_paused_event.is_set():
                await client.resume_generation()
        except BaseException as error:
            receipt["cleanup_failure_type"] = type(error).__name__
            if "failure_type" not in receipt:
                raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            dispositions = await asyncio.gather(*tasks, return_exceptions=True)
            receipt["cleanup_dispositions"] = [
                type(value).__name__ if isinstance(value, BaseException) else "returned" for value in dispositions
            ]
            receipt["finished_monotonic"] = clock()


def audit_queue(receipt, *, request_count, tokens_per_request):
    """Audit attempts without interpreting queued zero-token requests as sampled."""
    started, active, terminal = set(), set(), {}
    pauses, hosts = [], set()
    for row in receipt["states"]:
        state = row["state"]
        assert state["shared_time_and_uts_namespaces"] and state["clock_domain"] == "CLOCK_MONOTONIC"
        hosts.add((state["host"], state["actor_pid"], tuple(state["core_pids"])))
        ledger = state["request_accounting"]
        new = ledger["started_ids"]
        assert len(new) == len(set(new)) and not started.intersection(new), "duplicate native start"
        started.update(new)
        active.update(new)
        for result in ledger["terminal"]:
            identity = result["request_id"]
            assert identity in active and identity not in terminal, "missing or duplicate terminal"
            active.remove(identity)
            terminal[identity] = result
            assert result["reason"] in {"length", "stop", "abort"}, "native request failure"
            tokens, stamp = result["tokens"], result["first_token_time"]
            assert type(tokens) is int and tokens >= 0
            raw = result["native_first_token_time"]
            assert stamp == (None if tokens == 0 and raw == 0.0 else raw), "canonical timestamp differs"
            if tokens:
                assert stamp is not None and math.isfinite(stamp) and 0 < stamp <= state["observed_monotonic"]
                boundaries = state["policy_version_boundaries"]
                version = next((version for boundary, version in reversed(boundaries) if stamp >= boundary), None)
                assert version == result["policy_version_at_first_token"] == 0, "first-token version"
            else:
                assert stamp is None or (math.isfinite(stamp) and 0 < stamp <= state["observed_monotonic"])
        assert active == set(ledger["active_ids"]), "active conservation"
        pauses.extend(ledger["pauses"])
    assert len(hosts) == 1 and started == set(terminal) and not active, "lost request or engine replacement"
    assert len(pauses) == 1 and pauses[0]["pause_index"] == 1
    assert set(pauses[0]["frontend_before"]) <= set(pauses[0]["active_before"]) <= started
    logical = receipt["logical"]
    assert len(logical) == request_count and {row["index"] for row in logical} == set(range(request_count))
    assert all(
        row["tokens"] == row["logprobs"] == tokens_per_request
        and row["finite_logprobs"]
        and row["stop_reason"] == "length"
        for row in logical
    )
    assert receipt["cleanup_dispositions"] == ["returned"] * request_count
    by_label = {row["label"]: row for row in receipt["states"]}
    assert by_label["paused"]["state"]["paused"] and by_label["paused"]["state"]["frontend_requests"] == 0
    assert not by_label["final"]["state"]["paused"] and by_label["final"]["state"]["frontend_requests"] == 0
    # Native continuation appends each abort prefix once and subtracts its
    # length from the retry budget; no model death/reset is accepted here.
    assert sum(row["tokens"] for row in terminal.values()) == request_count * tokens_per_request, (
        "native token conservation"
    )
    outcomes = Counter(row["reason"] for row in terminal.values())
    return {
        "native_starts": len(started),
        "native_terminals": len(terminal),
        "lost": 0,
        "terminal_reasons": dict(outcomes),
        "zero_token_aborts": sum(row["reason"] == "abort" and row["tokens"] == 0 for row in terminal.values()),
        "native_pause_active": len(pauses[0]["active_before"]),
        "native_pause_frontend": len(pauses[0]["frontend_before"]),
        "native_attempt_tokens": sum(row["tokens"] for row in terminal.values()),
        "abort_tokens": sum(row["tokens"] for row in terminal.values() if row["reason"] == "abort"),
        "retry_attempts": len(started) - request_count,
        "logical_tokens": sum(row["tokens"] for row in logical),
        "pause_wall": receipt["pause_finished"] - receipt["pause_started"],
        "samples": [
            {
                "label": row["label"],
                "elapsed_before": row["read_started"] - receipt["all_admitted_monotonic"],
                "elapsed_after": row["read_finished"] - receipt["all_admitted_monotonic"],
                "active": len(row["state"]["request_accounting"]["active_ids"]),
                "frontend": row["state"]["frontend_requests"],
                "native_first_token_timestamps": row["state"]["first_token_timestamps"],
            }
            for row in receipt["states"]
            if row["label"].startswith("delay_")
        ],
        "completion_elapsed": sorted(row["finished"] - receipt["all_admitted_monotonic"] for row in logical),
    }


def compose_precursor(spec, model_path):
    location = Path(__file__).parents[2] / "skyrl_train/config"
    overrides = spec["hydra_args"]
    encoded = json.dumps(overrides, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == spec["hydra_args_sha256"]
    with initialize_config_dir(config_dir=str(location), version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=overrides)
    assert cfg.generator.num_inference_engines == 8
    assert cfg.generator.max_num_seqs == 1024
    assert cfg.generator.sampling_params.ignore_eos
    assert cfg.generator.sampling_params.max_generate_length == 1024
    # Change only the independent engine count and the scheduler capacity.
    # Local materialization preserves the frozen snapshot identity.
    with initialize_config_dir(config_dir=str(location), version_base=None):
        return compose(
            config_name="ppo_base_config",
            overrides=overrides
            + [
                "++generator.num_inference_engines=1",
                "generator.max_num_seqs=4",
                f"trainer.policy.model.path={model_path}",
            ],
        )
