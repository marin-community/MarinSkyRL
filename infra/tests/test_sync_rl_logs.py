"""Ray-log-prefix resolution for sync_rl_logs.py.

The point of this suite is a firewall: the non-agentic rendezvous support (`--rendezvous-dir`
and the finelog auto-derive) must not perturb how an AGENTIC job's ray logs are resolved. The
agentic-path tests below pin that default end to end; the non-agentic tests cover the new paths.
"""

import sync_rl_logs as srl


class FakeS3:
    """Minimal stand-in for the boto3 client — only `list_objects_v2` with a Delimiter, which is
    all `discover_run` uses. Records call count so tests can assert the agentic short-circuits
    (explicit --run / --rendezvous-dir) never hit the object store at all."""

    def __init__(self, run_dirs):
        self._run_dirs = run_dirs
        self.calls = 0

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None):  # noqa: N803 (boto3 kwarg names)
        self.calls += 1
        return {"CommonPrefixes": [{"Prefix": f"{Prefix}{d}/"} for d in self._run_dirs]}


SLUG = "delphi-1e23-wc50m-rl-d1-rlvrmath"
RUNS = ["run-20260721-1200", "run-20260721-1330", "run-20260721-0900"]  # newest = ...-1330 (lexical == chrono)


# ---------------------------------------------------------------------------
# Agentic path — must stay exactly as it was before rendezvous support existed
# ---------------------------------------------------------------------------


def test_agentic_default_autodiscovers_newest_run():
    """No --run, no --rendezvous-dir: resolve to the newest run-* under iris/<slug>/, unchanged."""
    s3 = FakeS3(RUNS)
    prefix, label = srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=None)
    assert prefix == f"iris/{SLUG}/run-20260721-1330/{srl.RAY_SUBDIR}/"
    assert label == "run-20260721-1330"


def test_agentic_explicit_run_does_not_touch_s3():
    """An explicit --run short-circuits discovery entirely (no list call)."""
    s3 = FakeS3(RUNS)
    prefix, label = srl.resolve_ray_prefix(s3, SLUG, run="run-20260721-1200", rendezvous_dir=None, finelog_path=None)
    assert prefix == f"iris/{SLUG}/run-20260721-1200/{srl.RAY_SUBDIR}/"
    assert label == "run-20260721-1200"
    assert s3.calls == 0


def test_agentic_run_without_prefix_is_normalized():
    """A bare timestamp (no `run-`) is normalized to run-<ts>, matching pre-change discover_run."""
    s3 = FakeS3(RUNS)
    prefix, _ = srl.resolve_ray_prefix(s3, SLUG, run="20260721-1200", rendezvous_dir=None, finelog_path=None)
    assert prefix == f"iris/{SLUG}/run-20260721-1200/{srl.RAY_SUBDIR}/"


def test_agentic_ignores_finelog_when_run_dir_exists(tmp_path):
    """Firewall: even with a rendezvous line sitting in the finelog, a discoverable run-* wins and the
    finelog is never consulted — the fallback cannot hijack an agentic resolution."""
    finelog = tmp_path / "finelog.log"
    finelog.write_text("[rl-iris] Rendezvous: s3://marin-us-east-02a/iris/rl-rdv/some-other-job\n")
    s3 = FakeS3(RUNS)
    prefix, _ = srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=str(finelog))
    assert prefix == f"iris/{SLUG}/run-20260721-1330/{srl.RAY_SUBDIR}/"
    assert "rl-rdv" not in prefix


def test_discover_run_picks_newest_and_ignores_non_run_dirs():
    s3 = FakeS3(["trace_jobs", "run-20260721-0900", "checkpoints", "run-20260721-1330"])
    assert srl.discover_run(s3, SLUG) == "run-20260721-1330"


def test_discover_run_returns_none_when_no_run_dirs():
    assert srl.discover_run(FakeS3(["trace_jobs", "checkpoints"]), SLUG) is None


# ---------------------------------------------------------------------------
# Non-agentic rendezvous path — the new behavior
# ---------------------------------------------------------------------------


def test_object_key_prefix_normalizes_every_uri_form():
    forms = [
        "s3://marin-us-east-02a/iris/rl-rdv/myjob",
        "gs://some-bucket/iris/rl-rdv/myjob",
        "iris/rl-rdv/myjob",
        "iris/rl-rdv/myjob/",
        "rl-rdv/myjob",
        "  s3://marin-us-east-02a/iris/rl-rdv/myjob/  ",
    ]
    for f in forms:
        assert srl._object_key_prefix(f) == "iris/rl-rdv/myjob", f


def test_object_key_prefix_preserves_lifecycle_managed_object_key():
    rendezvous = "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/alice/job/rendezvous"

    assert srl._object_key_prefix(rendezvous) == "tmp/ttl=14d/skyrl/users/alice/job/rendezvous"


def test_rendezvous_dir_short_circuits_without_s3():
    """--rendezvous-dir resolves purely from the string; discovery is never invoked."""
    s3 = FakeS3(RUNS)
    prefix, label = srl.resolve_ray_prefix(
        s3, SLUG, run=None, rendezvous_dir="s3://marin-us-east-02a/iris/rl-rdv/myjob", finelog_path=None
    )
    assert prefix == f"iris/rl-rdv/myjob/{srl.RAY_SUBDIR}/"
    assert label == "myjob"
    assert s3.calls == 0


def test_ray_log_dir_short_circuits_without_s3():
    s3 = FakeS3(RUNS)
    prefix, label = srl.resolve_ray_prefix(
        s3,
        SLUG,
        run=None,
        rendezvous_dir=None,
        finelog_path=None,
        ray_log_dir="s3://marin-us-east-02a/marin/users/alice/skyrl/myjob/ray_session_logs",
    )

    assert prefix == "marin/users/alice/skyrl/myjob/ray_session_logs/"
    assert label == "myjob"
    assert s3.calls == 0


def test_finelog_derive_from_rendezvous_banner(tmp_path):
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "some setup line\n[rl-iris] Rendezvous: s3://marin-us-east-02a/iris/rl-rdv/delphi-rl-d1\ntraining started\n"
    )
    s3 = FakeS3([])  # no run-* -> non-agentic
    prefix, label = srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=str(finelog))
    assert prefix == f"iris/rl-rdv/delphi-rl-d1/{srl.RAY_SUBDIR}/"
    assert label == "delphi-rl-d1"


def test_finelog_derive_from_ray_log_banner(tmp_path):
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "[rl-iris] Ray logs:   s3://marin-us-east-02a/marin/users/alice/skyrl/delphi-rl-d1/ray_session_logs\n"
    )

    prefix, label = srl.resolve_ray_prefix(FakeS3([]), SLUG, None, None, str(finelog))

    assert prefix == "marin/users/alice/skyrl/delphi-rl-d1/ray_session_logs/"
    assert label == "delphi-rl-d1"


def test_ray_log_banner_wins_over_historical_agentic_run(tmp_path):
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "[rl-iris] Ray logs:   s3://marin-us-east-02a/marin/users/alice/skyrl/current-job/ray_session_logs\n"
    )

    prefix, label = srl.resolve_ray_prefix(FakeS3(RUNS), SLUG, RUNS[0], None, str(finelog))

    assert prefix == "marin/users/alice/skyrl/current-job/ray_session_logs/"
    assert label == "current-job"


def test_finelog_derive_from_bare_session_uri(tmp_path):
    """Fallback also works off a raw ray_session_logs URI the controller logged, not just the banner."""
    finelog = tmp_path / "finelog.log"
    finelog.write_text("uploading to s3://marin-us-east-02a/iris/rl-rdv/jobx/ray_session_logs/worker-0.out\n")

    _, legacy_prefix = srl._ray_prefixes_from_finelog(str(finelog))

    assert legacy_prefix == "iris/rl-rdv/jobx/ray_session_logs/"


def test_finelog_derive_missing_or_unmatched_is_none(tmp_path):
    assert srl._ray_prefixes_from_finelog(None) == (None, None)
    assert srl._ray_prefixes_from_finelog(str(tmp_path / "nope.log")) == (None, None)
    empty = tmp_path / "finelog.log"
    empty.write_text("nothing rendezvous-shaped here\n")
    assert srl._ray_prefixes_from_finelog(str(empty)) == (None, None)


def test_unresolvable_returns_none(tmp_path):
    """No run-*, no --rendezvous-dir, no derivable finelog -> (None, None) so main() can error clearly."""
    s3 = FakeS3([])
    assert srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=None) == (None, None)
    blank = tmp_path / "finelog.log"
    blank.write_text("no rendezvous\n")
    assert srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=str(blank)) == (None, None)


# ---------------------------------------------------------------------------
# Store and credential resolution
# ---------------------------------------------------------------------------


def test_cli_accepts_east08_cluster():
    args = srl.argument_parser().parse_args(["/operator/job", "--cluster", "cw-us-east-08a"])

    assert args.cluster == "cw-us-east-08a"


def test_trace_batches_limit_total_download_bytes_and_report_oversized_non_logs():
    objects = [
        ("iris/job/trace_jobs/a/result.json", 40),
        ("iris/job/trace_jobs/a/agent/trajectory.json", 40),
        ("iris/job/trace_jobs/a/large.bin", 101),
    ]

    batches, skipped = srl.trace_object_batches(objects, batch_bytes=64, max_non_log_bytes=100)

    assert batches == [
        [("iris/job/trace_jobs/a/result.json", 40)],
        [("iris/job/trace_jobs/a/agent/trajectory.json", 40)],
    ]
    assert skipped == [{"key": "iris/job/trace_jobs/a/large.bin", "size": 101, "reason": "non_log_size_limit"}]


def test_trace_batches_keep_large_log_objects_when_non_log_guard_is_enabled():
    objects = [("iris/job/trace_jobs/a/opencode.txt", 120)]

    batches, skipped = srl.trace_object_batches(objects, batch_bytes=64, max_non_log_bytes=100)

    assert batches == [[("iris/job/trace_jobs/a/opencode.txt", 120)]]
    assert skipped == []
