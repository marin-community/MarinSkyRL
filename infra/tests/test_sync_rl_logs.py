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


def test_key_prefix_normalizes_every_uri_form():
    forms = [
        "s3://marin-us-east-02a/iris/rl-rdv/myjob",
        "gs://some-bucket/iris/rl-rdv/myjob",
        "iris/rl-rdv/myjob",
        "iris/rl-rdv/myjob/",
        "rl-rdv/myjob",
        "  s3://marin-us-east-02a/iris/rl-rdv/myjob/  ",
    ]
    for f in forms:
        assert srl._key_prefix_from_rendezvous(f) == "iris/rl-rdv/myjob", f


def test_rendezvous_dir_short_circuits_without_s3():
    """--rendezvous-dir resolves purely from the string; discovery is never invoked."""
    s3 = FakeS3(RUNS)
    prefix, label = srl.resolve_ray_prefix(
        s3, SLUG, run=None, rendezvous_dir="s3://marin-us-east-02a/iris/rl-rdv/myjob", finelog_path=None
    )
    assert prefix == f"iris/rl-rdv/myjob/{srl.RAY_SUBDIR}/"
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


def test_finelog_derive_from_bare_session_uri(tmp_path):
    """Fallback also works off a raw ray_session_logs URI the controller logged, not just the banner."""
    finelog = tmp_path / "finelog.log"
    finelog.write_text("uploading to s3://marin-us-east-02a/iris/rl-rdv/jobx/ray_session_logs/worker-0.out\n")
    assert srl._derive_rendezvous_from_finelog(str(finelog)) == "iris/rl-rdv/jobx"


def test_finelog_derive_missing_or_unmatched_is_none(tmp_path):
    assert srl._derive_rendezvous_from_finelog(None) is None
    assert srl._derive_rendezvous_from_finelog(str(tmp_path / "nope.log")) is None
    empty = tmp_path / "finelog.log"
    empty.write_text("nothing rendezvous-shaped here\n")
    assert srl._derive_rendezvous_from_finelog(str(empty)) is None


def test_unresolvable_returns_none(tmp_path):
    """No run-*, no --rendezvous-dir, no derivable finelog -> (None, None) so main() can error clearly."""
    s3 = FakeS3([])
    assert srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=None) == (None, None)
    blank = tmp_path / "finelog.log"
    blank.write_text("no rendezvous\n")
    assert srl.resolve_ray_prefix(s3, SLUG, run=None, rendezvous_dir=None, finelog_path=str(blank)) == (None, None)


# ---------------------------------------------------------------------------
# Store/creds resolution defaults — the object-store + kubeconfig wiring is untouched
# ---------------------------------------------------------------------------


def test_store_and_kubeconfig_defaults_include_every_gpu_cluster():
    assert srl.BUCKET == "marin-us-east-02a"
    assert srl.ENDPOINT == "https://cwobject.com"
    assert srl.EAST_KUBECONFIG.endswith("coreweave-iris")
    assert srl.RAY_SUBDIR == "ray_session_logs"
    assert srl.KCFG == {
        "cw-rno2a": "~/.kube/coreweave-iris",
        "cw-us-east-02a": "~/.kube/coreweave-iris",
        "cw-us-east-08a": "~/.kube/coreweave-iris",
    }
