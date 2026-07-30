#!/usr/bin/env python3
"""sync_rl_logs.py — pull an Iris RL job's full debug log set (and optionally its
Harbor rollout artifacts) to a local dir.

For a given job it syncs:
  (1) ray_session_logs  — the per-actor Ray logs (worker-*.out/.err, python-*, raylet, gcs) from the
      durable object store, reached via the LOTA endpoint cwobject.com + the in-cluster `iris-task-env`
      creds + virtual addressing. Two layouts are supported (see "Ray-log layouts" below):
        - agentic:     `s3://marin-us-east-02a/iris/<slug>/<run>/ray_session_logs/`
        - non-agentic: `s3://marin-us-east-02a/iris/<rendezvous>/ray_session_logs/` (e.g. `iris/rl-rdv/<job>/`)
  (2) finelog.log       — the aggregated controller/job finelog via `iris job logs --no-tail`.
  (3) trace_jobs.tar.gz — [OPT-IN, --trace-jobs] the Harbor rollout artifacts (config/result/trajectory/
      verifier per trial) under `s3://marin-us-east-02a/iris/<slug>/trace_jobs/`. STREAMED straight into
      ONE tar[.gz] — trace_jobs can be millions of tiny files (see the harbor filesystem-footprint issue),
      so we never materialize the tree on the (unified-memory) Mac: one archive, not millions of inodes.

Ray-log layouts:
  Agentic jobs publish per-run Ray logs under `iris/<slug>/run-<ts>/ray_session_logs/`; the run dir is
  auto-discovered (newest `run-*`). Non-agentic multi-node RL jobs instead rendezvous through a shared
  `--rendezvous-dir` (`launch_rl_iris.py`, e.g. `s3://marin-us-east-02a/iris/rl-rdv/<job>`) and write
  their Ray logs under THAT prefix — there is no `run-*` dir. For those, pass `--rendezvous-dir` with the
  same URI you launched with, or let the tool auto-derive it from the finelog (the launcher prints
  `Rendezvous: <uri>`), so non-agentic jobs sync with no extra flags as long as the finelog is fetched.

Ray logs land in the marin-us-east-02a bucket for every supported CoreWeave cluster, so the object-store
creds always come from the east-02a store; the finelog is fetched from the job's own cluster. trace_jobs
is written under the job SLUG (run-independent), unlike ray_session_logs.

Usage:
  sync_rl_logs.py /benjaminfeuer/<job> [--cluster cw-us-east-02a|cw-us-east-08a|cw-rno2a]
                  [--run run-<ts>] [--rendezvous-dir URI] [--dest DIR] [--finelog-lines N]
                  [--no-ray] [--no-finelog] [--trace-jobs] [--trace-jobs-no-gzip]

Defaults: cluster cw-us-east-02a; agentic run = newest run-* under the job prefix; non-agentic
rendezvous auto-derived from the finelog; dest = ./<slug>-<run|rendezvous>; trace_jobs OFF.
Re-runnable: existing same-size ray files are skipped, so re-syncing a live job only pulls new logs.
"""

import argparse
import base64
import io
import os
import re
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

BUCKET = "marin-us-east-02a"  # shared CoreWeave ray-log and trace-job store
ENDPOINT = "https://cwobject.com"
EAST_KUBECONFIG = os.path.expanduser("~/.kube/coreweave-iris")  # holds iris-task-env + the bucket
KCFG = {
    "cw-rno2a": "~/.kube/coreweave-iris",
    "cw-us-east-02a": "~/.kube/coreweave-iris",
    "cw-us-east-08a": "~/.kube/coreweave-iris",
}
RAY_SUBDIR = "ray_session_logs"  # the leaf under both the agentic run dir and the rendezvous dir
IRIS_CANDIDATES = [
    "/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris",
    "/Users/benjaminfeuer/Documents/marin/.venv/bin/iris",
    "iris",
]


def _secret(key, kubeconfig):
    env = {**os.environ, "KUBECONFIG": kubeconfig}
    out = subprocess.run(
        ["kubectl", "-n", "iris", "get", "secret", "iris-task-env", "-o", f"jsonpath={{.data.{key}}}"],
        capture_output=True,
        text=True,
        env=env,
    )
    if out.returncode != 0 or not out.stdout:
        sys.exit(f"[sync] cannot read iris-task-env/{key} (KUBECONFIG={kubeconfig}): {out.stderr.strip()}")
    return base64.b64decode(out.stdout).decode()


def s3client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=_secret("AWS_ACCESS_KEY_ID", EAST_KUBECONFIG),
        aws_secret_access_key=_secret("AWS_SECRET_ACCESS_KEY", EAST_KUBECONFIG),
        config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=32),
    )


def _list_all(s3, pfx):
    """Every (key, size) under a prefix, following pagination."""
    keys = []
    tok = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=pfx)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        keys += [(c["Key"], c["Size"]) for c in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            break
        tok = r["NextContinuationToken"]
    return keys


def discover_run(s3, slug):
    """Newest `run-*` dir under `iris/<slug>/` (agentic per-run layout), or None if there is none.

    None means the job did not use the agentic per-run layout — a non-agentic RL job rendezvouses
    under its own `--rendezvous-dir` instead (see resolve_ray_prefix / _derive_rendezvous_from_finelog).
    """
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"iris/{slug}/", Delimiter="/")
    runs = sorted(
        p["Prefix"].split("/")[-2] for p in r.get("CommonPrefixes", []) if p["Prefix"].split("/")[-2].startswith("run-")
    )
    return runs[-1] if runs else None


def _key_prefix_from_rendezvous(rdv):
    """Normalize a rendezvous URI/path to its object-store key prefix (no scheme, no bucket, no
    trailing slash), anchored at `iris/`.

    Accepts every form the launcher / operator might pass:
      s3://marin-us-east-02a/iris/rl-rdv/<job>  ->  iris/rl-rdv/<job>
      gs://some-bucket/iris/rl-rdv/<job>        ->  iris/rl-rdv/<job>
      iris/rl-rdv/<job>                         ->  iris/rl-rdv/<job>
      rl-rdv/<job>                              ->  iris/rl-rdv/<job>
    """
    p = re.sub(r"^[a-z0-9]+://", "", rdv.strip()).strip("/")  # drop scheme
    i = p.find("iris/")
    if i != -1:  # drop a leading bucket segment
        p = p[i:]
    elif not p.startswith("iris/"):  # bare `rl-rdv/<job>`
        p = f"iris/{p}"
    return p.rstrip("/")


def _derive_rendezvous_from_finelog(finelog_path):
    """Parse a fetched finelog for the launcher's rendezvous location and return its normalized
    `iris/...` key prefix, or None. Matches either the launcher's `Rendezvous: <uri>` banner line
    or any `<scheme>://.../ray_session_logs` URI the controller logged."""
    if not finelog_path or not os.path.exists(finelog_path):
        return None
    pat_rdv = re.compile(r"Rendezvous:\s*(\S+)")
    pat_uri = re.compile(r"[a-z0-9]+://[^\s'\"]+/" + RAY_SUBDIR + r"\b")
    with open(finelog_path, errors="replace") as f:
        for line in f:
            m = pat_rdv.search(line)
            if m:
                return _key_prefix_from_rendezvous(m.group(1))
            m = pat_uri.search(line)
            if m:
                return _key_prefix_from_rendezvous(m.group(0).rsplit("/" + RAY_SUBDIR, 1)[0])
    return None


def resolve_ray_prefix(s3, slug, run, rendezvous_dir, finelog_path):
    """Resolve the `iris/.../ray_session_logs/` key prefix for a job's Ray logs, plus a short label
    (used for the default dest name). Returns (prefix, label) or (None, None) if unresolvable.

    Resolution order — the agentic per-run path (2 & 3) is unchanged; the rendezvous paths (1 & 4)
    are the non-agentic additions:
      1. explicit --rendezvous-dir            → iris/<rendezvous>/ray_session_logs/   (non-agentic)
      2. explicit --run                        → iris/<slug>/run-<ts>/ray_session_logs/ (agentic)
      3. auto-discovered newest run-*          → iris/<slug>/run-<ts>/ray_session_logs/ (agentic default)
      4. rendezvous derived from the finelog   → iris/<rendezvous>/ray_session_logs/   (non-agentic fallback)
    """
    if rendezvous_dir:
        base = _key_prefix_from_rendezvous(rendezvous_dir)
        return f"{base}/{RAY_SUBDIR}/", base.split("/")[-1]
    if run:
        r = run if run.startswith("run-") else f"run-{run}"
        return f"iris/{slug}/{r}/{RAY_SUBDIR}/", r
    r = discover_run(s3, slug)
    if r:
        return f"iris/{slug}/{r}/{RAY_SUBDIR}/", r
    base = _derive_rendezvous_from_finelog(finelog_path)
    if base:
        return f"{base}/{RAY_SUBDIR}/", base.split("/")[-1]
    return None, None


RAY_SOURCE_MARKER = ".synced_from"


def _guard_ray_dest(outdir, prefix):
    """Refuse to mirror a second job's Ray logs into a dest that already holds another job's.

    Unlike the finelog, which is rewritten each run, the Ray mirror is ADDITIVE — it copies objects
    in and never removes what is already there. Reusing one ``--dest`` for two jobs therefore MERGES
    their ``rank*-<node>`` directories, and the result looks like one job that ran on twice the nodes.
    That has repeatedly misled readers into diagnosing the wrong job. A stamp of the source prefix
    makes the collision loud instead of silent.
    """
    marker = os.path.join(outdir, RAY_SOURCE_MARKER)
    if os.path.exists(marker):
        with open(marker) as f:
            previous = f.read().strip()
        if previous and previous != prefix:
            raise SystemExit(
                f"[ray] REFUSING to mirror into {outdir}: it already holds Ray logs from a different "
                f"job.\n      existing: {previous}\n      requested: {prefix}\n"
                "      The Ray mirror is additive, so this would merge two jobs' rank dirs into one "
                "tree and make the capture unreadable. Use a per-job --dest."
            )
    os.makedirs(outdir, exist_ok=True)
    with open(marker, "w") as f:
        f.write(prefix + "\n")


def sync_ray(s3, prefix, dest):
    """Mirror every object under the resolved ray_session_logs prefix into <dest>/ray_session_logs/."""
    keys = _list_all(s3, prefix)
    print(f"[ray] {len(keys)} objects under {prefix}")
    if not keys:
        print("[ray] (none yet — the s3 upload lags for a fresh/live job; re-run shortly)")
        return 0
    outdir = os.path.join(dest, RAY_SUBDIR)
    _guard_ray_dest(outdir, prefix)

    def dl(item):
        k, sz = item
        d = os.path.join(outdir, k[len(prefix) :])
        if os.path.exists(d) and os.path.getsize(d) == sz:  # idempotent skip
            return 0
        os.makedirs(os.path.dirname(d), exist_ok=True)
        s3.download_file(BUCKET, k, d)
        return 1

    n = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        for got in ex.map(dl, keys):
            n += got
    print(f"[ray] {n} new / {len(keys)} total -> {outdir}")
    return len(keys)


def sync_trace_jobs(s3, slug, dest, gzip=True):
    """Stream every object under iris/<slug>/trace_jobs/ into ONE local tar[.gz].

    trace_jobs is the Harbor rollout-artifact tree (per trial: config.json, result.json,
    opencode.txt, exception.txt, agent/trajectory.json, verifier/…) and can be MILLIONS of
    tiny files. We stream each S3 object straight into a single tar archive rather than
    downloading the tree — one inode on the Mac instead of millions (harbor-footprint /
    unified-memory-OOM discipline). Fetches are concurrent but bounded to one batch in
    memory at a time; tar writes are serialized (a tarfile handle is not thread-safe).
    """
    pfx = f"iris/{slug}/trace_jobs/"
    keys = _list_all(s3, pfx)
    # Skip tmux pane captures (terminus_2.pane / opencode .pane): huge full-terminal scrollback
    # dumps, useless for rollout analysis (config/result/trajectory/verifier are the signal). Also
    # dodges a NoSuchKey race when a LIVE job rotates/removes a .pane between listing and fetch.
    _n0 = len(keys)
    keys = [item for item in keys if not item[0].endswith(".pane")]
    _skipped = _n0 - len(keys)
    print(f"[trace] {len(keys)} objects under {pfx}" + (f"  (skipped {_skipped} .pane captures)" if _skipped else ""))
    if not keys:
        print("[trace] (none yet — trace_jobs is written as trials complete; re-run once rollouts start)")
        return 0
    ext = "tar.gz" if gzip else "tar"
    tar_path = os.path.join(dest, f"{slug}_trace_jobs.{ext}")
    batch = 512  # cap in-flight bodies -> bounded memory
    total_bytes = 0

    def _body(item):
        return s3.get_object(Bucket=BUCKET, Key=item[0])["Body"].read()

    with tarfile.open(tar_path, "w:gz" if gzip else "w") as tar, ThreadPoolExecutor(max_workers=24) as ex:
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            for (k, sz), body in zip(chunk, ex.map(_body, chunk)):
                ti = tarfile.TarInfo(name=k[len(pfx) :])  # path relative to trace_jobs/
                ti.size = len(body)
                tar.addfile(ti, io.BytesIO(body))
                total_bytes += len(body)
            print(f"[trace]   archived {min(i + batch, len(keys))}/{len(keys)} ...", end="\r", flush=True)
    print(
        f"\n[trace] {len(keys)} objects, {total_bytes / 1e6:.1f} MB uncompressed -> {tar_path} "
        f"({os.path.getsize(tar_path) / 1e6:.1f} MB on disk)"
    )
    return len(keys)


def sync_finelog(job, cluster, dest, lines):
    kcfg = os.path.expanduser(KCFG[cluster])
    env = {**os.environ, "KUBECONFIG": kcfg}
    iris = next((c for c in IRIS_CANDIDATES if c == "iris" or os.path.exists(c)), "iris")
    out_path = os.path.join(dest, "finelog.log")
    print(f"[finelog] {job} via {os.path.basename(iris)} --cluster={cluster} --no-tail --max-lines {lines} ...")
    with open(out_path, "w") as f:
        p = subprocess.run(
            [iris, f"--cluster={cluster}", "job", "logs", job, "--no-tail", "--max-lines", str(lines)],
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    n = sum(1 for _ in open(out_path, errors="replace"))
    note = f"  (rc={p.returncode}: {p.stderr.strip()[:200]})" if p.returncode else ""
    print(f"[finelog] {n} lines -> {out_path}{note}")
    if n >= lines:
        print(f"[finelog] ⚠ hit the --finelog-lines cap ({lines}) — raise it for a longer job")
    return n


def main():
    ap = argparse.ArgumentParser(description="Sync an Iris RL job's ray_session_logs + finelog (+ trace_jobs) locally.")
    ap.add_argument("job", help="full job id, e.g. /benjaminfeuer/rl-tasktrove-keep1-ncclnet")
    ap.add_argument("--cluster", default="cw-us-east-02a", choices=list(KCFG))
    ap.add_argument("--run", default=None, help="agentic rendezvous run-<ts> (default: newest under the job prefix)")
    ap.add_argument(
        "--rendezvous-dir",
        "--rendezvous_dir",
        dest="rendezvous_dir",
        default=None,
        help="non-agentic RL rendezvous URI (the same one passed to launch_rl_iris.py, e.g. "
        "s3://marin-us-east-02a/iris/rl-rdv/<job>); its ray_session_logs are synced. "
        "Omit to auto-derive from the finelog.",
    )
    ap.add_argument("--dest", default=None, help="output dir (default: ./<slug>-<run|rendezvous>)")
    ap.add_argument("--finelog-lines", type=int, default=500000)
    ap.add_argument("--no-ray", action="store_true")
    ap.add_argument("--no-finelog", action="store_true")
    ap.add_argument(
        "--trace-jobs",
        action="store_true",
        help="ALSO stream the Harbor trace_jobs artifacts into <dest>/<slug>_trace_jobs.tar.gz",
    )
    ap.add_argument(
        "--trace-jobs-no-gzip",
        action="store_true",
        help="with --trace-jobs, write an uncompressed .tar instead of .tar.gz",
    )
    a = ap.parse_args()
    slug = a.job.rstrip("/").split("/")[-1]
    s3 = s3client()

    # Resolve the run/rendezvous label up front — it names the default dest and (for the agentic
    # per-run layout) is the one s3 list needed to find the ray logs. discover_run is called at most
    # once; a non-agentic job (no run-*, no --rendezvous-dir) resolves to None here and is derived
    # from the finelog after it is fetched, below.
    run = a.run
    if not run and not a.rendezvous_dir and (not a.no_ray or a.dest is None):
        run = discover_run(s3, slug)
    if a.rendezvous_dir:
        label = _key_prefix_from_rendezvous(a.rendezvous_dir).split("/")[-1]
    else:
        label = run
    dest = a.dest or os.path.join(os.getcwd(), f"{slug}-{label}" if label else slug)
    os.makedirs(dest, exist_ok=True)
    finelog_path = os.path.join(dest, "finelog.log")
    print(f"job={a.job}  cluster={a.cluster}  run={run}  rendezvous={a.rendezvous_dir}\ndest={dest}\n")

    # Finelog FIRST: it is the input to the non-agentic ray-prefix fallback (the launcher prints the
    # rendezvous URI there), and it never depends on the ray run-dir.
    if not a.no_finelog:
        sync_finelog(a.job, a.cluster, dest, a.finelog_lines)

    if not a.no_ray:
        prefix, _ = resolve_ray_prefix(s3, slug, run, a.rendezvous_dir, finelog_path if not a.no_finelog else None)
        if not prefix:
            sys.exit(
                f"[sync] could not locate ray_session_logs: no run-* under iris/{slug}/, no "
                "--rendezvous-dir, and none derivable from the finelog. Pass --rendezvous-dir "
                "s3://marin-us-east-02a/iris/rl-rdv/<job> (the URI you launched with)."
            )
        sync_ray(s3, prefix, dest)
    if a.trace_jobs:
        sync_trace_jobs(s3, slug, dest, gzip=not a.trace_jobs_no_gzip)
    print(f"\nDONE -> {dest}")


if __name__ == "__main__":
    main()
