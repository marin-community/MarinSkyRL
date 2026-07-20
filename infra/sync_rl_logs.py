#!/usr/bin/env python3
"""sync_rl_logs.py — pull an Iris RL job's full debug log set (and optionally its
Harbor rollout artifacts) to a local dir.

For a given job it syncs:
  (1) ray_session_logs  — the per-actor Ray logs (worker-*.out/.err, python-*, raylet, gcs) from the
      durable object store (`s3://marin-us-east-02a/iris/<slug>/<run>/ray_session_logs/`, reached via
      the LOTA endpoint cwobject.com + the in-cluster `iris-task-env` creds + virtual addressing).
  (2) finelog.log       — the aggregated controller/job finelog via `iris job logs --no-tail`.
  (3) trace_jobs.tar.gz — [OPT-IN, --trace-jobs] the Harbor rollout artifacts (config/result/trajectory/
      verifier per trial) under `s3://marin-us-east-02a/iris/<slug>/trace_jobs/`. STREAMED straight into
      ONE tar[.gz] — trace_jobs can be millions of tiny files (see the harbor filesystem-footprint issue),
      so we never materialize the tree on the (unified-memory) Mac: one archive, not millions of inodes.

Ray logs land in the marin-us-east-02a bucket for BOTH east and rno2a jobs (it's east's LOTA store),
so the object-store creds always come from the EAST kubeconfig; the finelog is fetched per the job's
own cluster. trace_jobs is written under the job SLUG (run-independent), unlike ray_session_logs.

Usage:
  sync_rl_logs.py /benjaminfeuer/<job> [--cluster cw-us-east-02a|cw-rno2a]
                  [--run run-<ts>] [--dest DIR] [--finelog-lines N]
                  [--no-ray] [--no-finelog] [--trace-jobs] [--trace-jobs-no-gzip]

Defaults: cluster cw-us-east-02a; run = newest run-* under the job prefix; dest = ./<slug>-<run>;
trace_jobs OFF (opt in with --trace-jobs).
Re-runnable: existing same-size ray files are skipped, so re-syncing a live job only pulls new logs.
"""
import argparse
import base64
import io
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

BUCKET = "marin-us-east-02a"           # ray logs + trace_jobs land here for BOTH east + rno2a (east LOTA store)
ENDPOINT = "https://cwobject.com"
EAST_KUBECONFIG = os.path.expanduser("~/.kube/coreweave-iris-gpu")   # holds iris-task-env + the bucket
KCFG = {"cw-us-east-02a": "~/.kube/coreweave-iris-gpu", "cw-rno2a": "~/.kube/coreweave-iris"}
IRIS_CANDIDATES = [
    "/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris",
    "/Users/benjaminfeuer/Documents/marin/.venv/bin/iris",
    "iris",
]


def _secret(key, kubeconfig):
    env = {**os.environ, "KUBECONFIG": kubeconfig}
    out = subprocess.run(
        ["kubectl", "-n", "iris", "get", "secret", "iris-task-env", "-o", f"jsonpath={{.data.{key}}}"],
        capture_output=True, text=True, env=env,
    )
    if out.returncode != 0 or not out.stdout:
        sys.exit(f"[sync] cannot read iris-task-env/{key} (KUBECONFIG={kubeconfig}): {out.stderr.strip()}")
    return base64.b64decode(out.stdout).decode()


def s3client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=ENDPOINT,
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


def discover_run(s3, slug, run):
    if run:
        return run if run.startswith("run-") else f"run-{run}"
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"iris/{slug}/", Delimiter="/")
    runs = sorted(p["Prefix"].split("/")[-2] for p in r.get("CommonPrefixes", [])
                  if p["Prefix"].split("/")[-2].startswith("run-"))
    if not runs:
        sys.exit(f"[sync] no run-* under iris/{slug}/ in {BUCKET} — pass --run explicitly")
    return runs[-1]


def sync_ray(s3, slug, run, dest):
    pfx = f"iris/{slug}/{run}/ray_session_logs/"
    keys = _list_all(s3, pfx)
    print(f"[ray] {len(keys)} objects under {pfx}")
    if not keys:
        print("[ray] (none yet — the s3 upload lags for a fresh/live job; re-run shortly)")
        return 0
    outdir = os.path.join(dest, "ray_session_logs")

    def dl(item):
        k, sz = item
        d = os.path.join(outdir, k[len(pfx):])
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
    print(f"[trace] {len(keys)} objects under {pfx}")
    if not keys:
        print("[trace] (none yet — trace_jobs is written as trials complete; re-run once rollouts start)")
        return 0
    ext = "tar.gz" if gzip else "tar"
    tar_path = os.path.join(dest, f"{slug}_trace_jobs.{ext}")
    batch = 512                                            # cap in-flight bodies -> bounded memory
    total_bytes = 0

    def _body(item):
        return s3.get_object(Bucket=BUCKET, Key=item[0])["Body"].read()

    with tarfile.open(tar_path, "w:gz" if gzip else "w") as tar, ThreadPoolExecutor(max_workers=24) as ex:
        for i in range(0, len(keys), batch):
            chunk = keys[i:i + batch]
            for (k, sz), body in zip(chunk, ex.map(_body, chunk)):
                ti = tarfile.TarInfo(name=k[len(pfx):])   # path relative to trace_jobs/
                ti.size = len(body)
                tar.addfile(ti, io.BytesIO(body))
                total_bytes += len(body)
            print(f"[trace]   archived {min(i + batch, len(keys))}/{len(keys)} ...", end="\r", flush=True)
    print(f"\n[trace] {len(keys)} objects, {total_bytes / 1e6:.1f} MB uncompressed -> {tar_path} "
          f"({os.path.getsize(tar_path) / 1e6:.1f} MB on disk)")
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
            stdout=f, stderr=subprocess.PIPE, text=True, env=env,
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
    ap.add_argument("--run", default=None, help="rendezvous run-<ts> (default: newest under the job prefix)")
    ap.add_argument("--dest", default=None, help="output dir (default: ./<slug>-<run>)")
    ap.add_argument("--finelog-lines", type=int, default=500000)
    ap.add_argument("--no-ray", action="store_true")
    ap.add_argument("--no-finelog", action="store_true")
    ap.add_argument("--trace-jobs", action="store_true",
                    help="ALSO stream the Harbor trace_jobs artifacts into <dest>/<slug>_trace_jobs.tar.gz")
    ap.add_argument("--trace-jobs-no-gzip", action="store_true",
                    help="with --trace-jobs, write an uncompressed .tar instead of .tar.gz")
    a = ap.parse_args()
    slug = a.job.rstrip("/").split("/")[-1]
    s3 = s3client()
    # The finelog (via `iris job logs`) and trace_jobs (under the job SLUG) do NOT need the s3 run-dir;
    # only ray sync + the default dest name do. On a fresh job the s3 run-dir may not exist yet — don't
    # let that block --no-ray.
    run = None
    if not a.no_ray or a.dest is None:
        try:
            run = discover_run(s3, slug, a.run)
        except SystemExit:
            if not a.no_ray:
                raise  # ray sync genuinely needs it
            run = a.run or "pending"  # finelog/trace-only: proceed without the s3 run-dir
    dest = a.dest or os.path.join(os.getcwd(), f"{slug}-{run}")
    os.makedirs(dest, exist_ok=True)
    print(f"job={a.job}  cluster={a.cluster}  run={run}\ndest={dest}\n")
    if not a.no_ray:
        sync_ray(s3, slug, run, dest)
    if not a.no_finelog:
        sync_finelog(a.job, a.cluster, dest, a.finelog_lines)
    if a.trace_jobs:
        sync_trace_jobs(s3, slug, dest, gzip=not a.trace_jobs_no_gzip)
    print(f"\nDONE -> {dest}")


if __name__ == "__main__":
    main()
