"""Lean-only transport and compiled-environment inspection; contains no grader."""

import argparse
import json
import os
from pathlib import Path
import signal
import stat
import re
from functools import lru_cache
import subprocess
from tempfile import TemporaryDirectory
import time

import psutil

PROJECT = Path("/lean4/my_project")
COMPILER_UID = 65534
TASK_UID = 65533
MAX_TIMEOUT = 30


def kill_compilers(deadline, uid):
    # This dedicated serial runtime reserves this UID solely for candidate compilers.
    while True:
        processes = [
            p
            for p in psutil.process_iter(["uids", "status"])
            if p.info["uids"] is not None and p.info["uids"].real == uid and p.info["status"] != psutil.STATUS_ZOMBIE
        ]
        if not processes:
            return
        for process in processes:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError("candidate compiler UID did not become quiescent")
        psutil.wait_procs(processes, timeout=min(0.1, max(0, deadline - time.monotonic())))


def copy_artifact(source, target):
    # Only a bounded regular-file copy crosses the compiler/trusted-audit boundary.
    maximum = 16 * 1024 * 1024
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise RuntimeError("invalid compiled artifact")
        contents = stream.read(maximum + 1)
        if len(contents) > maximum:
            raise RuntimeError("compiled artifact exceeds limit")
    target.write_bytes(contents)
    target.chmod(0o444)


@lru_cache(maxsize=1)
def toolchain():
    # Resolve immutable binaries/search paths once, before unprivileged execution.
    prefix = subprocess.check_output(
        ["lake", "env", "lean", "--print-prefix"], cwd=PROJECT, text=True, timeout=10
    ).strip()
    paths = subprocess.check_output(
        ["lake", "env", "printenv", "LEAN_PATH"], cwd=PROJECT, text=True, timeout=10
    ).strip()
    absolute = [str(Path(path) if Path(path).is_absolute() else PROJECT / path) for path in paths.split(":")]
    return str(Path(prefix) / "bin/lean"), {
        "LEAN_PATH": ":".join(absolute),
        "PATH": str(Path(prefix) / "bin") + ":/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
    }


def execute(command, deadline, *, uid=None):
    binary, env = toolchain()
    command = [binary, *command[3:]] if command[:3] == ["lake", "env", "lean"] else command
    process = subprocess.Popen(
        command,
        cwd=PROJECT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
        **({"user": uid, "group": uid, "extra_groups": []} if uid is not None else {}),
    )
    try:
        stdout, stderr = process.communicate(timeout=max(0.001, deadline - time.monotonic()))
        return process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"), process.pid
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            if uid is not None:
                kill_compilers(time.monotonic() + 3, uid)
        finally:
            process.wait(timeout=1)


def compile_with_audit(proof, task, timeout, inspector):
    if os.getuid() != 0:
        raise RuntimeError("trusted runtime must control compiler UID and artifact ownership")
    deadline = time.monotonic() + timeout
    # The separate task declaration is compiler tooling, never candidate-authored.
    declaration = task["formal_statement"].rsplit(":=", 1)[0]
    declaration = re.sub(
        r"^\s*theorem\s+" + re.escape(task["theorem_name"]) + r"\b", "axiom AuditExpected", declaration, count=1
    )
    with TemporaryDirectory(prefix="lean-audit-", dir=PROJECT) as directory:
        root = Path(directory)
        root.chmod(0o755)
        trusted = root / "trusted"
        trusted.mkdir(mode=0o700)
        expected = root / "Trusted.lean"
        expected.write_text(task["header"] + declaration)
        expected.chmod(0o444)
        task_objects = root / "task_objects"
        task_objects.mkdir(mode=0o700)
        os.chown(task_objects, TASK_UID, TASK_UID)
        receipts = []
        try:
            code, stdout, stderr, pid = execute(
                ["lake", "env", "lean", "--root", str(root), "-o", str(task_objects / "Trusted.olean"), str(expected)],
                deadline,
                uid=TASK_UID,
            )
            receipts.append({"phase": "trusted_task", "pid": pid, "returncode": code})
            if code != 0:
                return {
                    "process_status": "error",
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_receipts": receipts,
                    "error_type": "trusted_task_compilation",
                }
            copy_artifact(task_objects / "Trusted.olean", trusted / "Trusted.olean")
            source = root / "Candidate.lean"
            source.write_text(proof)
            source.chmod(0o444)
            objects = root / "objects"
            objects.mkdir()
            os.chown(objects, COMPILER_UID, COMPILER_UID)
            code, stdout, stderr, pid = execute(
                ["lake", "env", "lean", "--root", str(root), "-o", str(objects / "Candidate.olean"), str(source)],
                deadline,
                uid=COMPILER_UID,
            )
            receipts.append({"phase": "compile", "pid": pid, "returncode": code})
            output = {
                "process_status": "completed" if code == 0 else "failed",
                "stdout": stdout,
                "stderr": stderr,
                "execution_receipts": receipts,
            }
            if code != 0:
                return output
            # No candidate UID processes remain; nofollow-copy into root-only storage.
            copy_artifact(objects / "Candidate.olean", trusted / "Candidate.olean")
            code, stdout, stderr, pid = execute(
                [
                    "lake",
                    "env",
                    "lean",
                    "--run",
                    str(inspector),
                    "Candidate",
                    task["theorem_name"],
                    str(trusted),
                    "Trusted",
                ],
                deadline,
            )
            receipts.append({"phase": "audit", "pid": pid, "returncode": code})
            output["lean_audit"] = json.loads(stdout) if code == 0 else {"process_status": "error", "stderr": stderr}
            return output
        except subprocess.TimeoutExpired:
            return {
                "process_status": "timeout",
                "stdout": "",
                "stderr": "Lean compile/audit total deadline expired",
                "execution_receipts": receipts,
            }


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspector", required=True, type=Path)
    args = parser.parse_args()
    toolchain()
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/execute")
    def run():
        data = request.get_json()
        if data["language"] != "lean4":
            return jsonify(process_status="error", stdout="", stderr="Lean-only runtime"), 400
        if "lean_audit" not in data:
            return jsonify(process_status="error", stdout="", stderr="Trusted audit task is required"), 400
        if (
            isinstance(data["timeout"], bool)
            or not isinstance(data["timeout"], (int, float))
            or not 0 < data["timeout"] <= MAX_TIMEOUT
        ):
            return jsonify(process_status="error", stdout="", stderr="Unsupported compile/audit deadline"), 400
        result = compile_with_audit(data["generated_code"], data["lean_audit"], data["timeout"], args.inspector)
        maximum = data.get("max_output_characters", 65536)
        for key in ("stdout", "stderr"):
            if len(result[key]) > maximum:
                result[key] = result[key][:maximum] + "<output cut>"
                result["output_truncated"] = True
        return jsonify(result)

    app.run(host="0.0.0.0", port=6000, threaded=False)


if __name__ == "__main__":
    serve()
