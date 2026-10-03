"""Pin the exact #860 checkout bytes shipped in this task-only Iris bundle."""

import hashlib
import json
import subprocess
from pathlib import Path

from iris.cluster.client.bundle import collect_workspace_files


MANIFEST = "hero-final-source-manifest.json"


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=root, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in collect_workspace_files(root)
        if path.name != MANIFEST
    ]
    manifest = root / MANIFEST
    manifest.write_text(json.dumps({"source_commit": commit, "files": files}, sort_keys=True) + "\n")
    print(json.dumps({"commit": commit, "files": len(files), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
