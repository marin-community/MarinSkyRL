"""Read retained SkyRL rollouts without loading a complete profiling run into RAM."""

import gzip
import json
from pathlib import Path
from zipfile import ZipFile


def iter_records(path: Path):
    """Read a JSONL file, one retention archive, or a local directory of archives."""
    if path.is_dir():
        archives = sorted(path.rglob("*.zip"))
        if not archives:
            raise ValueError(f"No trajectory retention archives in {path}")
        for archive in archives:
            yield from iter_records(archive)
    elif path.suffix == ".zip":
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for entry in manifest["records"]:
                yield json.loads(gzip.decompress(archive.read(entry["entry"])))
    else:
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)
