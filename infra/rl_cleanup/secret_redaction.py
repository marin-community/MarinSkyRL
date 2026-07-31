"""Redact credential-shaped text before publishing model or trace artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    """One credential-shaped value and its location."""

    shape: str
    location: str
    line: int | None = None


REDACTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pem_private_key", re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.DOTALL)),
    ("iris_proxy_url", re.compile(r"https?://iris\.oa\.dev/proxy/t/[^\s'\"/]+")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
)


def redact(text: str, *, location: str = "text") -> tuple[str, list[Finding]]:
    """Replace known credential shapes with stable markers."""
    findings: list[Finding] = []
    for shape, pattern in REDACTIONS:

        def replace(match: re.Match[str]) -> str:
            findings.append(Finding(shape, location, text.count("\n", 0, match.start()) + 1))
            return f"<redacted:{shape}>"

        text = pattern.sub(replace, text)
    return text, findings


def redact_record(record: dict[str, Any], *, location: str = "") -> tuple[dict[str, Any], list[Finding]]:
    """Return a recursively redacted copy of one JSON-compatible trace record."""
    findings: list[Finding] = []

    def visit(value: Any, value_location: str) -> Any:
        if isinstance(value, str):
            cleaned, found = redact(value, location=value_location)
            findings.extend(found)
            return cleaned
        if isinstance(value, list):
            return [visit(item, f"{value_location}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {
                key: visit(item, f"{value_location}.{key}" if value_location else key) for key, item in value.items()
            }
        return value

    cleaned = visit(record, location)
    if not isinstance(cleaned, dict):
        raise ValueError("Trace record redaction produced a non-mapping result")
    return cleaned, findings


def scan_tree(root: Path) -> list[Finding]:
    """Scan UTF-8 text files in a staging tree without changing them."""
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        _, found = redact(text, location=str(path.relative_to(root)))
        findings.extend(found)
    return findings


def redact_tree(root: Path) -> list[Finding]:
    """Redact UTF-8 text files in a staging tree in place."""
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        cleaned, found = redact(text, location=str(path.relative_to(root)))
        if found:
            path.write_text(cleaned, encoding="utf-8")
            findings.extend(found)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Redact or check credential-shaped text in a staging tree.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true", help="Report findings without modifying files and fail if any remain.")
    args = parser.parse_args()
    findings = scan_tree(args.root) if args.check else redact_tree(args.root)
    print(json.dumps({"findings": [asdict(finding) for finding in findings]}, indent=2))
    if args.check and findings:
        raise SystemExit("Credential-shaped text remains in the checked tree")


if __name__ == "__main__":
    main()
