#!/usr/bin/env python3
"""Mirror one Hugging Face revision to the regional model store."""

from __future__ import annotations

import argparse
import json

from cloud.iris.hf_model_cache import ensure_hugging_face_model_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Hugging Face repository ID")
    parser.add_argument("--revision", default="main", help="Commit, tag, or branch to resolve once")
    parser.add_argument("--source-prefix", required=True, help="S3/GS path used to select the mirror region")
    parser.add_argument("--ttl-days", type=int, default=30)
    args = parser.parse_args(argv)
    uri, manifest = ensure_hugging_face_model_cache(
        args.model_id,
        args.revision,
        ttl_days=args.ttl_days,
        source_prefix=args.source_prefix,
    )
    print(
        json.dumps(
            {
                "identity": manifest.identity,
                "model_id": manifest.model_id,
                "revision": manifest.revision,
                "uri": uri,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
