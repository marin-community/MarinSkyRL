# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client for NVIDIA NeMo Skills' local execution sandbox protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class SandboxClient:
    host: str = "127.0.0.1"
    port: int = 6000

    def execute(
        self,
        code: str,
        *,
        language: str,
        timeout_seconds: float,
        session_id: str | None = None,
        max_output_characters: int = 1000,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if session_id is not None:
            headers["X-Session-ID"] = session_id
        response = requests.post(
            f"http://{self.host}:{self.port}/execute",
            headers=headers,
            json={
                "generated_code": code,
                "language": language,
                "timeout": timeout_seconds,
                "max_output_characters": max_output_characters,
                **({"traceback_verbosity": "Plain"} if language == "ipython" else {}),
            },
            timeout=timeout_seconds + 5.0,
        )
        if response.status_code == 502:
            return {"process_status": "error", "stdout": "", "stderr": "Sandbox 502 error"}
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError(f"sandbox returned a non-object response: {value!r}")
        return value


def sandbox_output_text(result: dict[str, Any]) -> str:
    output = f"{result.get('stdout', '')}{result.get('stderr', '')}"
    return output[:-1] if output.endswith("\n") else output
