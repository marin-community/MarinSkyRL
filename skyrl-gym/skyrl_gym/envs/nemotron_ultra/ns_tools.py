# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful Python-tool loop used by NVIDIA's ns_tools agent."""

from __future__ import annotations

import json
from typing import Any

from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient, sandbox_output_text


def execute_python_calls(
    assistant_message: dict[str, Any],
    *,
    sandbox: SandboxClient,
    session_id: str,
    timeout_seconds: float = 10.0,
) -> list[dict[str, Any]] | None:
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None
    observations: list[dict[str, Any]] = []
    for call in calls:
        function = call.get("function") or {}
        call_id = call.get("id")
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (json.JSONDecodeError, TypeError) as error:
            output = json.dumps({"error": f"Invalid tool call arguments: {error!r}"})
        else:
            try:
                if function.get("name") != "stateful_python_code_exec":
                    raise ValueError(f"Unknown tool: {function.get('name')}")
                if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
                    raise ValueError("stateful_python_code_exec requires a string code argument")
                result = sandbox.execute(
                    arguments["code"],
                    language="ipython",
                    timeout_seconds=timeout_seconds,
                    session_id=session_id,
                )
                output = sandbox_output_text(result)
            except Exception as error:
                output = json.dumps({"error": f"{type(error).__name__}: {error}"})
        observations.append({"role": "tool", "tool_call_id": str(call_id), "content": output})
    return observations
