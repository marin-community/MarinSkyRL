"""Synchronous OpenAI-compatible judge client for verifier worker threads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OpenAIJudge:
    base_url: str
    model: str
    api_key: str = "dummy_key"
    timeout_seconds: float = 600.0

    def generate(self, messages: list[dict[str, str]], *, max_tokens: int = 8192) -> str:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_completion_tokens": max_tokens,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        content = body["choices"][0]["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Judge returned no message content: {body}")
        return content

    def generate_response(
        self,
        input_messages: list[dict[str, str]],
        *,
        metadata: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Call NVIDIA's GenRM Responses-API wrapper with comparison metadata."""
        response = requests.post(
            f"{self.base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "input": input_messages,
                "metadata": metadata,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        for item in reversed(body.get("output", [])):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        raise RuntimeError(f"GenRM judge returned no output text: {body}")
