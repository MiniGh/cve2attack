"""LLM client wrapper for local generation API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict
from urllib import error, request


@dataclass
class LLMClientConfig:
    """Config for local LLM generation endpoint."""

    base_url: str = "http://172.23.216.73:11434/api/generate"
    model: str = "qwen3:32b"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.5


class LLMClient:
    """Simple retry-enabled client for /api/generate style endpoints."""

    def __init__(self, config: LLMClientConfig) -> None:
        """Initialize client with endpoint settings."""
        self.config = config

    def _build_payload(self, prompt: str) -> bytes:
        """Build request payload for generation endpoint."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
        }
        return json.dumps(payload).encode("utf-8")

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API with retries and return parsed response object."""
        last_exception: Exception | None = None
        body = self._build_payload(prompt)

        for attempt in range(1, self.config.max_retries + 1):
            req = request.Request(
                self.config.base_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_exception = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds * attempt)

        raise RuntimeError(f"LLM request failed after retries: {last_exception}")
