"""Small configurable client for an Ollama-compatible generate endpoint."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib import error, request


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_seconds: int = 120
    max_retries: int = 3

    def generate(self, *, system: str, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model, "system": system, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = request.Request(
                    self.base_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    value = json.loads(response.read().decode("utf-8"))
                return str(value.get("response") or "").strip()
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(float(2**attempt))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")
