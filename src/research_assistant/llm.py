from __future__ import annotations

import json
import os
from typing import Any

import httpx


DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class LLMError(RuntimeError):
    """Raised when an LLM request or response is unusable."""


class OpenAIClient:
    def __init__(self, api_key: str, model: str, base_url: str = DEFAULT_OPENAI_BASE_URL) -> None:
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @classmethod
    def from_env(cls) -> OpenAIClient:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("live mode requires OPENAI_API_KEY")
        return cls(
            api_key,
            os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        )

    def close(self) -> None:
        self.client.close()

    def complete_json(self, system: str, prompt: str, *, max_tokens: int, timeout: float) -> tuple[dict[str, Any], int]:
        content, tokens = self._complete(system, prompt, max_tokens=max_tokens, timeout=timeout, json_mode=True)
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError("model returned malformed JSON") from exc
        if not isinstance(value, dict):
            raise LLMError("model JSON response must be an object")
        return value, tokens

    def complete_text(self, system: str, prompt: str, *, max_tokens: int, timeout: float) -> tuple[str, int]:
        return self._complete(system, prompt, max_tokens=max_tokens, timeout=timeout, json_mode=False)

    def _complete(self, system: str, prompt: str, *, max_tokens: int, timeout: float, json_mode: bool) -> tuple[str, int]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max(1, max_tokens),
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = self.client.post("/chat/completions", json=body, timeout=max(0.001, timeout))
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise KeyError("empty content")
            usage = payload.get("usage") or {}
            tokens = int(usage.get("total_tokens") or usage.get("completion_tokens") or 0)
            return content.strip(), max(0, tokens)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LLMError("OpenAI-compatible provider request failed") from exc
