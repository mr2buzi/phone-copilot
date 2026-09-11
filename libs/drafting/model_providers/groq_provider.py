from __future__ import annotations

import os
import time

import httpx

from libs.drafting.model_providers.base import ModelMessage, ModelResponse


class GroqProvider:
    name = "groq"
    default_model = "llama-3.1-8b-instant"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or ""

    async def generate(
        self,
        messages: list[ModelMessage],
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 180,
        response_format: str = "text",
        timeout_seconds: float = 25,
    ) -> ModelResponse:
        api_key = (self.api_key or os.getenv("GROQ_API_KEY", "")).strip()
        selected_model = model or os.getenv("PHONE_COPILOT_GROQ_MODEL", self.default_model).strip() or self.default_model
        if not api_key:
            return self._error(selected_model, "missing_api_key")
        body: dict[str, object] = {
            "model": selected_model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    self.endpoint,
                    json=body,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return self._error(selected_model, _safe_error(exc), latency_ms=_elapsed_ms(started))
        choice = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = str(message.get("content", "")).strip() if isinstance(message, dict) else ""
        return ModelResponse(
            text=text,
            provider=self.name,
            model=selected_model,
            latency_ms=_elapsed_ms(started),
            raw_finish_reason=str(choice.get("finish_reason", "")) if isinstance(choice, dict) else None,
            error=None if text else "empty_response",
            external_api_used=True,
        )

    def _error(self, model: str, error: str, latency_ms: int | None = None) -> ModelResponse:
        return ModelResponse("", self.name, model, latency_ms, None, error, True)


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _safe_error(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"http_{status}" if status else exc.__class__.__name__
