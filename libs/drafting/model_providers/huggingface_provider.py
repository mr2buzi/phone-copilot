from __future__ import annotations

import os
import time

import httpx

from libs.drafting.model_providers.base import ModelMessage, ModelResponse


class HuggingFaceProvider:
    name = "huggingface"

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
        api_key = (self.api_key or os.getenv("HUGGINGFACE_API_KEY", "")).strip()
        selected_model = model or os.getenv("PHONE_COPILOT_HUGGINGFACE_MODEL", "").strip()
        if not api_key:
            return self._error(selected_model, "missing_api_key")
        if not selected_model:
            return self._error("", "missing_model")
        prompt = "\n\n".join(f"{message.role}: {message.content}" for message in messages if message.content.strip())
        if response_format == "json":
            prompt += "\n\nReturn JSON only."
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"https://api-inference.huggingface.co/models/{selected_model}",
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "temperature": temperature,
                            "max_new_tokens": max_tokens,
                            "return_full_text": False,
                        },
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return self._error(selected_model, _safe_error(exc), latency_ms=_elapsed_ms(started))
        text = ""
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            text = str(payload[0].get("generated_text", "")).strip()
        elif isinstance(payload, dict):
            text = str(payload.get("generated_text", "")).strip()
        return ModelResponse(text, self.name, selected_model, _elapsed_ms(started), None, None if text else "empty_response", True)

    def _error(self, model: str, error: str, latency_ms: int | None = None) -> ModelResponse:
        return ModelResponse("", self.name, model, latency_ms, None, error, True)


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _safe_error(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"http_{status}" if status else exc.__class__.__name__
