from __future__ import annotations

import asyncio
import os
import re
import time

import httpx

from libs.drafting.model_providers.base import ModelMessage, ModelResponse


class GeminiProvider:
    name = "gemini"
    default_model = "gemini-2.5-flash"

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
        api_key = (self.api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        selected_model = model or os.getenv("PHONE_COPILOT_GEMINI_MODEL", self.default_model).strip() or self.default_model
        if not api_key:
            return self._error(selected_model, "missing_api_key")
        prompt = _messages_to_prompt(messages)
        if response_format == "json":
            prompt += "\n\nReturn JSON only. Do not wrap it in markdown."
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if response_format == "json":
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reply": {"type": "string"},
                                "reason": {"type": "string"},
                                "style_score": {"type": "number"},
                                "risk_score": {"type": "number"},
                            },
                            "required": ["reply", "reason", "style_score", "risk_score"],
                        },
                    },
                },
                "required": ["candidates"],
            }
        started = time.perf_counter()
        attempts = 2
        last_error = "empty_response"
        last_finish_reason: str | None = None
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.post(
                        url,
                        json=body,
                        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    )
                    payload = _safe_json(response)
            except Exception as exc:
                return self._error(selected_model, _safe_error(exc), latency_ms=_elapsed_ms(started))

            if response.status_code in {429, 503}:
                last_error = _error_from_payload(payload, response.status_code)
                retry_delay = min(timeout_seconds, _retry_delay_seconds(payload) or (0.75 * (attempt + 1)))
                if attempt + 1 < attempts and retry_delay > 0:
                    await asyncio.sleep(retry_delay)
                    continue
                return self._error(selected_model, last_error, latency_ms=_elapsed_ms(started))

            if response.status_code >= 400:
                return self._error(
                    selected_model,
                    _error_from_payload(payload, response.status_code),
                    latency_ms=_elapsed_ms(started),
                )

            if isinstance(payload, dict) and "error" in payload:
                error_payload = payload.get("error")
                if isinstance(error_payload, dict):
                    message = str(error_payload.get("message", "")).strip()
                    code = error_payload.get("code")
                    error = f"http_{code}" if code else "api_error"
                    return self._error(selected_model, error if not message else f"{error}:{message[:120]}", latency_ms=_elapsed_ms(started))
            candidates = payload.get("candidates") if isinstance(payload, dict) else []
            first = candidates[0] if isinstance(candidates, list) and candidates else {}
            content = first.get("content", {}) if isinstance(first, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
            last_finish_reason = str(first.get("finishReason", "")) if isinstance(first, dict) else None
            return ModelResponse(
                text=text,
                provider=self.name,
                model=selected_model,
                latency_ms=_elapsed_ms(started),
                raw_finish_reason=last_finish_reason,
                error=None if text else "empty_response",
                external_api_used=True,
            )
        return self._error(selected_model, last_error, latency_ms=_elapsed_ms(started))

    def _error(self, model: str, error: str, latency_ms: int | None = None) -> ModelResponse:
        return ModelResponse("", self.name, model, latency_ms, None, error, True)


def _messages_to_prompt(messages: list[ModelMessage]) -> str:
    return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages if message.content.strip())


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _safe_error(exc: Exception) -> str:
    name = exc.__class__.__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"http_{status}" if status else name


def _safe_json(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_from_payload(payload: dict[str, object], status_code: int) -> str:
    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        message = str(error_payload.get("message", "")).strip()
        code = error_payload.get("code")
        error = f"http_{code}" if code else f"http_{status_code}"
        return error if not message else f"{error}:{message[:120]}"
    return f"http_{status_code}"


def _retry_delay_seconds(payload: dict[str, object]) -> float:
    error_payload = payload.get("error")
    if not isinstance(error_payload, dict):
        return 0.0
    details = error_payload.get("details")
    if not isinstance(details, list):
        return 0.0
    for item in details:
        if not isinstance(item, dict):
            continue
        retry_delay = str(item.get("retryDelay", "")).strip()
        if not retry_delay:
            continue
        match = re.fullmatch(r"(?:(\d+))?(?:\.(\d+))?s", retry_delay)
        if not match:
            continue
        seconds = int(match.group(1) or 0)
        fraction = match.group(2) or ""
        if fraction:
            seconds += float(f"0.{fraction}")
        return max(0.0, seconds)
    return 0.0
