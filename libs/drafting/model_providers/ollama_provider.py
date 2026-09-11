from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import httpx

from libs.drafting.model_providers.base import ModelMessage, ModelResponse


class OllamaProvider:
    name = "ollama"
    default_model = "qwen3:4b"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, fallback_models: list[str] | None = None) -> None:
        self.api_key = (api_key or os.getenv("OLLAMA_API_KEY", "") or os.getenv("PHONE_COPILOT_OLLAMA_API_KEY", "")).strip()
        self.base_url = base_url or _ollama_base_url_from_env(api_key_present=bool(self.api_key))
        self.fallback_models = [model.strip() for model in (fallback_models or []) if model.strip()]
        self.is_cloud = _is_cloud_ollama_base_url(self.base_url)

    async def generate(
        self,
        messages: list[ModelMessage],
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 180,
        response_format: str = "text",
        timeout_seconds: float = 25,
    ) -> ModelResponse:
        selected_model = model or os.getenv("PHONE_COPILOT_OLLAMA_MODEL", self.default_model).strip() or self.default_model
        model_candidates = _ollama_model_candidates(
            selected_model,
            response_format=response_format,
            fallback_models=self.fallback_models,
        )
        started = time.perf_counter()
        last_error = "empty_response"
        last_finish_reason: str | None = None
        max_predict = min(int(max_tokens), _ollama_max_predict_tokens())
        prompt = _ollama_prompt_from_messages(messages, response_format=response_format)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for index, candidate_model in enumerate(model_candidates):
            candidate_timeout = _ollama_candidate_timeout(
                candidate_model=candidate_model,
                selected_model=selected_model,
                base_timeout=timeout_seconds,
                is_primary=index == 0,
            )
            for url, body in (
                (
                    _ollama_generate_url(self.base_url),
                    {
                        "model": candidate_model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": os.getenv("PHONE_COPILOT_OLLAMA_KEEP_ALIVE", "10m") or "10m",
                        "options": {"temperature": temperature, "num_predict": max_predict},
                    },
                ),
                (
                    self.base_url,
                    {
                        "model": candidate_model,
                        "messages": [{"role": message.role, "content": message.content} for message in messages],
                        "stream": False,
                        "think": False,
                        "keep_alive": os.getenv("PHONE_COPILOT_OLLAMA_KEEP_ALIVE", "10m") or "10m",
                        "options": {"temperature": temperature, "num_predict": max_predict},
                    },
                ),
            ):
                try:
                    async with httpx.AsyncClient(timeout=candidate_timeout) as client:
                        response = await client.post(url, json=body, headers=headers)
                        response.raise_for_status()
                        payload = response.json()
                except Exception as exc:
                    last_error = _safe_error(exc)
                    continue
                text = _ollama_extract_text(payload)
                last_finish_reason = str(payload.get("done_reason", "")) if isinstance(payload, dict) else None
                if text:
                    return ModelResponse(
                        text=text,
                        provider=self.name,
                        model=candidate_model,
                        latency_ms=_elapsed_ms(started),
                        raw_finish_reason=last_finish_reason,
                        error=None,
                        external_api_used=False,
                    )
                last_error = "empty_response"
        return ModelResponse(
            "",
            self.name,
            selected_model,
            _elapsed_ms(started),
            last_finish_reason,
            last_error,
            False,
        )


def _ollama_base_url_from_env(*, api_key_present: bool = False) -> str:
    explicit = os.getenv("PHONE_COPILOT_OLLAMA_BASE_URL", "").strip()
    if explicit:
        return explicit
    if api_key_present or bool(os.getenv("OLLAMA_API_KEY", "").strip()):
        return "https://ollama.com/api/chat"
    ai_url = os.getenv("PHONE_COPILOT_AI_REPLY_BASE_URL", "").strip()
    return ai_url or "http://127.0.0.1:11434/api/chat"


def _ollama_generate_url(base_url: str) -> str:
    if base_url.rstrip("/").endswith("/api/chat"):
        return f"{base_url.rstrip('/')[:-len('/api/chat')]}/api/generate"
    if base_url.rstrip("/").endswith("/api/generate"):
        return base_url
    return base_url.rstrip("/") + "/api/generate"


def _ollama_prompt_from_messages(messages: list[ModelMessage], *, response_format: str) -> str:
    prompt = "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages if message.content.strip())
    return prompt


def _ollama_extract_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    response_text = str(payload.get("response", "")).strip()
    if response_text:
        return response_text
    message = payload.get("message", {})
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return ""


def _ollama_model_candidates(
    selected_model: str,
    *,
    response_format: str,
    fallback_models: list[str] | None = None,
) -> list[str]:
    fallback_values = list(fallback_models or _ollama_fallback_models_from_env())
    candidates = []
    if response_format == "json":
        candidates.extend(fallback_values)
        if not candidates:
            candidates.append(selected_model)
    else:
        candidates = [selected_model]
        candidates.extend(fallback_values)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [selected_model]


def _ollama_max_predict_tokens() -> int:
    raw = os.getenv("PHONE_COPILOT_OLLAMA_MAX_TOKENS", "96").strip()
    try:
        return max(32, min(512, int(raw)))
    except ValueError:
        return 96


def _ollama_candidate_timeout(*, candidate_model: str, selected_model: str, base_timeout: float, is_primary: bool) -> float:
    if is_primary and candidate_model == selected_model:
        raw = os.getenv("PHONE_COPILOT_OLLAMA_PRIMARY_TIMEOUT_SECONDS", "8").strip()
    else:
        raw = os.getenv("PHONE_COPILOT_OLLAMA_FALLBACK_TIMEOUT_SECONDS", "20").strip()
    try:
        candidate_timeout = float(raw)
    except ValueError:
        candidate_timeout = 8.0 if is_primary else 20.0
    return max(1.0, min(float(base_timeout), candidate_timeout))


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _safe_error(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"http_{status}" if status else exc.__class__.__name__


def _ollama_fallback_models_from_env() -> list[str]:
    raw_values = [
        os.getenv("PHONE_COPILOT_AI_REPLY_FALLBACK_MODELS", ""),
        os.getenv("PHONE_COPILOT_OLLAMA_FALLBACK_MODELS", ""),
    ]
    models: list[str] = []
    for value in raw_values:
        models.extend(model.strip() for model in value.split(",") if model.strip())
    return models


def _is_cloud_ollama_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    return parsed.scheme == "https" and host.endswith("ollama.com")
