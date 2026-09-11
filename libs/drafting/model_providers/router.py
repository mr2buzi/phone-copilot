from __future__ import annotations

import os
from typing import Any

from libs.drafting.model_providers.base import ModelMessage, ModelProvider, ModelResponse
from libs.drafting.model_providers.gemini_provider import GeminiProvider
from libs.drafting.model_providers.groq_provider import GroqProvider
from libs.drafting.model_providers.huggingface_provider import HuggingFaceProvider
from libs.drafting.model_providers.ollama_provider import OllamaProvider
from libs.drafting.model_providers.openrouter_provider import OpenRouterProvider
from libs.drafting.privacy_guard import check_external_api_allowed


EXTERNAL_PROVIDERS = {"gemini", "groq", "openrouter", "huggingface"}


def get_provider(name: str, *, settings: Any | None = None) -> ModelProvider:
    normalized = (name or "ollama").strip().lower()
    if normalized == "gemini":
        return GeminiProvider(_setting(settings, "gemini_api_key", ""))
    if normalized == "groq":
        return GroqProvider(_setting(settings, "groq_api_key", ""))
    if normalized == "openrouter":
        return OpenRouterProvider(_setting(settings, "openrouter_api_key", ""))
    if normalized == "huggingface":
        return HuggingFaceProvider(_setting(settings, "huggingface_api_key", ""))
    if normalized == "ollama":
        fallback_models = _split_models(_setting(settings, "ai_reply_fallback_models", ""))
        return OllamaProvider(
            _setting(settings, "ollama_base_url", None) or _setting(settings, "ai_reply_base_url", None),
            api_key=_setting(settings, "ollama_api_key", ""),
            fallback_models=fallback_models,
        )
    raise ValueError(f"unknown provider: {name}")


def get_draft_provider(settings: Any) -> ModelProvider:
    return get_provider(_setting(settings, "draft_provider", "ollama"), settings=settings)


def get_fast_provider(settings: Any) -> ModelProvider:
    return get_provider(_setting(settings, "fast_provider", "ollama"), settings=settings)


def get_router_provider(settings: Any) -> ModelProvider:
    return get_provider(_setting(settings, "router_provider", "ollama"), settings=settings)


def get_private_provider(settings: Any) -> ModelProvider:
    return get_provider(_setting(settings, "private_provider", "ollama"), settings=settings)


async def generate_with_fallback(
    *,
    settings: Any,
    messages: list[ModelMessage],
    provider_names: list[str] | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 180,
    response_format: str = "text",
    timeout_seconds: float | None = None,
    context_texts: list[str] | None = None,
) -> tuple[ModelResponse, dict[str, object]]:
    external_enabled = _as_bool(_setting(settings, "external_api_enabled", True))
    allow_sensitive = _as_bool(_setting(settings, "external_api_allow_sensitive", False))
    timeout = float(timeout_seconds or _setting(settings, "external_api_timeout_seconds", 25) or 25)
    requested = provider_names or [
        _setting(settings, "draft_provider", "gemini"),
        _setting(settings, "fast_provider", "groq"),
        _setting(settings, "router_provider", "openrouter"),
        _setting(settings, "private_provider", "ollama"),
    ]
    order = _dedupe_provider_names(requested)
    if not external_enabled:
        order = [name for name in order if name == "ollama"] or ["ollama"]
    privacy_texts = context_texts or [message.content for message in messages]
    privacy = check_external_api_allowed(privacy_texts, allow_sensitive=allow_sensitive)
    if privacy.external_api_blocked:
        order = [name for name in order if name not in EXTERNAL_PROVIDERS] or ["ollama"]
    errors: list[dict[str, str]] = []
    for provider_name in order:
        provider = get_provider(provider_name, settings=settings)
        provider_is_external = provider_name in EXTERNAL_PROVIDERS or (provider_name == "ollama" and getattr(provider, "is_cloud", False))
        if not external_enabled and provider_is_external:
            errors.append({"provider": provider_name, "model": str(model or _model_for_provider(settings, provider_name) or ""), "error": "external_api_disabled"})
            continue
        if privacy.external_api_blocked and provider_is_external:
            errors.append({"provider": provider_name, "model": str(model or _model_for_provider(settings, provider_name) or ""), "error": "sensitive_content"})
            continue
        provider_model = model or _model_for_provider(settings, provider_name)
        response = await provider.generate(
            messages,
            model=provider_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout_seconds=timeout,
        )
        if response.error:
            errors.append({"provider": response.provider, "model": response.model, "error": response.error})
            continue
        metadata = {
            "fallback_errors": errors,
            "external_api_blocked": privacy.external_api_blocked,
            "blocked_reason": privacy.blocked_reason,
            "provider_configured": _provider_is_configured(response.provider, settings=settings),
        }
        return response, metadata
    fallback = _deterministic_fallback(messages, model=_model_for_provider(settings, "fallback"))
    return fallback, {
        "fallback_errors": errors,
        "external_api_blocked": privacy.external_api_blocked,
        "blocked_reason": privacy.blocked_reason,
        "provider_configured": False,
        "manual_review_fallback": True,
    }


def provider_config_status(settings: Any) -> dict[str, dict[str, object]]:
    ollama_api_key = str(_setting(settings, "ollama_api_key", "") or os.getenv("OLLAMA_API_KEY", "")).strip()
    ollama_base_url = _setting(settings, "ollama_base_url", None) or _setting(settings, "ai_reply_base_url", None)
    if not ollama_base_url:
        ollama_base_url = "https://ollama.com/api/chat" if ollama_api_key else "http://127.0.0.1:11434/api/chat"
    gemini_key_present = _secret_present(settings, "gemini_api_key", "GEMINI_API_KEY")
    groq_key_present = _secret_present(settings, "groq_api_key", "GROQ_API_KEY")
    openrouter_key_present = _secret_present(settings, "openrouter_api_key", "OPENROUTER_API_KEY")
    huggingface_key_present = _secret_present(settings, "huggingface_api_key", "HUGGINGFACE_API_KEY")
    return {
        "gemini": {
            "configured": gemini_key_present,
            "model": _setting(settings, "gemini_model", os.getenv("PHONE_COPILOT_GEMINI_MODEL", "gemini-2.5-flash")),
            "key_present": gemini_key_present,
        },
        "groq": {
            "configured": groq_key_present,
            "model": _setting(settings, "groq_model", os.getenv("PHONE_COPILOT_GROQ_MODEL", "llama-3.1-8b-instant")),
            "key_present": groq_key_present,
        },
        "openrouter": {
            "configured": openrouter_key_present,
            "model": _setting(settings, "openrouter_model", os.getenv("PHONE_COPILOT_OPENROUTER_MODEL", "google/gemini-2.0-flash-001")),
            "key_present": openrouter_key_present,
        },
        "huggingface": {
            "configured": huggingface_key_present and bool(_setting(settings, "huggingface_model", os.getenv("PHONE_COPILOT_HUGGINGFACE_MODEL", "")).strip()),
            "model": _setting(settings, "huggingface_model", os.getenv("PHONE_COPILOT_HUGGINGFACE_MODEL", "")),
            "key_present": huggingface_key_present,
        },
        "ollama": {
            "configured": True,
            "model": _setting(settings, "ollama_model", os.getenv("PHONE_COPILOT_OLLAMA_MODEL", "qwen3:4b")),
            "base_url": ollama_base_url,
            "key_present": bool(ollama_api_key),
        },
    }


def _deterministic_fallback(messages: list[ModelMessage], *, model: str = "deterministic") -> ModelResponse:
    latest = next((message.content for message in reversed(messages) if message.role == "user"), "")
    text = latest.casefold()
    if "?" in text or any(term in text for term in ("coming", "come", "time", "where", "when")):
        replies = ["what time", "depends what time", "where u lot going"]
    elif any(term in text for term in ("hey", "hi", "hii", "yo")):
        replies = ["yo", "heyy", "u good"]
    else:
        replies = ["yeah maybe", "not sure yet", "what do you mean"]
    payload = {
        "candidates": [
            {"reply": reply, "reason": "deterministic fallback", "style_score": 0.5, "risk_score": 0.2}
            for reply in replies
        ]
    }
    import json

    return ModelResponse(json.dumps(payload), "fallback", model, None, "fallback", None, False)


def _dedupe_provider_names(names: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = str(name or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    if "ollama" not in seen:
        result.append("ollama")
    return result


def _model_for_provider(settings: Any, provider: str) -> str | None:
    normalized = provider.strip().lower()
    return _setting(settings, f"{normalized}_model", None)


def _provider_is_configured(provider: str, *, settings: Any | None = None) -> bool:
    normalized = provider.strip().lower()
    if normalized == "ollama" or normalized == "fallback":
        return True
    attr_names = {
        "gemini": "gemini_api_key",
        "groq": "groq_api_key",
        "openrouter": "openrouter_api_key",
        "huggingface": "huggingface_api_key",
    }
    env_names = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
    }
    env_name = env_names.get(normalized)
    attr_name = attr_names.get(normalized)
    if not env_name or not attr_name:
        return False
    return _secret_present(settings, attr_name, env_name)


def _secret_present(settings: Any, attr_name: str, env_name: str) -> bool:
    value = _setting(settings, attr_name, "")
    return bool(str(value or os.getenv(env_name, "")).strip())


def _setting(settings: Any, name: str, default: Any) -> Any:
    if settings is None:
        return os.getenv(f"PHONE_COPILOT_{name.upper()}", default)
    return getattr(settings, name, os.getenv(f"PHONE_COPILOT_{name.upper()}", default))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _split_models(value: Any) -> list[str]:
    return [model.strip() for model in str(value or "").split(",") if model.strip()]
