from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from apps.controller.models import ProviderCompareRequest
from libs.drafting import DraftingService
from libs.drafting.model_providers import ModelMessage, ModelResponse
from libs.drafting.model_providers.gemini_provider import GeminiProvider
from libs.drafting.model_providers.groq_provider import GroqProvider
from libs.drafting.model_providers.ollama_provider import OllamaProvider
from libs.drafting.model_providers.openrouter_provider import OpenRouterProvider
from libs.drafting.model_providers.router import generate_with_fallback, get_draft_provider
from libs.drafting.privacy_guard import check_external_api_allowed


def test_missing_gemini_key_does_not_crash(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = asyncio.run(GeminiProvider().generate([ModelMessage("user", "hi")]))
    assert response.error == "missing_api_key"
    assert response.provider == "gemini"


def test_gemini_provider_retries_on_rate_limit(monkeypatch) -> None:
    calls: list[int] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return FakeResponse(
                    429,
                    {
                        "error": {
                            "code": 429,
                            "message": "Too many requests",
                            "details": [{"retryDelay": "0.01s"}],
                        }
                    },
                )
            return FakeResponse(
                200,
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "yo"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            )

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr("libs.drafting.model_providers.gemini_provider.httpx.AsyncClient", lambda timeout=None: FakeClient())
    monkeypatch.setattr("libs.drafting.model_providers.gemini_provider.asyncio.sleep", fake_sleep)
    monkeypatch.setenv("GEMINI_API_KEY", "test-token")

    response = asyncio.run(GeminiProvider().generate([ModelMessage("user", "hi")], timeout_seconds=1))

    assert len(calls) == 2
    assert response.error is None
    assert response.text == "yo"


def test_missing_groq_key_does_not_crash(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = asyncio.run(GroqProvider().generate([ModelMessage("user", "hi")]))
    assert response.error == "missing_api_key"
    assert response.provider == "groq"


def test_missing_openrouter_key_does_not_crash(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = asyncio.run(OpenRouterProvider().generate([ModelMessage("user", "hi")]))
    assert response.error == "missing_api_key"
    assert response.provider == "openrouter"


def test_ollama_provider_tries_fallback_models(monkeypatch) -> None:
    monkeypatch.setenv("PHONE_COPILOT_OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("PHONE_COPILOT_AI_REPLY_FALLBACK_MODELS", "qwen3:0.6b,mistral:latest")

    calls: list[str] = []
    options_seen: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers=None):
            calls.append(str(json["model"]))
            options_seen.append(dict(json.get("options", {})))
            if json["model"] == "qwen3:4b":
                raise RuntimeError("timeout")
            return FakeResponse(
                {
                    "message": {
                        "content": '{"candidates":[{"reply":"yo","reason":"fallback","style_score":0.6,"risk_score":0.2}]}'
                    },
                    "done_reason": "stop",
                }
            )

    monkeypatch.setattr("libs.drafting.model_providers.ollama_provider.httpx.AsyncClient", lambda timeout=None: FakeClient())

    response = asyncio.run(
        OllamaProvider().generate([ModelMessage("user", "u coming?")], timeout_seconds=1, response_format="json")
    )

    assert calls == ["qwen3:0.6b"]
    assert all(int(opts["num_predict"]) <= 256 for opts in options_seen)
    assert response.provider == "ollama"
    assert response.model == "qwen3:0.6b"
    assert response.error is None
    assert "yo" in response.text


def test_ollama_cloud_uses_auth_header(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"response": "hi", "done_reason": "stop"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["headers"] = dict(headers)
            captured["json"] = dict(json)
            return FakeResponse()

    monkeypatch.setattr("libs.drafting.model_providers.ollama_provider.httpx.AsyncClient", lambda timeout=None: FakeClient())

    response = asyncio.run(
        OllamaProvider(api_key="test-token").generate([ModelMessage("user", "hi")], model="gpt-oss:120b", timeout_seconds=1)
    )

    assert response.error is None
    assert str(captured["url"]).startswith("https://ollama.com/api/generate")
    assert captured["headers"]["Authorization"] == "Bearer test-token"


def test_cloud_ollama_sensitive_text_stays_off_external_api(monkeypatch) -> None:
    calls: list[str] = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            calls.append("called")
            raise AssertionError("cloud ollama should not be called for sensitive text")

    monkeypatch.setattr("libs.drafting.model_providers.ollama_provider.httpx.AsyncClient", lambda timeout=None: FakeClient())

    settings = SimpleNamespace(
        draft_provider="ollama",
        fast_provider="ollama",
        router_provider="ollama",
        private_provider="ollama",
        external_api_enabled=True,
        external_api_allow_sensitive=False,
        external_api_timeout_seconds=1,
        ollama_api_key="test-token",
        ai_reply_fallback_models="qwen3:0.6b",
        ollama_model="gpt-oss:120b",
    )

    response, metadata = asyncio.run(
        generate_with_fallback(
            settings=settings,
            messages=[ModelMessage("user", "my password is 1234")],
            response_format="text",
            context_texts=["my password is 1234"],
            provider_names=["ollama"],
        )
    )

    assert calls == []
    assert response.provider == "fallback"
    assert metadata["external_api_blocked"] is True


def test_status_never_returns_api_key_values(service, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-value")
    monkeypatch.setenv("GROQ_API_KEY", "secret-groq-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-openrouter-value")

    payload = service.ai_core_status()
    serialized = json.dumps(payload)

    assert payload["providers"]["gemini"]["key_present"] is True
    assert "secret-gemini-value" not in serialized
    assert "secret-groq-value" not in serialized
    assert "secret-openrouter-value" not in serialized


def test_privacy_guard_blocks_sensitive_text() -> None:
    decision = check_external_api_allowed(["my bank card cvv is in the chat"], allow_sensitive=False)
    assert decision.external_api_blocked is True
    assert decision.blocked_reason == "sensitive_content"


def test_privacy_guard_does_not_treat_romantic_pin_as_secret_pin() -> None:
    decision = check_external_api_allowed(["nah id pin ur hips down and kiss ur neck"], allow_sensitive=False)
    assert decision.external_api_blocked is False


def test_privacy_guard_blocks_secret_pin_context() -> None:
    decision = check_external_api_allowed(["my card pin is 1234"], allow_sensitive=False)
    assert decision.external_api_blocked is True
    assert "pin" in (decision.matched_terms or [])


def test_sensitive_text_falls_back_to_private_provider(monkeypatch) -> None:
    calls: list[str] = []

    class FakeProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(self, *args, **kwargs):
            calls.append(self.name)
            return ModelResponse(
                '{"candidates":[{"reply":"what do you mean","reason":"private","style_score":0.5,"risk_score":0.2}]}',
                self.name,
                "fake",
                1,
                None,
                None,
                self.name != "ollama",
            )

    monkeypatch.setattr(
        "libs.drafting.model_providers.router.get_provider",
        lambda name, settings=None: FakeProvider(name),
    )
    settings = SimpleNamespace(
        draft_provider="gemini",
        fast_provider="groq",
        router_provider="openrouter",
        private_provider="ollama",
        external_api_enabled=True,
        external_api_allow_sensitive=False,
        external_api_timeout_seconds=1,
        ollama_model="fake",
    )
    response, metadata = asyncio.run(
        generate_with_fallback(
            settings=settings,
            messages=[ModelMessage("user", "my password is in this chat")],
            response_format="json",
            context_texts=["my password is in this chat"],
        )
    )

    assert calls == ["ollama"]
    assert response.provider == "ollama"
    assert metadata["external_api_blocked"] is True


def test_provider_router_chooses_configured_draft_provider() -> None:
    provider = get_draft_provider(SimpleNamespace(draft_provider="groq"))
    assert provider.name == "groq"


def test_provider_router_falls_back_safely_when_provider_fails(monkeypatch) -> None:
    class FakeProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(self, *args, **kwargs):
            if self.name == "gemini":
                return ModelResponse("", "gemini", "fake", 1, None, "boom", True)
            return ModelResponse(
                '{"candidates":[{"reply":"what time","reason":"ok","style_score":0.7,"risk_score":0.1}]}',
                self.name,
                "fake",
                2,
                None,
                None,
                False,
            )

    monkeypatch.setattr(
        "libs.drafting.model_providers.router.get_provider",
        lambda name, settings=None: FakeProvider(name),
    )
    settings = SimpleNamespace(
        draft_provider="gemini",
        fast_provider="ollama",
        router_provider="ollama",
        private_provider="ollama",
        external_api_enabled=True,
        external_api_allow_sensitive=True,
        external_api_timeout_seconds=1,
        gemini_model="fake",
        ollama_model="fake",
    )

    response, metadata = asyncio.run(
        generate_with_fallback(
            settings=settings,
            messages=[ModelMessage("user", "u coming?")],
            response_format="json",
        )
    )

    assert response.provider == "ollama"
    assert metadata["fallback_errors"][0]["provider"] == "gemini"


def test_provider_compare_endpoint_never_uses_adb(service, mock_adb) -> None:
    service.settings.external_api_enabled = False
    response = service.ai_core_provider_compare(
        ProviderCompareRequest(
            incoming="u coming?",
            context=[],
            relationship_type="close_friend",
            intent_type="planning",
            providers=["gemini"],
        )
    )

    assert response["results"][0]["error"] == "external_api_disabled"
    assert mock_adb.commands == []


def test_candidate_metadata_includes_provider_model_and_scores(tmp_path, monkeypatch) -> None:
    drafting = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        fast_provider="ollama",
        router_provider="ollama",
        private_provider="ollama",
    )

    def fake_generation(**kwargs):
        return (
            ModelResponse(
                '{"candidates":[{"reply":"what time","reason":"asks detail","style_score":0.8,"risk_score":0.1}]}',
                "gemini",
                "gemini-test",
                12,
                "stop",
                None,
                True,
            ),
            {"provider_configured": True, "external_api_blocked": False},
        )

    monkeypatch.setattr(drafting, "_run_provider_generation", fake_generation)
    bundle = drafting.build_bundle(["u coming?"], contact_name="Ali")
    candidate = bundle.reply_candidates[0]

    assert candidate.provider == "gemini"
    assert candidate.model == "gemini-test"
    assert candidate.latency_ms == 12
    assert candidate.assistant_likeness_score >= 0.0
    assert candidate.naturalness_score >= 0.0
    assert candidate.style_score >= 0.0
    assert candidate.risk_score >= 0.0


def test_provider_compare_scores_plain_text_fallback(tmp_path, monkeypatch) -> None:
    drafting = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        fast_provider="ollama",
        router_provider="ollama",
        private_provider="ollama",
    )

    class FakeProvider:
        async def generate(self, *args, **kwargs):
            return ModelResponse("what time\nwhere u lot going", "gemini", "gemini-test", 8, "stop", None, True)

    monkeypatch.setattr("libs.drafting.service.get_provider", lambda name, settings=None: FakeProvider())

    results = drafting.compare_providers(
        incoming="u coming?",
        context=[],
        relationship_type="close_friend",
        intent_type="planning",
        providers=["gemini"],
    )

    assert results[0]["error"] is None
    assert results[0]["reply"] == "what time"


def test_casual_examples_are_short_and_not_assistant_like(tmp_path) -> None:
    drafting = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    bundle = drafting.build_bundle(["u coming?"], contact_name="Ali")
    candidate = bundle.reply_candidates[0]

    assert len(candidate.text.split()) <= 6
    assert candidate.assistant_likeness_score < 0.4
