import httpx
from urllib import request
from fastapi.testclient import TestClient
import pytest

from apps.demo.main import create_demo_app


@pytest.fixture()
def demo(monkeypatch):
    attempts = []
    def no_network(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("The recording demo must not contact a model provider")
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    monkeypatch.setattr(request, "urlopen", no_network)
    monkeypatch.setattr("libs.drafting.service.generate_with_fallback", no_network)
    from libs.adb.client import ADBClient
    monkeypatch.setattr(ADBClient, "__init__", no_network)
    with TestClient(create_demo_app()) as client:
        yield client
    assert not attempts


@pytest.mark.parametrize("scenario", ["plans", "work", "unknown"])
def test_demo_uses_core_drafts_and_approval_never_delivers(demo, scenario):
    response = demo.post('/api/draft', json={'scenario': scenario})
    assert response.status_code == 200
    payload = response.json()
    assert payload['generation'] == 'deterministic'
    assert payload['delivery_enabled'] is False
    assert payload['bundle']['reply_candidates']
    approval = demo.post('/api/approve', json={'scenario': scenario, 'candidate_index': 0})
    assert approval.status_code == 200
    assert approval.json()['sent'] is False
    assert demo.post('/api/approve', json={'scenario': scenario, 'candidate_index': 0}).status_code == 409
    assert demo.post('/api/send').status_code == 403


def test_demo_rejects_unknown_scenario_and_approval_without_draft(demo):
    assert demo.post('/api/draft', json={'scenario': '../data'}).status_code == 404
    assert demo.post('/api/approve', json={'scenario': 'plans', 'candidate_index': 0}).status_code == 409
    assert demo.post('/api/approve', json={'scenario': 'plans', 'candidate_index': -1}).status_code == 422


def test_demo_isolated_from_live_configuration(demo):
    service = demo.app.state.drafting
    assert service.ai_reply_enabled is False
    assert service.external_api_enabled is False
    assert service.retrieval_config.backend == 'lexical'
    assert service.training_messages_dir.is_absolute()
    assert not service.gemini_api_key and not service.groq_api_key
    assert demo.get('/api/scenarios').json()['device_connected'] is False


def test_demo_does_not_read_live_files_or_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / '.env').write_text('GEMINI_API_KEY=private-test-value\n', encoding='utf-8')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'style_rubric.json').write_text('not valid JSON', encoding='utf-8')
    (data / 'contact_profiles.json').write_text('not valid JSON', encoding='utf-8')
    monkeypatch.setenv('GEMINI_API_KEY', 'private-test-value')
    with TestClient(create_demo_app()) as client:
        service = client.app.state.drafting
        assert not service.gemini_api_key
        assert service.style_rubric_path.parent == service.training_messages_dir.parent
        assert service.data_root != data
        result = client.post('/api/draft', json={'scenario': 'plans'})
        assert result.status_code == 200
        assert 'private-test-value' not in result.text
