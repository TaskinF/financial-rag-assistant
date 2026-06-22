import pytest

from app.llm.ollama_client import OllamaLLMClient

def test_ollama_client_can_be_created_with_valid_constructor():
    client = OllamaLLMClient(
        base_url="http://localhost:11434",
        model="gemma3:4b",
        timeout=120,
    )

    assert client.base_url == "http://localhost:11434"
    assert client.model == "gemma3:4b"
    assert client.timeout == 120


def test_ollama_client_strips_trailing_slash_from_base_url():
    client = OllamaLLMClient(base_url="http://localhost:11434/")

    assert client.base_url == "http://localhost:11434"


def test_ollama_client_generate_raises_value_error_for_empty_prompt():
    client = OllamaLLMClient()

    with pytest.raises(ValueError):
        client.generate("")


def test_ollama_client_generate_returns_mocked_response(monkeypatch):
    captured = {}

    class MockResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": "Mock Ollama answer"}

    def mock_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return MockResponse()

    monkeypatch.setattr("app.llm.ollama_client.requests.post", mock_post)

    client = OllamaLLMClient(
        base_url="http://localhost:11434/",
        model="gemma3:4b",
        timeout=120,
    )

    result = client.generate("test prompt")

    assert result == "Mock Ollama answer"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "gemma3:4b"
    assert captured["json"]["prompt"] == "test prompt"
    assert captured["json"]["stream"] is False
    assert captured["timeout"] == 120


def test_ollama_client_generate_raises_value_error_when_response_field_is_missing(monkeypatch):
    class MockResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"unexpected": "value"}

    def mock_post(url, json, timeout):
        return MockResponse()

    monkeypatch.setattr("app.llm.ollama_client.requests.post", mock_post)

    client = OllamaLLMClient()

    with pytest.raises(ValueError):
        client.generate("test prompt")