import pytest

from app.llm.llm_client import FakeLLMClient, LLMClient


def test_fake_llm_client_returns_configured_response():
    client = FakeLLMClient(response="Test response")

    result = client.generate("What is the fund strategy?")

    assert result == "Test response"


def test_fake_llm_client_raises_value_error_for_empty_prompt():
    client = FakeLLMClient()

    with pytest.raises(ValueError):
        client.generate("")


def test_llm_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMClient()