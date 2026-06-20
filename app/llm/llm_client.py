from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Base interface for LLM clients.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from a prompt.
        """
        raise NotImplementedError


class FakeLLMClient(LLMClient):
    """
    Simple fake LLM client for tests.
    """

    def __init__(self, response: str = "Fake answer") -> None:
        """
        Initialize the fake client with a fixed response.
        """
        self.response = response

    def generate(self, prompt: str) -> str:
        """
        Return the predefined response for a valid prompt.
        """
        if prompt is None or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        return self.response