import requests

from app.llm.llm_client import LLMClient


class OllamaLLMClient(LLMClient):
    """
    LLM client implementation for a local Ollama server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: int = 120,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL of the Ollama server.
            model: Ollama model name.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Generate a response from Ollama for the given prompt.
        """
        if prompt is None or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        generated_text = data.get("response")

        if generated_text is None:
            raise ValueError("response field not found in Ollama response")

        return str(generated_text)