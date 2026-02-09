from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


class LLMProvider(ABC):
    """Bridge implementor: concrete providers implement only text generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests.

    You can provide either a fixed response or a response factory based on prompt.
    """

    fixed_response: Optional[str] = None
    response_factory: Optional[Callable[[str], str]] = None

    def generate(self, prompt: str) -> str:
        if self.response_factory is not None:
            return self.response_factory(prompt)
        if self.fixed_response is not None:
            return self.fixed_response
        raise ValueError("MockLLMProvider requires fixed_response or response_factory")


class LocalLLMProvider(LLMProvider):
    """Local provider stub.

    This intentionally keeps runtime dependencies lazy; actual wiring will be done
    during CR01 GUI integration.
    """

    def __init__(
        self,
        model_name: str,
        host: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        response_format: Optional[str] = None,
    ):
        self.model_name = model_name
        self.host = host
        self.options = options
        self.response_format = response_format

    def generate(self, prompt: str) -> str:
        try:
            import ollama  # lazy import
        except Exception as e:
            raise RuntimeError(
                "ollama is not available; cannot use LocalLLMProvider"
            ) from e

        try:
            if self.host:
                client = ollama.Client(host=self.host)
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": self.options,
                }
                if self.response_format:
                    kwargs["format"] = self.response_format
                response = client.generate(**kwargs)
            else:
                kwargs2: dict[str, Any] = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": self.options,
                }
                if self.response_format:
                    kwargs2["format"] = self.response_format
                response = ollama.generate(**kwargs2)
            return response.get("response", "")
        except Exception as e:
            host_hint = f" ({self.host})" if self.host else ""
            raise RuntimeError(
                "Failed to connect to Ollama server" + host_hint + ". "
                "Ensure Ollama is installed and running (default: http://localhost:11434)."
            ) from e


class ApiLLMProvider(LLMProvider):
    """API provider stub.

    Note: concrete API specifics (OpenAI/Claude/Gemini) will be added later.
    For now, it can call a generic HTTP endpoint that returns plain text.
    """

    def __init__(self, base_url: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        try:
            import httpx  # lazy import
        except Exception as e:
            raise RuntimeError(
                "httpx is not available; cannot use ApiLLMProvider"
            ) from e

        # Generic endpoint contract (to be refined): POST /generate {prompt}
        url = f"{self.base_url}/generate"
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json={"prompt": prompt})
            resp.raise_for_status()
            data = resp.json() if "application/json" in resp.headers.get("content-type", "") else None
            if isinstance(data, dict) and "response" in data:
                return str(data["response"])
            return resp.text
