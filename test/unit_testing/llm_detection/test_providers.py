import builtins
import types

import pytest

from llm_detection.providers import ApiLLMProvider, LocalLLMProvider, MockLLMProvider


def test_mock_provider_fixed_response():
    p = MockLLMProvider(fixed_response="ok")
    assert p.generate("prompt") == "ok"


def test_mock_provider_factory_takes_precedence():
    p = MockLLMProvider(
        fixed_response="ignored",
        response_factory=lambda prompt: f"resp:{prompt}",
    )
    assert p.generate("x") == "resp:x"


def test_mock_provider_requires_response_configuration():
    p = MockLLMProvider()
    with pytest.raises(ValueError):
        p.generate("x")


def test_local_provider_raises_if_ollama_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ollama":
            raise ImportError("no")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    p = LocalLLMProvider(model_name="m")
    with pytest.raises(RuntimeError, match="ollama is not available"):
        p.generate("prompt")


def test_local_provider_wraps_connection_errors(monkeypatch):
    # Provide a stub 'ollama' module that raises on generate.
    stub = types.SimpleNamespace(generate=lambda **kwargs: (_ for _ in ()).throw(Exception("boom")))
    monkeypatch.setitem(__import__("sys").modules, "ollama", stub)

    p = LocalLLMProvider(model_name="m")
    with pytest.raises(RuntimeError, match="Failed to connect to Ollama server"):
        p.generate("prompt")


def test_local_provider_host_path_uses_client_and_host_hint_on_failure(monkeypatch):
    class StubClient:
        def __init__(self, host: str):
            self.host = host

        def generate(self, **kwargs):
            raise Exception("boom")

    stub = types.SimpleNamespace(
        Client=lambda host: StubClient(host=host),
        generate=lambda **kwargs: {"response": "should-not-be-used"},
    )
    monkeypatch.setitem(__import__("sys").modules, "ollama", stub)

    p = LocalLLMProvider(model_name="m", host="http://my-ollama")
    with pytest.raises(RuntimeError, match=r"Failed to connect to Ollama server \(http://my-ollama\)"):
        p.generate("prompt")


def test_local_provider_adds_format_param_when_response_format_set(monkeypatch):
    captured = {}

    def stub_generate(**kwargs):
        captured.update(kwargs)
        return {"response": "ok"}

    stub = types.SimpleNamespace(generate=stub_generate)
    monkeypatch.setitem(__import__("sys").modules, "ollama", stub)

    p = LocalLLMProvider(model_name="m", response_format="json")
    assert p.generate("prompt") == "ok"
    assert captured["format"] == "json"



def test_api_provider_raises_if_httpx_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("no")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    p = ApiLLMProvider(base_url="http://example")
    with pytest.raises(RuntimeError, match="httpx is not available"):
        p.generate("prompt")


def test_api_provider_returns_json_response_field(monkeypatch):
    class StubResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "hello"}

        @property
        def text(self):
            return "fallback"

    class StubClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            assert url == "http://example/generate"
            assert json == {"prompt": "p"}
            return StubResp()

    stub_httpx = types.SimpleNamespace(Client=lambda timeout: StubClient(timeout=timeout))
    monkeypatch.setitem(__import__("sys").modules, "httpx", stub_httpx)

    p = ApiLLMProvider(base_url="http://example")
    assert p.generate("p") == "hello"


def test_api_provider_json_without_response_falls_back_to_text(monkeypatch):
    class StubResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"not_response": "x"}

        @property
        def text(self):
            return "plain-text"

    class StubClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            return StubResp()

    stub_httpx = types.SimpleNamespace(Client=lambda timeout: StubClient(timeout=timeout))
    monkeypatch.setitem(__import__("sys").modules, "httpx", stub_httpx)

    p = ApiLLMProvider(base_url="http://example")
    assert p.generate("p") == "plain-text"


def test_api_provider_raise_for_status_error_propagates(monkeypatch):
    class StubResp:
        def __init__(self):
            self.headers = {"content-type": "text/plain"}

        def raise_for_status(self):
            raise Exception("bad status")

        @property
        def text(self):
            return "unused"

    class StubClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            return StubResp()

    stub_httpx = types.SimpleNamespace(Client=lambda timeout: StubClient(timeout=timeout))
    monkeypatch.setitem(__import__("sys").modules, "httpx", stub_httpx)

    p = ApiLLMProvider(base_url="http://example")
    with pytest.raises(Exception, match="bad status"):
        p.generate("p")


def test_api_provider_falls_back_to_text_if_not_json(monkeypatch):
    class StubResp:
        def __init__(self):
            self.headers = {"content-type": "text/plain"}

        def raise_for_status(self):
            return None

        @property
        def text(self):
            return "plain"

    class StubClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            return StubResp()

    stub_httpx = types.SimpleNamespace(Client=lambda timeout: StubClient(timeout=timeout))
    monkeypatch.setitem(__import__("sys").modules, "httpx", stub_httpx)

    p = ApiLLMProvider(base_url="http://example/")
    assert p.generate("p") == "plain"
