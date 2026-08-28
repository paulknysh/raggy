import sys
from types import ModuleType

import pytest

from raggy import llm_factory


def _fake_provider_module(name: str, attrs: dict):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_get_llm_ollama(monkeypatch):
    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        _fake_provider_module("langchain_ollama", {"ChatOllama": FakeChatOllama}),
    )

    llm = llm_factory.get_llm("ollama", "llama3.2", 0.0)

    assert isinstance(llm, FakeChatOllama)
    assert llm.kwargs == {"model": "llama3.2", "temperature": 0.0}


def test_get_llm_openai(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        _fake_provider_module("langchain_openai", {"ChatOpenAI": FakeChatOpenAI}),
    )

    llm = llm_factory.get_llm("openai", "gpt-4o", 0.7)

    assert isinstance(llm, FakeChatOpenAI)
    assert llm.kwargs["model"] == "gpt-4o"
    assert llm.kwargs["temperature"] == 0.7
    assert llm.kwargs["api_key"] is None  # resolves from env at runtime


def test_get_llm_anthropic_sets_max_tokens(monkeypatch):
    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        _fake_provider_module(
            "langchain_anthropic", {"ChatAnthropic": FakeChatAnthropic}
        ),
    )

    llm = llm_factory.get_llm("anthropic", "claude-sonnet", 0.0)

    assert isinstance(llm, FakeChatAnthropic)
    assert llm.kwargs["model"] == "claude-sonnet"
    assert llm.kwargs["api_key"] is None
    assert llm.kwargs["max_tokens"] == llm_factory._DEFAULT_MAX_TOKENS


def test_get_llm_google(monkeypatch):
    class FakeChatGoogle:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        _fake_provider_module(
            "langchain_google_genai", {"ChatGoogleGenerativeAI": FakeChatGoogle}
        ),
    )

    llm = llm_factory.get_llm("google", "gemini-1.5-pro", 0.3)

    assert isinstance(llm, FakeChatGoogle)
    assert llm.kwargs["model"] == "gemini-1.5-pro"
    assert llm.kwargs["temperature"] == 0.3
    assert llm.kwargs["api_key"] is None


def test_get_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        llm_factory.get_llm("grok", "grok-1", 0.0)
