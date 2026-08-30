import subprocess
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


def test_ensure_ollama_model_already_present(monkeypatch):
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        assert kwargs.pop("capture_output") is True
        assert kwargs.pop("text") is True
        assert kwargs.pop("check", False) is False

        class Result:
            stdout = (
                "NAME\tID\tSIZE\tMODIFIED\n"
                "llama3.2:latest\ta80c4f17acd5\t2.0 GB\t2 days ago\n"
                "nomic-embed-text:latest\t0a109f422b47\t274 MB\t2 days ago\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    assert llm_factory.ensure_ollama_model("llama3.2") is False


def test_ensure_ollama_model_pulls_missing(monkeypatch):
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))

        class Result:
            stdout = "NAME\tID\tSIZE\tMODIFIED\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    assert llm_factory.ensure_ollama_model("llama3.2") is True
    assert calls[-1][0] == ["/usr/local/bin/ollama", "pull", "llama3.2"]
    assert calls[-1][1]["check"] is True


def test_ensure_ollama_model_no_cli_raises(monkeypatch):
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ollama CLI not found"):
        llm_factory.ensure_ollama_model("llama3.2")


def test_ensure_ollama_model_already_present_with_explicit_latest_tag(monkeypatch):
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        class Result:
            stdout = (
                "NAME\tID\tSIZE\tMODIFIED\n"
                "llama3.2:latest\ta80c4f17acd5\t2.0 GB\t2 days ago\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)
    assert llm_factory.ensure_ollama_model("llama3.2:latest") is False


def test_ensure_ollama_model_missing_tag_triggers_pull(monkeypatch):
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))

        class Result:
            stdout = (
                "NAME\tID\tSIZE\tMODIFIED\n"
                "llama3.2:latest\ta80c4f17acd5\t2.0 GB\t2 days ago\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    assert llm_factory.ensure_ollama_model("llama3.2:q4_K_M") is True
    assert calls[-1][0] == ["/usr/local/bin/ollama", "pull", "llama3.2:q4_K_M"]


def test_ensure_ollama_model_pull_failure_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        if cmd[1] == "list":

            class ListResult:
                stdout = "NAME\tID\tSIZE\tMODIFIED\n"
                stderr = ""

            return ListResult()

        exc = subprocess.CalledProcessError(1, cmd)
        exc.stderr = "error: pull model manifest: file does not exist"
        raise exc

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Failed to pull Ollama model 'ghost'"):
        llm_factory.ensure_ollama_model("ghost")
