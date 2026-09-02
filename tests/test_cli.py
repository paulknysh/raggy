import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from raggy import cli


@pytest.fixture
def captured_console(monkeypatch):
    """Swap the module console for one writing to an in-memory buffer."""
    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    monkeypatch.setattr(cli, "console", console)
    return console


@pytest.fixture
def stub_config(monkeypatch):
    """Make config loading and DB refresh no-ops so tests drive the chat loop."""
    monkeypatch.setattr(cli, "load_config", lambda: {"stub": True})
    monkeypatch.setattr(cli, "refresh_db", lambda cfg: False)


def _output(console) -> str:
    """Console text with wrapping collapsed, so asserts survive line breaks."""
    return " ".join(console.file.getvalue().split())


def _script_inputs(monkeypatch, *queries) -> None:
    """Feed ``queries`` to the chat loop, then leave with /exit."""
    pending = [*queries, "/exit"]
    monkeypatch.setattr(cli, "_ask", lambda prompt: pending.pop(0))


def _doc(name="a.pdf", text="chunk text"):
    return SimpleNamespace(page_content=text, metadata={"source": f"/docs/{name}"})


def test_successful_turn_prints_answer_and_citations(
    monkeypatch, captured_console, stub_config
):
    def fake_stream(query, doc_sink=None, chat_history=None):
        doc_sink.append([_doc()])
        yield "the answer"

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    out = _output(captured_console)
    assert "the answer" in out
    assert "a.pdf" in out


def test_failed_turn_does_not_end_the_session(
    monkeypatch, captured_console, stub_config
):
    asked = []

    def fake_stream(query, doc_sink=None, chat_history=None):
        asked.append(query)
        if query == "boom":
            raise RuntimeError("the provider exploded")
        yield "recovered"

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "boom", "second question")

    cli.run_chat()

    assert asked == ["boom", "second question"]
    out = _output(captured_console)
    assert "the provider exploded" in out
    assert "recovered" in out


def test_failed_turn_keeps_memory_but_records_nothing(
    monkeypatch, captured_console, stub_config
):
    histories = []

    def fake_stream(query, doc_sink=None, chat_history=None):
        histories.append(list(chat_history))
        if query == "boom":
            raise RuntimeError("nope")
        yield f"answer for {query}"

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "first", "boom", "third")

    cli.run_chat()

    earlier_turn = [("human", "first"), ("ai", "answer for first")]
    assert histories[0] == []
    assert histories[1] == earlier_turn
    # The failed turn survived without adding itself to the conversation.
    assert histories[2] == earlier_turn


def test_midstream_failure_keeps_partial_answer_and_citations(
    monkeypatch, captured_console, stub_config
):
    histories = []

    def fake_stream(query, doc_sink=None, chat_history=None):
        histories.append(list(chat_history))
        if query == "half":
            doc_sink.append([_doc()])
            yield "partial text"
            raise RuntimeError("stream died")
        yield "next answer"

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "half", "after")

    cli.run_chat()

    out = _output(captured_console)
    assert "partial text" in out
    assert "stream died" in out
    assert "a.pdf" in out
    assert histories[1] == []


def test_empty_stream_reports_a_clear_message(
    monkeypatch, captured_console, stub_config
):
    def fake_stream(query, doc_sink=None, chat_history=None):
        yield from ()

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert "empty answer" in _output(captured_console)


def test_interrupt_during_retrieval_cancels_only_that_turn(
    monkeypatch, captured_console, stub_config
):
    asked = []

    def fake_stream(query, doc_sink=None, chat_history=None):
        asked.append(query)
        if query == "slow":
            raise KeyboardInterrupt
        yield "answer"

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "slow", "after")

    cli.run_chat()

    assert asked == ["slow", "after"]
    assert "Cancelled" in _output(captured_console)


def test_connection_failure_suggests_starting_ollama(
    monkeypatch, captured_console, stub_config
):
    def fake_stream(query, doc_sink=None, chat_history=None):
        raise RuntimeError("[Errno 61] Connection refused")
        yield  # pragma: no cover  keeps this a generator function

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert "ollama serve" in _output(captured_console)


def test_invalid_config_exits_before_prompting(monkeypatch, captured_console):
    def broken_config():
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def fail_if_asked(prompt):
        raise AssertionError("run_chat prompted despite an unusable config")

    monkeypatch.setattr(cli, "load_config", broken_config)
    monkeypatch.setattr(cli, "_ask", fail_if_asked)

    cli.run_chat()

    assert "chunk_overlap must be smaller" in _output(captured_console)


def test_error_hint_is_none_for_unrecognized_failures():
    assert cli._error_hint(RuntimeError("something unfamiliar")) is None
