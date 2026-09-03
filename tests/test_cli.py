import io
import logging
import sys
from types import SimpleNamespace
from typing import ClassVar

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
    """Make config loading, model pulls, and DB refresh no-ops for the chat loop."""
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(cli, "ensure_models", lambda cfg, progress=None: None)
    monkeypatch.setattr(cli, "refresh_db", lambda cfg, **kwargs: False)


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
    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
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

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
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

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
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

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
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
    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        yield from ()

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert "empty answer" in _output(captured_console)


def test_interrupt_during_retrieval_cancels_only_that_turn(
    monkeypatch, captured_console, stub_config
):
    asked = []

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
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
    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        raise RuntimeError("[Errno 61] Connection refused")
        yield  # pragma: no cover  keeps this a generator function

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert "ollama serve" in _output(captured_console)


def test_invalid_config_exits_before_prompting(monkeypatch, captured_console):
    def broken_config(*args, **kwargs):
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def fail_if_asked(prompt):
        raise AssertionError("run_chat prompted despite an unusable config")

    monkeypatch.setattr(cli, "load_config", broken_config)
    monkeypatch.setattr(cli, "_ask", fail_if_asked)

    cli.run_chat()

    assert "chunk_overlap must be smaller" in _output(captured_console)


def test_error_hint_is_none_for_unrecognized_failures():
    assert cli._error_hint(RuntimeError("something unfamiliar")) is None


@pytest.fixture
def status_lines(monkeypatch, captured_console):
    """Record what the CLI puts on its in-place status line.

    rich renders a live status only on a real terminal, so the display is
    stubbed out and its messages captured instead.
    """
    messages: list[str] = []

    class FakeStatus:
        def start(self):
            pass

        def update(self, renderable):
            messages.append(str(getattr(renderable, "plain", renderable)))

        def stop(self):
            messages.append("<stopped>")

    def fake_status(renderable, **kwargs):
        messages.append(str(getattr(renderable, "plain", renderable)))
        return FakeStatus()

    monkeypatch.setattr(captured_console, "status", fake_status)
    return messages


def test_indexing_progress_is_reported_on_one_line(monkeypatch, status_lines):
    """Indexing reports into a single status line, closed before retrieval."""
    forwarded = []

    def fake_refresh_db(cfg, progress=None, on_stale=None, **kwargs):
        forwarded.append(progress)
        progress("[1/2] ingesting a.pdf ...")
        progress("[2/2] ingesting b.pdf ...")
        return True

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        yield "the answer"

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(cli, "ensure_models", lambda cfg, progress=None: None)
    monkeypatch.setattr(cli, "refresh_db", fake_refresh_db)
    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert callable(forwarded[0])
    assert status_lines[:3] == [
        "[1/2] ingesting a.pdf ...",
        "[2/2] ingesting b.pdf ...",
        "<stopped>",
    ]
    assert "Retrieving" in status_lines[3]


def test_no_progress_line_when_nothing_is_indexed(monkeypatch, status_lines):
    """A session whose DB is already current never starts the status display."""

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        yield "the answer"

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(cli, "ensure_models", lambda cfg, progress=None: None)
    monkeypatch.setattr(cli, "refresh_db", lambda cfg, **kwargs: False)
    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert not any("ingesting" in line for line in status_lines)


class _FakeProgress:
    """Records what the CLI draws on its download bar."""

    instances: ClassVar[list["_FakeProgress"]] = []

    def __init__(self, *columns, **kwargs):
        self.tasks = []
        self.updates = []
        self.started = False
        self.stopped = False
        _FakeProgress.instances.append(self)

    def start(self):
        self.started = True

    def add_task(self, description, total=None, completed=0):
        self.tasks.append((description, completed, total))
        return len(self.tasks) - 1

    def update(self, task, description=None, total=None, completed=None):
        self.updates.append((task, description, completed, total))

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_bar(monkeypatch):
    """Swap the download bar for a recorder (rich draws nothing off-terminal)."""
    _FakeProgress.instances = []
    monkeypatch.setattr(cli, "Progress", _FakeProgress)
    return _FakeProgress.instances


def test_model_pull_renders_a_download_bar(monkeypatch, status_lines, fake_bar):
    """Byte-counted work gets a bar; the bar closes when text-only work starts."""

    def fake_ensure_models(cfg, progress=None):
        progress("pulling nomic-embed-text ... (pulling manifest)")
        progress("pulling nomic-embed-text ...", 100, 274)
        progress("pulling nomic-embed-text ...", 274, 274)

    def fake_refresh_db(cfg, progress=None, on_stale=None, **kwargs):
        progress("[1/1] ingesting a.pdf ...")
        return True

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        yield "the answer"

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(cli, "ensure_models", fake_ensure_models)
    monkeypatch.setattr(cli, "refresh_db", fake_refresh_db)
    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    assert len(fake_bar) == 1
    bar = fake_bar[0]
    assert bar.started and bar.stopped
    assert bar.tasks == [("pulling nomic-embed-text ...", 100, 274)]
    assert bar.updates == [(0, "pulling nomic-embed-text ...", 274, 274)]

    # The manifest step (no byte counts) and the ingest line share the status,
    # and the status is closed while the bar is up rather than drawn over it.
    assert status_lines[:2] == [
        "pulling nomic-embed-text ... (pulling manifest)",
        "<stopped>",
    ]
    assert status_lines[2] == "[1/1] ingesting a.pdf ..."


def test_stale_db_is_announced_before_and_after_the_update(
    monkeypatch, captured_console
):
    """The wait is announced when it starts, not only when it ends."""

    def fake_refresh_db(cfg, progress=None, on_stale=None, **kwargs):
        on_stale()
        return True

    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        yield "the answer"

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(cli, "ensure_models", lambda cfg, progress=None: None)
    monkeypatch.setattr(cli, "refresh_db", fake_refresh_db)
    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    out = _output(captured_console)
    assert "DB needs updating..." in out
    assert out.index("DB needs updating...") < out.index("DB updated.")


def test_current_db_announces_nothing(monkeypatch, captured_console, stub_config):
    def fake_stream(query, doc_sink=None, chat_history=None, **kwargs):
        yield "the answer"

    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat()

    out = _output(captured_console)
    assert "DB needs updating" not in out
    assert "DB updated" not in out


def test_main_silences_the_bm25s_debug_logger(monkeypatch):
    """bm25s sets its own logger to DEBUG at import; the CLI must undo that."""
    logging.getLogger("bm25s").setLevel(logging.DEBUG)
    monkeypatch.setattr(sys, "argv", ["raggy"])
    monkeypatch.setattr(cli, "run_chat", lambda *args: None)

    cli.main()

    assert logging.getLogger("bm25s").level == logging.WARNING


def test_config_flag_defaults_to_working_directory_config():
    assert cli._parse_args([]).config == "config.yaml"


def test_main_passes_the_config_flag_to_the_chat(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["raggy", "--config", "/tmp/other.yaml"])
    monkeypatch.setattr(cli, "run_chat", lambda config_path: seen.append(config_path))

    cli.main()

    assert seen == ["/tmp/other.yaml"]


def test_turn_runs_against_the_given_config(monkeypatch, captured_console):
    """Every config reader in a turn is pointed at the CLI's --config path."""
    seen = {}

    def fake_stream(query, doc_sink=None, chat_history=None, config_path=None):
        seen["stream"] = config_path
        yield "the answer"

    monkeypatch.setattr(
        cli, "load_config", lambda path="config.yaml": seen.setdefault("load", path)
    )
    monkeypatch.setattr(cli, "ensure_models", lambda cfg, progress=None: None)
    monkeypatch.setattr(
        cli,
        "refresh_db",
        lambda cfg, config_path=None, **kwargs: seen.setdefault("refresh", config_path),
    )
    monkeypatch.setattr(cli, "run_pipeline_stream", fake_stream)
    _script_inputs(monkeypatch, "a question")

    cli.run_chat("/tmp/other.yaml")

    assert seen["load"] == "/tmp/other.yaml"
    assert seen["refresh"] == "/tmp/other.yaml"
    assert seen["stream"] == "/tmp/other.yaml"


def test_banner_names_the_config_in_use(monkeypatch, captured_console, stub_config):
    monkeypatch.setattr(cli, "_ask", lambda prompt: "/exit")

    cli.run_chat("/tmp/other.yaml")

    assert "/tmp/other.yaml" in _output(captured_console)
