"""Interactive chat CLI for raggy, built on `rich` for a minimalist look."""

import argparse
import logging
import textwrap
from pathlib import Path

try:
    import readline  # noqa: F401  enables arrow-key editing/history in input()
except ImportError:  # pragma: no cover - readline is unavailable on Windows
    pass

from rich.box import SIMPLE_HEAD, SQUARE
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from .loaders import source_label
from .pipeline import SCORE_KEY
from .raggy import ensure_models, load_config, refresh_db, run_pipeline_stream

logger = logging.getLogger(__name__)

ACCENT = "blue"

# Answers and citations are prose, and prose stops being readable somewhere
# around 100 columns: on a wide terminal an unbounded panel runs 25+ words to
# the line. Both blocks are measured against this so they also line up with
# each other.
MAX_WIDTH = 100

# A citation is a pointer, not the passage: enough characters to recognize the
# chunk the answer came from, while keeping every row to a line or two.
MAX_SNIPPET_CHARS = 140

MEMORY_TURNS = 8

LOGO = textwrap.dedent(
    r"""
    █▀█ █▀█ █▀▀ █▀▀ █ █
    █▀▄ █▀█ █ █ █ █ ▀█▀
    ▀ ▀ ▀ ▀ ▀▀▀ ▀▀▀  ▀
    """
).strip("\n")

console = Console()


def _ask(prompt: str) -> str:
    """Prompt for input, using a plain (uncolored) readline prompt.

    rich's ``Prompt.ask``/``console.input`` and ANSI-colored prompts are
    unreliable here: on macOS the ``readline`` module is backed by Apple's
    libedit, which mangles any ANSI escapes in the prompt (cursor jumps while
    typing) and wipes the label when backspacing. A plain-text prompt passed
    to ``input()`` is measured correctly by both libedit and GNU readline, so
    backspacing and arrow keys behave as expected.
    """
    plain = Text.from_markup(prompt).plain
    return input(plain)


class _ProgressLine:
    """Single-line, in-place status for the slow steps of a turn.

    Pulling models and indexing documents are what keep a user waiting, so
    both report what they are working on instead of hanging on a bare spinner.
    A pull knows how many bytes it has left and gets a download bar; the
    indexing steps can only name the file or batch and get a spinner and a
    message. Only one of the two displays is ever live, and neither is started
    until there is something to report: sessions with nothing to do print
    nothing.
    """

    def __init__(self) -> None:
        self._status = None
        self._bar = None
        self._task = None

    def __enter__(self):
        return self

    def __call__(
        self,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if total is None:
            self._close_bar()
            self._show_message(message)
        else:
            self._close_status()
            self._show_bar(message, completed or 0, total)

    def _show_message(self, message: str) -> None:
        text = Text(message, style=ACCENT)
        if self._status is None:
            self._status = console.status(text)
            self._status.start()
        else:
            self._status.update(text)

    def _show_bar(self, message: str, completed: int, total: int) -> None:
        if self._bar is None:
            self._bar = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}", style=ACCENT),
                BarColumn(complete_style=ACCENT, finished_style=ACCENT),
                DownloadColumn(),
                console=console,
                transient=True,
            )
            self._bar.start()
            self._task = self._bar.add_task(message, total=total, completed=completed)
        else:
            self._bar.update(
                self._task, description=message, total=total, completed=completed
            )

    def _close_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.stop()
            self._bar = None
            self._task = None

    def __exit__(self, *exc_info) -> None:
        self._close_status()
        self._close_bar()


def _body_width() -> int:
    """Width shared by the answer and its citations, honoring narrow terminals."""
    return min(console.width, MAX_WIDTH)


def _answer_panel(text: str) -> Panel:
    return Panel(
        Markdown(text),
        title=f"[{ACCENT}]Answer[/{ACCENT}]",
        border_style=ACCENT,
        box=SQUARE,
        padding=(1, 2),
        width=_body_width(),
    )


def print_citations(retrieved_docs) -> None:
    if not retrieved_docs:
        return

    has_scores = any(SCORE_KEY in doc.metadata for doc in retrieved_docs)

    table = Table(
        title=f"[{ACCENT}]Citations[/{ACCENT}]",
        # Explicit: rich's default table title style is italic.
        title_style=ACCENT,
        border_style=ACCENT,
        header_style=ACCENT,
        box=SIMPLE_HEAD,
        pad_edge=False,
        width=_body_width(),
    )
    table.add_column("#", justify="right", style=ACCENT)  # width=3
    table.add_column("Source", style=ACCENT)
    if has_scores:
        table.add_column("Score", justify="right", style=ACCENT)
    table.add_column("Snippet")

    for i, doc in enumerate(retrieved_docs, 1):
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
        score = doc.metadata.get(SCORE_KEY)
        score_cell = f"{score:.3f}" if score is not None else ""
        if has_scores:
            table.add_row(str(i), source_label(doc), score_cell, snippet)
        else:
            table.add_row(str(i), source_label(doc), snippet)

    console.print(table)


_OLLAMA_HINT = (
    "Could not reach the model service. Embeddings always run locally, so make "
    "sure Ollama is up: `ollama serve`."
)

_API_KEY_HINT = (
    "Set the provider's API key (OPENAI_API_KEY / ANTHROPIC_API_KEY / "
    "GEMINI_API_KEY), then restart raggy — the process reads it at startup."
)

# Matched (lowercased) against the exception message. The underlying libraries
# report these as bare transport/validation errors, which say nothing about
# what the user should do next.
_ERROR_HINTS = (
    ("connection refused", _OLLAMA_HINT),
    ("failed to connect", _OLLAMA_HINT),
    ("api_key", _API_KEY_HINT),
    ("api key", _API_KEY_HINT),
)


def _error_hint(exc: Exception) -> str | None:
    """Return an actionable next step for the failure shapes we can recognize."""
    text = str(exc).lower()
    for needle, hint in _ERROR_HINTS:
        if needle in text:
            return hint
    return None


def _report_error(exc: Exception) -> None:
    """Print a failed turn's cause, with a hint when we recognize it.

    The traceback goes to the logger at debug level instead of the console:
    printing a full traceback above the message buries the one line the user
    can act on. Raise the ``raggy`` logger to DEBUG to get it back.
    """
    logger.debug("Turn failed", exc_info=exc)
    console.print(f"[red]Error:[/red] {exc or exc.__class__.__name__}")
    hint = _error_hint(exc)
    if hint:
        console.print(f"[{ACCENT}]{hint}[/{ACCENT}]")


def _run_turn(
    query: str,
    chat_history: list[tuple[str, str]],
    config_path: str = "config.yaml",
) -> None:
    """Answer one query, printing the answer and its citations. Never raises.

    Every failure is contained in the turn that caused it. Most causes are
    transient or fixable while the session is open — Ollama not yet running, a
    source file being rewritten mid-index, a provider timeout — so ending the
    chat (and discarding the conversation memory built up so far) would throw
    away more than the failed question.

    ``chat_history`` is only extended when a turn produces a complete answer:
    a partial answer is left on screen for the user to read but kept out of the
    context of later turns.
    """
    doc_sink: list = []
    try:
        cfg = load_config(config_path)
        with _ProgressLine() as progress:
            # Pulled here rather than left to the pipeline: a first run
            # downloads gigabytes, and the download belongs on the status line
            # instead of inside the retrieval spinner.
            ensure_models(cfg, progress=progress)
            rebuilt = refresh_db(
                cfg,
                progress=progress,
                on_stale=lambda: console.print(
                    f"[{ACCENT}]DB needs updating...[/{ACCENT}]"
                ),
                config_path=config_path,
            )
        with console.status(f"[{ACCENT}]Retrieving...[/{ACCENT}]"):
            stream = run_pipeline_stream(
                query,
                doc_sink=doc_sink,
                chat_history=chat_history,
                config_path=config_path,
            )
            first_chunk = next(stream)
    except StopIteration:
        console.print("[red]Error:[/red] the model returned an empty answer.")
        return
    except KeyboardInterrupt:
        console.print(f"\n[{ACCENT}]Cancelled.[/{ACCENT}]")
        return
    except Exception as e:  # noqa: BLE001  reported, then the session continues
        _report_error(e)
        return

    if rebuilt:
        console.print(f"[{ACCENT}]DB updated.[/{ACCENT}]")

    response_parts = [first_chunk]
    complete = True
    try:
        with Live(
            _answer_panel(response_parts[0]), console=console, refresh_per_second=12
        ) as live:
            for chunk in stream:
                response_parts.append(chunk)
                live.update(_answer_panel("".join(response_parts)))
    except KeyboardInterrupt:
        complete = False
        console.print(
            f"\n[{ACCENT}]Answer interrupted; showing what was generated.[/{ACCENT}]"
        )
    except Exception as e:  # noqa: BLE001  the partial answer stays on screen
        complete = False
        _report_error(e)

    # Printed even for a failed generation: retrieval populates the sink before
    # the LLM runs, so the citations show what the answer was cut short on.
    console.print()
    print_citations(doc_sink[-1] if doc_sink else [])
    console.print()

    if complete:
        chat_history.append(("human", query))
        chat_history.append(("ai", "".join(response_parts)))
        del chat_history[: -MEMORY_TURNS * 2]


def run_chat(config_path: str = "config.yaml") -> None:
    console.print(
        Panel.fit(
            f"[{ACCENT}]{LOGO}[/{ACCENT}]\n\n"
            "using config located @ "
            f"[{ACCENT}]{Path(config_path).resolve()}[/{ACCENT}]\n\n"
            f"[{ACCENT}]/clear[/{ACCENT}] : reset conversation memory\n"
            f"[{ACCENT}]/exit[/{ACCENT}]  : leave",
            border_style=ACCENT,
            box=SQUARE,
        )
    )
    console.print()

    # The only unrecoverable failure: without a valid config there is nothing to
    # answer from, so fail here rather than on every question the user types.
    try:
        load_config(config_path)
    except Exception as e:  # noqa: BLE001
        _report_error(e)
        return

    chat_history: list[tuple[str, str]] = []

    while True:
        try:
            query = _ask(">>> ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return

        query = query.strip()
        if query in {"/exit", "exit"}:
            console.print()
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return
        if query in {"/clear", "clear"}:
            chat_history.clear()
            console.print(f"[{ACCENT}]Conversation memory cleared.[/{ACCENT}]")
            continue
        if not query:
            continue

        _run_turn(query, chat_history, config_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="raggy",
        description="Interactive chat CLI for raggy.",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        help="path to the config file (e.g. config.yaml)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("raggy").setLevel(logging.WARNING)
    # bm25s raises its own logger to DEBUG at import time, so the root level
    # set above does not gate it: without this, every BM25 index build prints
    # "DEBUG: Building index from IDs objects" into the chat.
    logging.getLogger("bm25s").setLevel(logging.WARNING)
    run_chat(args.config)
