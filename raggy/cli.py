"""Interactive chat CLI for raggy, built on `rich` for a minimalist look."""

import logging
import readline  # noqa: F401  enables arrow-key editing/history in input()
import textwrap

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .raggy import load_config, refresh_db, run_pipeline_stream, source_label

logger = logging.getLogger(__name__)

ACCENT = "blue"
STATUS_ACCENT = "blue"

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
    from rich.text import Text

    plain = Text.from_markup(prompt).plain
    return input(plain)


def _answer_panel(text: str) -> Panel:
    return Panel(
        Markdown(text),
        title=f"[bold {ACCENT}]Answer[/bold {ACCENT}]",
        border_style=ACCENT,
        padding=(1, 2),
    )


def print_citations(retrieved_docs) -> None:
    if not retrieved_docs:
        return

    has_scores = any("relevance_score" in doc.metadata for doc in retrieved_docs)

    table = Table(
        title=f"[bold {ACCENT}]Citations[/bold {ACCENT}]",
        title_style="bold",
        border_style=ACCENT,
        header_style=ACCENT,
        box=SIMPLE_HEAD,
        pad_edge=False,
    )
    table.add_column("#", justify="right", style=ACCENT)  # width=3
    table.add_column("Source", style=ACCENT)
    if has_scores:
        table.add_column("Score", justify="right", style=ACCENT)
    table.add_column("Snippet", style="white")

    for i, doc in enumerate(retrieved_docs, 1):
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 140:
            snippet = snippet[:140] + "..."
        score = doc.metadata.get("relevance_score")
        score_cell = f"{score:.3f}" if score is not None else ""
        if has_scores:
            table.add_row(str(i), source_label(doc), score_cell, snippet)
        else:
            table.add_row(str(i), source_label(doc), snippet)

    console.print(table)


def run_chat() -> None:
    console.print(
        Panel.fit(
            f"[{ACCENT}]{LOGO}[/{ACCENT}]\n\n"
            "Specify source folders/files and other parameters in "
            "[bold]config.yaml[/bold]\n\n"
            "[bold]/clear[/bold] to reset the conversation memory\n"
            "[bold]/exit[/bold] to leave",
            border_style=ACCENT,
        )
    )
    console.print()

    chat_history: list[tuple[str, str]] = []

    while True:
        try:
            query = _ask(">>> ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return

        query = query.strip()
        if query == "/exit":
            console.print()
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return
        if query == "/clear":
            chat_history.clear()
            console.print(f"[{ACCENT}]Conversation memory cleared.[/{ACCENT}]")
            continue
        if not query:
            continue

        try:
            cfg = load_config()
            rebuilt = refresh_db(cfg)
            doc_sink: list = []
            with console.status(f"[{STATUS_ACCENT}]Retrieving...[/{STATUS_ACCENT}]"):
                stream = run_pipeline_stream(
                    query, doc_sink=doc_sink, chat_history=chat_history
                )
                first_chunk = next(stream)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
        except Exception as e:
            logger.exception("An unexpected error occurred")
            console.print(f"[red]Error:[/red] {e}")
            return

        if rebuilt:
            console.print(f"[{ACCENT}]DB re-indexed.[/{ACCENT}]")

        response_parts = [first_chunk]
        interrupted = False
        try:
            with Live(_answer_panel(response_parts[0]), refresh_per_second=12) as live:
                for chunk in stream:
                    response_parts.append(chunk)
                    live.update(_answer_panel("".join(response_parts)))
        except KeyboardInterrupt:
            interrupted = True
            console.print(
                f"\n[{ACCENT}]Answer interrupted; showing what was generated.[/{ACCENT}]"
            )
        except Exception as e:
            logger.exception("An unexpected error occurred")
            console.print(f"[red]Error:[/red] {e}")
            return

        retrieved_docs = doc_sink[-1] if doc_sink else []

        console.print()
        print_citations(retrieved_docs)
        console.print()

        if not interrupted:
            chat_history.append(("human", query))
            chat_history.append(("ai", "".join(response_parts)))
            del chat_history[: -MEMORY_TURNS * 2]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("raggy").setLevel(logging.WARNING)
    run_chat()
