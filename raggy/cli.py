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
from .vectorstore import build_index_config, db_needs_rebuild

logger = logging.getLogger(__name__)

ACCENT = "blue"
STATUS_ACCENT = "grey"

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
    table.add_column("Snippet", style="white")

    for i, doc in enumerate(retrieved_docs, 1):
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 140:
            snippet = snippet[:140] + "..."
        table.add_row(str(i), source_label(doc), snippet)

    console.print(table)


def run_chat() -> None:
    console.print(
        Panel.fit(
            f"[{ACCENT}]{LOGO}[/{ACCENT}]\n\n"
            "You can specify source folders/files and other parameters in "
            "[bold]config.yaml[/bold].\n\n"
            "Type [bold]exit[/bold], [bold]quit[/bold] or [bold]q[/bold] to leave.",
            border_style=ACCENT,
        )
    )
    console.print()

    while True:
        try:
            query = _ask("Question: ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return

        query = query.strip()
        if query.lower() in {"exit", "quit", "q"}:
            console.print()
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return
        if not query:
            continue

        try:
            cfg = load_config()
            index_cfg = build_index_config(
                sources=cfg["sources"],
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                embedding_model=cfg["embedding_model"],
            )
            needs_rebuild = db_needs_rebuild(cfg["persist_directory"], index_cfg)
            if needs_rebuild:
                with console.status(
                    f"[{STATUS_ACCENT}]DB needs (re-)build...[/{STATUS_ACCENT}]"
                ):
                    rebuilt = refresh_db()
            else:
                rebuilt = refresh_db()
            doc_sink: list = []
            with console.status(f"[{STATUS_ACCENT}]Retrieving...[/{STATUS_ACCENT}]"):
                stream = run_pipeline_stream(query, doc_sink=doc_sink)
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
        try:
            with Live(_answer_panel(response_parts[0]), refresh_per_second=12) as live:
                for chunk in stream:
                    response_parts.append(chunk)
                    live.update(_answer_panel("".join(response_parts)))
        except KeyboardInterrupt:
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("raggy").setLevel(logging.WARNING)
    run_chat()
