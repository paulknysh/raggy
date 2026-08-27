"""Interactive chat CLI for raggy, built on `rich` for a minimalist look."""

import logging
import readline  # noqa: F401  enables arrow-key editing/history in input()

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .raggy import refresh_db, run_pipeline, source_label

logger = logging.getLogger(__name__)

ACCENT = "blue"

console = Console()


def print_answer(response: str) -> None:
    panel = Panel(
        Text(response),
        title=f"[bold {ACCENT}]Answer[/bold {ACCENT}]",
        border_style=ACCENT,
        padding=(1, 2),
    )
    console.print(panel)


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
            f"[bold {ACCENT}]raggy[/bold {ACCENT}] - chat with your documents!\n\n"
            "You can specify source folders/files and tune RAG parameters in "
            "[bold]config.yaml[/bold].",
            border_style=ACCENT,
        )
    )
    console.print(
        f"[{ACCENT}]Ask a question, or type [bold]exit[/bold] / press Ctrl+C to quit.\n"
    )

    while True:
        try:
            query = Prompt.ask(f"[bold {ACCENT}]Question[/bold {ACCENT}]")
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n[{ACCENT}]See you![/{ACCENT}]")
            return

        query = query.strip()
        if query.lower() in {"exit", "quit", "q"}:
            console.print(f"[{ACCENT}]See you![/{ACCENT}]")
            return
        if not query:
            continue

        try:
            rebuilt = refresh_db()
            with console.status(f"[{ACCENT}]Retrieving...[/{ACCENT}]"):
                response, retrieved_docs = run_pipeline(query)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
        except Exception as e:
            logger.exception("An unexpected error occurred")
            console.print(f"[red]Error:[/red] {e}")
            return

        if rebuilt:
            console.print(
                f"[{ACCENT}]Source documents changed - DB re-indexed.[/{ACCENT}]"
            )
        console.print()
        print_answer(response)
        console.print()
        print_citations(retrieved_docs)
        console.print()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("raggy").setLevel(logging.INFO)
    run_chat()
