"""The progress-reporting contract shared by the slow steps of a run.

Pulling models, reading source files, and embedding chunks are the three
things that keep a user waiting. Each reports through the same callback so a
front end can render them all on one status line; the callback receives an
already-formatted message plus, when the work is measured in bytes, the
counters needed to draw a progress bar.
"""

from typing import Protocol


class ProgressCallback(Protocol):
    """Receives one status line per unit of slow work.

    ``completed``/``total`` are only passed for work with a known size (a
    model pull); steps that can only name what they are doing pass the
    message alone.
    """

    def __call__(
        self,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None: ...
