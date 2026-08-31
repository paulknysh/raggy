"""Pydantic schema for the runtime config (config.yaml).

``load_config`` (in :mod:`raggy.raggy`) reads the YAML file and validates it
against :class:`RaggySettings`, which enforces types and value constraints at
load time so invalid configuration fails fast with a clear message instead of
surfacing deep in the pipeline.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class RaggySettings(BaseModel):
    """Validated view of the runtime configuration.

    Field types mirror the coercion the library already performed by hand
    (``int()`` / ``float()`` / ``str()``) so numeric strings in the YAML are
    accepted, but ranges and enum membership are enforced.
    """

    model_config = ConfigDict(extra="ignore")

    sources: list[str]
    persist_directory: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    embedding_model: str
    llm_provider: Literal["ollama", "openai", "anthropic", "google"] = "ollama"
    llm_model: str
    temperature: float
    retrieve_k: int
    mmr_fetch_k: int | None = None
    search_type: Literal["mmr", "similarity"]
    relevance_filter: bool = False
    rerank_enabled: bool = False
    rerank_model: str
    rerank_k: int | None = None
    system_prompt: str

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("sources")
    @classmethod
    def _sources_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("'sources' must contain at least one file or directory")
        return value

    @field_validator("chunk_size", "batch_size", "retrieve_k")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("mmr_fetch_k")
    @classmethod
    def _mmr_fetch_k_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be a non-negative integer")
        return value

    @field_validator("temperature")
    @classmethod
    def _temperature_range(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("must be between 0.0 and 2.0")
        return value

    @field_validator("rerank_k")
    @classmethod
    def _positive_rerank_k(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "RaggySettings":
        if self.search_type == "mmr":
            if self.mmr_fetch_k is None:
                raise ValueError("mmr_fetch_k is required when search_type is 'mmr'")
            if self.mmr_fetch_k < self.retrieve_k:
                raise ValueError(
                    f"mmr_fetch_k ({self.mmr_fetch_k}) must not be less than "
                    f"retrieve_k ({self.retrieve_k})"
                )
        if self.rerank_k is not None and self.rerank_k > self.retrieve_k:
            raise ValueError(
                f"rerank_k ({self.rerank_k}) must not be greater than "
                f"retrieve_k ({self.retrieve_k})"
            )
        return self
