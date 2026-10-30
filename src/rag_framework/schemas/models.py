"""Data schemas for the Adaptive RAG Evaluation & Optimization Framework.

Provides strongly-typed, validated Pydantic models for documents, chunked text units,
and retrieval ranking results.
"""

from __future__ import annotations

import math
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _check_non_blank_str(value: str, field_name: str) -> str:
    """Validate that a string is neither empty nor purely whitespace.

    Args:
        value: Input string to validate.
        field_name: Name of the field for error reporting.

    Returns:
        The validated string.

    Raises:
        ValueError: If value is empty or consists solely of whitespace.
    """
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"'{field_name}' cannot be empty or whitespace-only.")
    return value


class Document(BaseModel):
    """Represents a raw or ingested source document prior to chunking.

    Attributes:
        document_id: Unique, stable identifier for the document.
        text: Full textual content of the document.
        source: Source identifier (e.g. filename, URI, or corpus reference).
        metadata: Arbitrary metadata dictionary (e.g. authors, creation date).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    document_id: str = Field(
        ...,
        description="Unique, stable identifier for the document.",
    )
    text: str = Field(
        ...,
        description="Full textual content of the document.",
    )
    source: str = Field(
        ...,
        description="Source identifier (e.g. filename, URI, or corpus reference).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata dictionary for document attributes.",
    )

    @field_validator("document_id", "text", "source", mode="after")
    @classmethod
    def validate_non_blank_fields(cls, value: str, info: Any) -> str:
        """Ensure string fields contain non-whitespace text."""
        return _check_non_blank_str(value, info.field_name)


class Chunk(BaseModel):
    """Represents a discrete text segment derived from a parent Document.

    Maintains character offset lineage to the original document for provenance
    and grounded citation tracking.

    Attributes:
        chunk_id: Unique, stable identifier for this chunk.
        document_id: Identifier of the parent Document.
        text: Text content of this specific chunk.
        chunk_index: Zero-based sequential position within the parent Document.
        source: Source identifier of the parent Document.
        start_char: Starting character offset in the parent Document (inclusive).
        end_char: Ending character offset in the parent Document (exclusive).
        metadata: Flexible metadata dictionary for chunk-level attributes.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    chunk_id: str = Field(
        ...,
        description="Unique, stable identifier for this chunk.",
    )
    document_id: str = Field(
        ...,
        description="Identifier of the parent Document.",
    )
    text: str = Field(
        ...,
        description="Text content of the chunk.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based sequential position of the chunk in its document.",
    )
    source: str = Field(
        ...,
        description="Source identifier matching the parent Document.",
    )
    start_char: int = Field(
        ...,
        ge=0,
        description="Starting character offset in the original document (inclusive).",
    )
    end_char: int = Field(
        ...,
        ge=0,
        description="Ending character offset in the original document (exclusive).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata dictionary for chunk attributes.",
    )

    @field_validator("chunk_id", "document_id", "text", "source", mode="after")
    @classmethod
    def validate_non_blank_fields(cls, value: str, info: Any) -> str:
        """Ensure string fields contain non-whitespace text."""
        return _check_non_blank_str(value, info.field_name)

    @model_validator(mode="after")
    def validate_character_offsets(self) -> Self:
        """Validate that end_char is strictly greater than start_char."""
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be strictly greater than "
                f"start_char ({self.start_char})."
            )
        return self

    @property
    def char_length(self) -> int:
        """Return the character span of this chunk in the original document."""
        return self.end_char - self.start_char


class RetrievalResult(BaseModel):
    """Represents a single ranked retrieval candidate returned by a retriever.

    Attributes:
        chunk: The retrieved Chunk instance.
        score: Relevance score assigned by the retrieval algorithm.
        rank: 1-based ordinal position in the retrieval ranking list.
        retriever_name: Identifier of the retriever (e.g. 'bm25', 'dense_faiss').
        metadata: Flexible dictionary for per-result retrieval metadata.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    chunk: Chunk = Field(
        ...,
        description="The retrieved Chunk instance.",
    )
    score: float = Field(
        ...,
        description="Relevance score assigned by the retrieval mechanism.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-based ordinal rank in the retrieved candidate list.",
    )
    retriever_name: str = Field(
        ...,
        description="Name of the retrieval component (e.g. 'bm25', 'dense_faiss').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata dictionary for retrieval details.",
    )

    @field_validator("retriever_name", mode="after")
    @classmethod
    def validate_retriever_name(cls, value: str, info: Any) -> str:
        """Ensure retriever_name is non-empty and not purely whitespace."""
        return _check_non_blank_str(value, info.field_name)

    @field_validator("score", mode="after")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        """Ensure the score is a finite real number (rejects NaN, +Inf, -Inf)."""
        if not math.isfinite(value):
            raise ValueError(f"Retrieval score must be a finite number, got {value}.")
        return value
