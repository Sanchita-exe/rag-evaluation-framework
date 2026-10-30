"""Unit tests for foundational data schemas.

Tests Document, Chunk, and RetrievalResult models for:
- Valid instantiation and field access
- Rejection of invalid values (empty strings, negative offsets/indices, non-positive ranks)
- Independence of default metadata dictionaries (no shared mutable state)
- Round-trip serialization and deserialization (dict and JSON)
- Assignment validation
"""

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from rag_framework.schemas import Chunk, Document, RetrievalResult


# ============================================================================
# Document Tests
# ============================================================================


def test_document_valid_minimal() -> None:
    """A Document should instantiate successfully with minimal required fields."""
    doc = Document(
        document_id="doc_001",
        text="Dense and sparse retrieval algorithms serve complementary roles.",
        source="retrieval_overview.txt",
    )
    assert doc.document_id == "doc_001"
    assert doc.text == "Dense and sparse retrieval algorithms serve complementary roles."
    assert doc.source == "retrieval_overview.txt"
    assert doc.metadata == {}


def test_document_valid_with_metadata() -> None:
    """A Document should retain custom metadata provided at instantiation."""
    meta: dict[str, Any] = {"author": "Alice", "year": 2026, "tags": ["ir", "rag"]}
    doc = Document(
        document_id="doc_002",
        text="Information retrieval metrics include MRR and nDCG.",
        source="metrics.pdf",
        metadata=meta,
    )
    assert doc.metadata == meta
    assert doc.metadata["author"] == "Alice"


@pytest.mark.parametrize(
    ("doc_id", "text", "source"),
    [
        ("", "Valid text", "source.txt"),
        ("   ", "Valid text", "source.txt"),
        ("doc_001", "", "source.txt"),
        ("doc_001", "   \t\n  ", "source.txt"),
        ("doc_001", "Valid text", ""),
        ("doc_001", "Valid text", "   "),
    ],
)
def test_document_rejects_empty_or_whitespace_strings(
    doc_id: str, text: str, source: str
) -> None:
    """Document fields must not be empty or purely whitespace."""
    with pytest.raises(ValidationError):
        Document(document_id=doc_id, text=text, source=source)


def test_document_rejects_extra_fields() -> None:
    """Document must reject undeclared fields due to extra='forbid'."""
    with pytest.raises(ValidationError):
        Document(  # type: ignore[call-arg]
            document_id="doc_001",
            text="Valid text",
            source="source.txt",
            unknown_field="unexpected",
        )


def test_document_metadata_default_independence() -> None:
    """Default metadata dictionaries must not be shared between Document instances."""
    doc1 = Document(document_id="d1", text="Text 1", source="s1.txt")
    doc2 = Document(document_id="d2", text="Text 2", source="s2.txt")

    doc1.metadata["key"] = "value_1"

    assert "key" not in doc2.metadata
    assert doc2.metadata == {}
    assert doc1.metadata != doc2.metadata


def test_document_serialization_round_trip() -> None:
    """Document should serialize to dict/JSON and deserialize identically."""
    original = Document(
        document_id="doc_100",
        text="Test content for serialization check.",
        source="test.md",
        metadata={"category": "benchmark", "version": 1},
    )

    # Dictionary round-trip
    dumped_dict = original.model_dump()
    from_dict = Document.model_validate(dumped_dict)
    assert from_dict == original

    # JSON round-trip
    dumped_json = original.model_dump_json()
    from_json = Document.model_validate_json(dumped_json)
    assert from_json == original

    # Verify JSON parsed structure
    parsed_json = json.loads(dumped_json)
    assert parsed_json["document_id"] == "doc_100"
    assert parsed_json["metadata"]["version"] == 1


# ============================================================================
# Chunk Tests
# ============================================================================


def test_chunk_valid_minimal() -> None:
    """A Chunk should instantiate successfully with valid boundary values."""
    chunk = Chunk(
        chunk_id="chunk_001_0",
        document_id="doc_001",
        text="Dense and sparse retrieval algorithms",
        chunk_index=0,
        source="retrieval_overview.txt",
        start_char=0,
        end_char=38,
    )
    assert chunk.chunk_id == "chunk_001_0"
    assert chunk.document_id == "doc_001"
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 38
    assert chunk.char_length == 38
    assert chunk.metadata == {}


@pytest.mark.parametrize(
    ("chunk_id", "doc_id", "text", "source"),
    [
        ("", "doc_1", "text", "src"),
        ("   ", "doc_1", "text", "src"),
        ("c_1", "", "text", "src"),
        ("c_1", "doc_1", "", "src"),
        ("c_1", "doc_1", "  \n  ", "src"),
        ("c_1", "doc_1", "text", ""),
    ],
)
def test_chunk_rejects_empty_strings(
    chunk_id: str, doc_id: str, text: str, source: str
) -> None:
    """Chunk must reject empty or blank strings for required identifier/text fields."""
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=text,
            chunk_index=0,
            source=source,
            start_char=0,
            end_char=10,
        )


def test_chunk_rejects_negative_chunk_index() -> None:
    """Chunk must reject negative chunk indices."""
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="c_1",
            document_id="d_1",
            text="Content",
            chunk_index=-1,
            source="src.txt",
            start_char=0,
            end_char=7,
        )


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [
        (-1, 10),  # Negative start offset
        (0, -5),  # Negative end offset
        (10, 10),  # Zero-span chunk (end == start)
        (20, 10),  # Inverted span (end < start)
    ],
)
def test_chunk_rejects_invalid_character_offsets(start_char: int, end_char: int) -> None:
    """Chunk must enforce start_char >= 0, end_char >= 0, and end_char > start_char."""
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="c_1",
            document_id="d_1",
            text="Content",
            chunk_index=0,
            source="src.txt",
            start_char=start_char,
            end_char=end_char,
        )


def test_chunk_metadata_independence() -> None:
    """Default metadata dicts between Chunk instances must remain isolated."""
    chunk1 = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="A",
        chunk_index=0,
        source="s.txt",
        start_char=0,
        end_char=1,
    )
    chunk2 = Chunk(
        chunk_id="c2",
        document_id="d1",
        text="B",
        chunk_index=1,
        source="s.txt",
        start_char=2,
        end_char=3,
    )

    chunk1.metadata["model"] = "test"
    assert "model" not in chunk2.metadata
    assert chunk2.metadata == {}


def test_chunk_serialization_round_trip() -> None:
    """Chunk must serialize and deserialize accurately without data corruption."""
    chunk = Chunk(
        chunk_id="chunk_42",
        document_id="doc_7",
        text="Exact chunk text for evaluation.",
        chunk_index=3,
        source="evaluation_guide.md",
        start_char=120,
        end_char=153,
        metadata={"token_count": 6, "overlap": False},
    )

    # Dict round trip
    dumped = chunk.model_dump()
    reconstructed = Chunk.model_validate(dumped)
    assert reconstructed == chunk

    # JSON round trip
    json_str = chunk.model_dump_json()
    from_json = Chunk.model_validate_json(json_str)
    assert from_json == chunk
    assert from_json.char_length == 33


# ============================================================================
# RetrievalResult Tests
# ============================================================================


@pytest.fixture
def sample_chunk() -> Chunk:
    """Fixture providing a valid Chunk for retrieval result tests."""
    return Chunk(
        chunk_id="chunk_fixture_0",
        document_id="doc_fixture",
        text="Sample text for testing retrieval scoring.",
        chunk_index=0,
        source="fixture.txt",
        start_char=0,
        end_char=42,
    )


def test_retrieval_result_valid(sample_chunk: Chunk) -> None:
    """RetrievalResult should instantiate cleanly with valid chunk, score, and rank."""
    result = RetrievalResult(
        chunk=sample_chunk,
        score=0.875,
        rank=1,
        retriever_name="bm25",
        metadata={"k1": 1.5, "b": 0.75},
    )
    assert result.chunk == sample_chunk
    assert result.score == 0.875
    assert result.rank == 1
    assert result.retriever_name == "bm25"
    assert result.metadata["k1"] == 1.5


@pytest.mark.parametrize("invalid_rank", [0, -1, -100])
def test_retrieval_result_rejects_non_positive_rank(
    sample_chunk: Chunk, invalid_rank: int
) -> None:
    """Rank must be a positive integer (>= 1)."""
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk=sample_chunk,
            score=1.0,
            rank=invalid_rank,
            retriever_name="bm25",
        )


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf")],
)
def test_retrieval_result_rejects_non_finite_scores(
    sample_chunk: Chunk, invalid_score: float
) -> None:
    """Retrieval score must be a finite number; NaN and infinities are rejected."""
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk=sample_chunk,
            score=invalid_score,
            rank=1,
            retriever_name="dense_faiss",
        )


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_retrieval_result_rejects_blank_retriever_name(
    sample_chunk: Chunk, name: str
) -> None:
    """Retriever name must contain non-whitespace text."""
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk=sample_chunk,
            score=0.5,
            rank=1,
            retriever_name=name,
        )


def test_retrieval_result_allows_negative_and_zero_scores(sample_chunk: Chunk) -> None:
    """Scores may be zero or negative depending on similarity metrics or distance models."""
    res_zero = RetrievalResult(chunk=sample_chunk, score=0.0, rank=1, retriever_name="custom")
    res_neg = RetrievalResult(chunk=sample_chunk, score=-0.42, rank=2, retriever_name="cosine")

    assert res_zero.score == 0.0
    assert res_neg.score == -0.42


def test_retrieval_result_metadata_independence(sample_chunk: Chunk) -> None:
    """Metadata dictionaries between RetrievalResult instances must be separate."""
    res1 = RetrievalResult(chunk=sample_chunk, score=1.0, rank=1, retriever_name="bm25")
    res2 = RetrievalResult(chunk=sample_chunk, score=0.5, rank=2, retriever_name="bm25")

    res1.metadata["fused"] = True
    assert "fused" not in res2.metadata


def test_retrieval_result_nested_serialization_round_trip(sample_chunk: Chunk) -> None:
    """Nested Chunk inside RetrievalResult must serialize and deserialize accurately."""
    result = RetrievalResult(
        chunk=sample_chunk,
        score=2.345,
        rank=1,
        retriever_name="hybrid_rrf",
        metadata={"rrf_k": 60, "bm25_weight": 0.5, "dense_weight": 0.5},
    )

    # Dict round trip
    as_dict = result.model_dump()
    assert isinstance(as_dict["chunk"], dict)
    reloaded_dict = RetrievalResult.model_validate(as_dict)
    assert reloaded_dict == result
    assert reloaded_dict.chunk.chunk_id == sample_chunk.chunk_id

    # JSON round trip
    json_text = result.model_dump_json()
    reloaded_json = RetrievalResult.model_validate_json(json_text)
    assert reloaded_json == result
    assert reloaded_json.chunk.start_char == 0
    assert reloaded_json.chunk.end_char == 42


# ============================================================================
# Assignment Validation Tests (validate_assignment=True)
# ============================================================================


def test_validate_assignment_enforced_on_all_models(sample_chunk: Chunk) -> None:
    """Assigning invalid values to existing instances must trigger ValidationError."""
    doc = Document(document_id="d1", text="Valid text", source="s1.txt")
    with pytest.raises(ValidationError):
        doc.text = ""  # Disallowed empty string

    with pytest.raises(ValidationError):
        sample_chunk.chunk_index = -5  # Disallowed negative index

    result = RetrievalResult(chunk=sample_chunk, score=1.0, rank=1, retriever_name="bm25")
    with pytest.raises(ValidationError):
        result.rank = 0  # Disallowed non-positive rank
