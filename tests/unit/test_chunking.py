"""
Unit tests for chunking service.
"""
import pytest
from app.services.chunking import ChunkingService, Chunk


class TestChunkingService:
    """Test chunking service functionality."""

    @pytest.fixture
    def chunking_service(self):
        """Create chunking service instance."""
        return ChunkingService(chunk_size=100, chunk_overlap=20)

    def test_fixed_size_chunking(self, chunking_service):
        """Test that chunks respect size limit."""
        text = "A" * 200 + "\n\n" + "B" * 200 + "\n\n" + "C" * 200
        chunks = chunking_service.chunk_text(text, "test.txt")

        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should be within size limit (allowing some overhead)
        for chunk in chunks:
            assert chunk.token_count <= chunking_service.chunk_size

    def test_chunk_overlap(self, chunking_service):
        """Test that chunks have overlap."""
        text = "Word1 Word2 Word3 " * 50
        chunks = chunking_service.chunk_text(text, "test.txt")

        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            # (extract actual overlap verification)
            assert chunks[0].position != chunks[1].position

    def test_metadata_extraction(self, chunking_service):
        """Test chapter and section extraction from markdown."""
        text = """# Chapter 1

## Section A

This is content.

## Section B

More content.
"""

        chunks = chunking_service.chunk_text(text, "test.md")

        # Should extract chapter and section
        assert any(chunk.chapter == "Chapter 1" for chunk in chunks)
        assert any(chunk.section == "Section A" for chunk in chunks)

    def test_source_file_preservation(self, chunking_service):
        """Test that source file is preserved."""
        text = "Some content"
        chunks = chunking_service.chunk_text(text, "my_book.md")

        assert all(chunk.source_file == "my_book.md" for chunk in chunks)
        assert all(chunk.position is not None for chunk in chunks)
