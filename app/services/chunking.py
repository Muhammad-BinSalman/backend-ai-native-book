"""
Text chunking service for book content.

Implements fixed-size chunking with overlap.
"""
import re
from typing import List, Tuple
from dataclasses import dataclass

from app.config.settings import settings


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    source_file: str
    chapter: str | None
    section: str | None
    position: int
    token_count: int
    embedding: List[float] = None  # Added for ingestion


class ChunkingService:
    """Service for chunking text into manageable segments."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """Initialize chunking service."""
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_text(
        self,
        text: str,
        source_file: str,
    ) -> List[Chunk]:
        """Chunk text into overlapping segments."""
        chunks = []
        position = 0

        # Split by paragraphs first for better semantic boundaries
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = ""
        current_position = 0

        for paragraph in paragraphs:
            # Check if adding paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(
                        Chunk(
                            text=current_chunk.strip(),
                            source_file=source_file,
                            chapter=self._extract_chapter(current_chunk),
                            section=self._extract_section(current_chunk),
                            position=current_position,
                            token_count=self._estimate_tokens(current_chunk),
                        )
                    )
                    current_position += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(chunks) > 0:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    source_file=source_file,
                    chapter=self._extract_chapter(current_chunk),
                    section=self._extract_section(current_chunk),
                    position=current_position,
                    token_count=self._estimate_tokens(current_chunk),
                )
            )

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines
        paragraphs = re.split(r'\n\n+', text)

        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _extract_chapter(self, text: str) -> str | None:
        """Extract chapter from markdown headers (#)."""
        match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_section(self, text: str) -> str | None:
        """Extract section from markdown subheaders (##)."""
        match = re.search(r'^##\s+(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        words = text.split()
        overlap_words = words[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
        return " ".join(overlap_words)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4


# Global chunking service instance
chunking_service = ChunkingService()
