"""
Pydantic models for book ingestion.
"""
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class IngestRequest(BaseModel):
    """Request model for book ingestion."""

    book_path: str = Field(..., description="Path to book directory or file")
    book_id: Optional[str] = Field(None, description="Optional book ID for updates")
    format: str = Field(default="md", description="File format (md, txt, pdf)")


class IngestResponse(BaseModel):
    """Response model for book ingestion."""

    book_id: str
    chunks_created: int
    status: str
    message: str
    processing_time_seconds: float


class ChunkMetadata(BaseModel):
    """Metadata for a content chunk."""

    chunk_id: str
    book_id: str
    source_file: str
    chapter: Optional[str]
    section: Optional[str]
    position: int
    text: str
    token_count: int
    created_at: datetime
