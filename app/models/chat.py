"""
Pydantic models for chat requests and responses.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation reference to a book passage."""

    chunk_id: str
    text: str
    source: str
    chapter: Optional[str]
    section: Optional[str]
    score: float


class ChatRequest(BaseModel):
    """Request model for chat query."""

    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    selected_text: Optional[str] = Field(
        None,
        max_length=5000,
        description="Optional selected passage for focused Q&A",
    )
    book_id: Optional[str] = Field(None, description="Optional book ID filter")
    mode: Literal["full_book", "selected_text"] = Field(
        default="full_book", description="Chat mode"
    )
    max_chunks: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")


class ChatResponse(BaseModel):
    """Response model for chat query."""

    answer: str
    citations: List[Citation]
    mode: str
    chunks_retrieved: int
    latency_ms: float
    model_used: str
