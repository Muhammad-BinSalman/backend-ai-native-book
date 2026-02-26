"""
Pydantic models for health check responses.
"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    qdrant_connected: bool
    neon_connected: bool
    cohere_connected: bool
    collection_name: str
