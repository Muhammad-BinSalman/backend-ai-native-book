"""
Ingestion API endpoints.

Provides REST API for book ingestion.
"""
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.ingest import IngestRequest, IngestResponse
from app.services.neon_service import neon_service
from app.services.qdrant_service import qdrant_service


router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_book(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest book content into the RAG knowledge base.

    Args:
        request: Ingest request with book_path and optional book_id

    Returns:
        IngestResponse with book_id, chunks_created, status
    """
    try:
        book_path = Path(request.book_path)

        if not book_path.exists():
            raise HTTPException(
                status_code=400, detail=f"Path does not exist: {request.book_path}"
            )

        # Import orchestrator here to avoid circular imports
        from scripts.ingest_book import BookIngestionOrchestrator

        orchestrator = BookIngestionOrchestrator(book_path)

        # If book_id provided, delete existing chunks first (idempotent)
        if request.book_id:
            await qdrant_service.delete_by_book(request.book_id)
            await neon_service.delete_book_chunks(request.book_id)
            orchestrator.book_id = request.book_id

        # Run ingestion
        result = await orchestrator.ingest()

        return IngestResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/chunks")
async def list_chunks(
    book_id: str,
    limit: int = 100,
    offset: int = 0,
):
    """
    List chunks for debugging and verification.

    Args:
        book_id: Book identifier
        limit: Maximum chunks to return
        offset: Pagination offset

    Returns:
        List of chunk metadata
    """
    try:
        chunks = await neon_service.get_chunks_by_book(
            book_id, limit=limit, offset=offset
        )
        return {"chunks": chunks, "count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chunks: {str(e)}")
