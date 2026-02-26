"""
Health check API endpoints.

Provides service health status and connectivity checks.
"""
from fastapi import APIRouter
from app.models.health import HealthResponse
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.db.connection import DatabaseConnection


router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service connectivity status for all dependencies.
    """
    from app.config.settings import settings

    # Check Qdrant
    qdrant_connected = False
    try:
        if qdrant_service.client:
            info = await qdrant_service.get_collection_info()
            qdrant_connected = info.get("status") == "ok"
    except Exception as e:
        qdrant_connected = False

    # Check Neon
    neon_connected = False
    try:
        result = await DatabaseConnection.execute("SELECT 1", fetch="val")
        neon_connected = result == 1
    except Exception:
        neon_connected = False

    # Check Cohere
    cohere_connected = False
    try:
        base_url = cohere_service.verify_base_url()
        cohere_connected = base_url == settings.cohere_base_url
    except Exception:
        cohere_connected = False

    return HealthResponse(
        status="healthy" if all([qdrant_connected, neon_connected, cohere_connected]) else "degraded",
        qdrant_connected=qdrant_connected,
        neon_connected=neon_connected,
        cohere_connected=cohere_connected,
        collection_name=settings.qdrant_collection_name,
    )
