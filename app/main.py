"""
Main FastAPI application.

Entrypoint for the RAG chatbot backend API.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.api import ingest, chat, health

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else "DEBUG",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting RAG Chatbot Backend...")
    logger.info(f"📦 Qdrant collection: {settings.qdrant_collection_name}")
    logger.info(f"🤖 Cohere model: {settings.chat_model}")
    logger.info(f"🔢 Embedding model: {settings.embedding_model}")

    # Initialize services
    try:
        await qdrant_service.initialize()
        logger.info("✅ Qdrant service connected")

        await cohere_service.initialize_async()
        logger.info("✅ Cohere service connected")
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    from app.db.connection import DatabaseConnection
    await DatabaseConnection.close_pool()
    logger.info("✅ Database connection pool closed")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade RAG chatbot backend for published books",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://localhost:3001"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["X-API-KEY", "Authorization", "Content-Type"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )


# Include routers
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(health.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot Backend API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug)
