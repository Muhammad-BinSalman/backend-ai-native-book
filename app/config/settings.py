"""
Application configuration using Pydantic BaseSettings.

Loads all environment variables and provides validation.
"""
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required: Neon Postgres
    neon_database_url: str = Field(
        ..., description="Neon Serverless Postgres connection string"
    )

    # Required: Qdrant Cloud
    qdrant_cluster_id: str = Field(..., description="Qdrant cluster identifier")
    qdrant_api_endpoint: str = Field(..., description="Qdrant API endpoint URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")

    # Required: Cohere
    cohere_api_key: str = Field(..., description="Cohere API key")

    # Optional: Collection name
    qdrant_collection_name: str = Field(
        default="book_chunks", description="Qdrant collection name"
    )

    # Optional: Model configurations
    embedding_model: str = Field(
        default="embed-english-v3.0", description="Cohere embedding model"
    )
    chat_model: str = Field(
        default="command-a-03-2025", description="Cohere chat model"
    )

    # Optional: Retrieval parameters
    max_retrieved_chunks: int = Field(
        default=5, ge=1, le=20, description="Maximum chunks to retrieve"
    )
    chunk_size: int = Field(
        default=500, ge=100, le=2000, description="Chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Chunk overlap in tokens"
    )

    # Optional: Cohere Compatibility API
    cohere_base_url: str = Field(
        default="https://api.cohere.ai/compatibility/v1",
        description="Cohere OpenAI Compatibility API base URL"
    )

    # Optional: Application
    app_name: str = Field(default="RAG Chatbot Backend", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    @field_validator("neon_database_url")
    @classmethod
    def validate_neon_url(cls, v: str) -> str:
        """Validate Neon database URL format."""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError(
                "NEON_DATABASE_URL must start with 'postgresql://' or 'postgresql+asyncpg://'"
            )
        return v

    @field_validator("qdrant_api_endpoint")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant API endpoint."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT_API_ENDPOINT must be a valid URL")
        return v


# Global settings instance
settings = Settings()
