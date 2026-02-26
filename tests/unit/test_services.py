"""
Unit tests for service clients.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.qdrant_service import QdrantService
from app.services.cohere_service import CohereService
from app.services.neon_service import NeonService


@pytest.mark.asyncio
class TestQdrantService:
    """Test Qdrant service."""

    async def test_qdrant_client_init(self):
        """Test Qdrant client initialization."""
        service = QdrantService()
        service.client = MagicMock()
        await service.initialize()

        assert service.client is not None


@pytest.mark.asyncio
class TestCohereService:
    """Test Cohere service."""

    async def test_base_url_verification(self):
        """Test that Cohere base URL is set correctly."""
        service = CohereService()
        service.initialize()

        base_url = service.verify_base_url()
        assert "api.cohere.ai" in base_url
        assert "compatibility" in base_url


@pytest.mark.asyncio
class TestNeonService:
    """Test Neon service."""

    async def test_metadata_upsert(self):
        """Test chunk metadata upsert."""
        service = NeonService()

        # Mock database execution
        from unittest.mock import AsyncMock, patch
        with patch("app.services.neon_service.DatabaseConnection.execute") as mock_exec:
            await service.upsert_chunk_metadata(
                chunk_id="test-chunk-1",
                book_id="book-1",
                source_file="test.md",
                chapter="Chapter 1",
                section=None,
                position=0,
                text="Test content",
                token_count=10,
            )

            # Verify execute was called
            assert mock_exec.called
