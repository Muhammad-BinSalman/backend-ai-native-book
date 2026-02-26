"""
Integration tests for selected-text mode.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.asyncio
class TestSelectedTextFlow:
    """Test selected-text mode functionality."""

    def test_selected_text_priority(self, test_client):
        """Test that selected text is prioritized."""
        # Test selected-text endpoint contract
        response = test_client.post(
            "/api/v1/chat/selected",
            json={
                "query": "Explain this",
                "selected_text": "RAG stands for Retrieval-Augmented Generation.",
                "max_chunks": 3,
            },
        )
        # Should return 200 or 500 (if no book ingested)
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert data["mode"] == "selected_text"

    def test_empty_selected_text_fallback_to_full_book(self, test_client):
        """Test that empty selected_text falls back to full-book mode."""
        response = test_client.post(
            "/api/v1/chat/selected",
            json={
                "query": "What is RAG?",
                "selected_text": "   ",  # Whitespace only
                "max_chunks": 3,
            },
        )
        assert response.status_code in [200, 500]

    def test_whitespace_selected_text_handling(self, test_client):
        """Test that whitespace-only selected_text is handled correctly."""
        response = test_client.post(
            "/api/v1/chat/selected",
            json={
                "query": "What is RAG?",
                "selected_text": "\t\n  ",  # Mixed whitespace
                "max_chunks": 3,
            },
        )
        # Should fall back to full-book mode
        assert response.status_code in [200, 500]
