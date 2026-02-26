"""
Pytest fixtures for testing.

Provides test clients, mock services, and sample data.
"""
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from app.config.settings import settings
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_qdrant_service():
    """Mock Qdrant service for testing."""
    service = MagicMock(spec=qdrant_service)
    service.collection_name = settings.qdrant_collection_name
    service.initialize = AsyncMock()
    service.collection_exists = AsyncMock(return_value=True)
    service.upsert_chunks = AsyncMock()
    service.search = AsyncMock(return_value=[])
    service.delete_by_book = AsyncMock()
    service.get_collection_info = AsyncMock(return_value={"status": "ok"})
    yield service


@pytest.fixture
async def mock_cohere_service():
    """Mock Cohere service for testing."""
    service = MagicMock(spec=cohere_service)
    service.embedding_model = settings.embedding_model
    service.chat_model = settings.chat_model
    service.initialize_async = AsyncMock()
    service.embed_text = AsyncMock(return_value=[0.1] * 1024)
    service.embed_batch = AsyncMock(return_value=[[0.1] * 1024] * 5)
    service.chat = AsyncMock(return_value="Test response")
    service.verify_base_url = MagicMock(return_value=settings.cohere_base_url)
    yield service


@pytest.fixture
def sample_book_content():
    """Sample book content for testing."""
    return """# Chapter 1: Introduction to RAG

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with generative AI. It allows language models to access external knowledge bases before generating responses.

## How RAG Works

The RAG process consists of three main steps:

1. **Retrieval**: Find relevant documents from a knowledge base
2. **Augmentation**: Add retrieved documents to the model context
3. **Generation**: Generate a response using the augmented context

This approach ensures that AI responses are grounded in actual data rather than purely in the model's training parameters.

# Chapter 2: Vector Databases

## Introduction

Vector databases store high-dimensional vectors that represent the semantic meaning of text. They enable fast similarity search to find related content.
"""


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": "chunk-1",
            "text": "RAG stands for Retrieval-Augmented Generation.",
            "source_file": "chapter1.md",
            "chapter": "Introduction to RAG",
            "position": 0,
        },
        {
            "chunk_id": "chunk-2",
            "text": "The RAG process has three main steps: Retrieval, Augmentation, and Generation.",
            "source_file": "chapter1.md",
            "chapter": "How RAG Works",
            "position": 1,
        },
    ]


@pytest.fixture
async def test_client():
    """Create test FastAPI client."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    yield client
