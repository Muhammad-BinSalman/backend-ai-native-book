"""
Cohere embedding and chat service.

Uses OpenAI SDK with Cohere's OpenAI Compatibility API.
"""
from typing import List, Optional
from openai import AsyncOpenAI, OpenAI

from app.config.settings import settings


class CohereService:
    """Cohere API client using OpenAI Compatibility API."""

    def __init__(self) -> None:
        """Initialize Cohere-compatible OpenAI client."""
        self.sync_client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        self.embedding_model = settings.embedding_model
        self.chat_model = settings.chat_model
        self.base_url = settings.cohere_base_url

    def initialize(self) -> None:
        """Initialize synchronous Cohere client."""
        self.sync_client = OpenAI(
            api_key=settings.cohere_api_key,
            base_url=self.base_url,
        )
        print(f"✅ Cohere client initialized: {self.base_url}")

    async def initialize_async(self) -> None:
        """Initialize asynchronous Cohere client."""
        self.async_client = AsyncOpenAI(
            api_key=settings.cohere_api_key,
            base_url=self.base_url,
        )

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.async_client is None:
            await self.initialize_async()

        response = await self.async_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.async_client is None:
            await self.initialize_async()

        # Cohere supports batch embedding
        response = await self.async_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query (alias for embed_text)."""
        return await self.embed_text(query)

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Generate chat response."""
        if self.async_client is None:
            await self.initialize_async()

        response = await self.async_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def verify_base_url(self) -> str:
        """Verify Cohere base URL is set correctly."""
        if self.sync_client is None:
            self.initialize()

        return self.sync_client.base_url


# Global Cohere service instance
cohere_service = CohereService()
