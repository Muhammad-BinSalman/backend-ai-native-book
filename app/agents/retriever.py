"""
Retriever agent for Qdrant vector search.

Provides tool/function for retrieving relevant book chunks.
"""
from typing import List, Dict, Any, Optional
from qdrant_client.models import PointStruct

from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service


class RetrieverAgent:
    """Retriever agent for vector search."""

    async def retrieve(
        self,
        query: str,
        book_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User question
            book_id: Optional book filter
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score

        Returns:
            List of retrieved chunks with metadata
        """
        # Generate query embedding
        query_embedding = await cohere_service.embed_text(query)

        # Search Qdrant
        results = await qdrant_service.search(
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            book_id=book_id,
        )

        # Format results
        chunks = []
        for result in results:
            chunks.append(
                {
                    "chunk_id": result.payload.get("chunk_id"),
                    "text": result.payload.get("text"),
                    "source_file": result.payload.get("source_file"),
                    "chapter": result.payload.get("chapter"),
                    "section": result.payload.get("section"),
                    "position": result.payload.get("position"),
                    "score": result.score,
                }
            )

        return chunks

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for OpenAI Agents SDK."""
        return {
            "type": "function",
            "function": {
                "name": "retrieve_relevant_chunks",
                "description": "Retrieve relevant book chunks for a user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User question or query",
                        },
                        "book_id": {
                            "type": "string",
                            "description": "Optional book ID filter",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to retrieve",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


# Global retriever agent instance
retriever_agent = RetrieverAgent()
