"""
Selected-text agent for focused passage Q&A.

Handles user-selected text passages with highest priority context.
"""
from typing import List, Dict, Any, Optional
from app.services.cohere_service import cohere_service
from app.services.qdrant_service import qdrant_service


class SelectedTextAgent:
    """Selected-text agent for passage-focused Q&A."""

    async def answer_with_selected_text(
        self,
        query: str,
        selected_text: str,
        book_id: Optional[str] = None,
        retrieve_additional: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer query using selected text as primary context.

        Args:
            query: User question
            selected_text: User-selected passage
            book_id: Optional book filter
            retrieve_additional: Whether to retrieve additional chunks

        Returns:
            Answer with citations
        """
        # Sanitize selected text
        selected_text = selected_text.strip()[:5000]  # Max 5000 chars

        # Add selected text as forced context (highest priority)
        forced_context = f"[Selected Text]\n{selected_text}\n"

        # Optionally retrieve additional chunks
        additional_chunks = []
        if retrieve_additional:
            # Embed selected text for similarity search
            selected_embedding = await cohere_service.embed_text(selected_text)

            # Search for similar chunks
            results = await qdrant_service.search(
                query_vector=selected_embedding,
                limit=3,
                book_id=book_id,
            )

            additional_chunks = [
                {
                    "chunk_id": r.payload.get("chunk_id"),
                    "text": r.payload.get("text"),
                    "source": r.payload.get("source_file"),
                    "score": r.score,
                }
                for r in results
            ]

        return {
            "forced_context": forced_context,
            "selected_text": selected_text,
            "additional_chunks": additional_chunks,
            "mode": "selected_text",
        }


# Global selected-text agent instance
selected_text_agent = SelectedTextAgent()
