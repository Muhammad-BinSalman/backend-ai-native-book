"""
Main RAG agent for chat functionality.

Orchestrates retrieval, generation, and citation extraction using Cohere models via OpenAI Compatibility API.
"""
from typing import List, Dict, Any, Optional
import time

from app.config.settings import settings
from app.services.cohere_service import cohere_service
from app.services.qdrant_service import qdrant_service
from app.agents.retriever import retriever_agent
from app.agents.selected_text import selected_text_agent
from app.agents.router import router_agent
from app.models.chat import ChatRequest, Citation


class RAGAgent:
    """Main RAG agent for generating grounded responses."""

    # System prompt enforcing strict grounding
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided book content.

CRITICAL RULES:
1. You MUST answer questions using ONLY the retrieved book passages provided in the context
2. Every answer MUST include citations to specific book passages
3. If the retrieved passages do not contain information to answer the question, respond with: "I cannot answer this from the book content provided"
4. DO NOT use any outside knowledge or information not present in the retrieved passages
5. DO NOT make up or hallucinate information

Response Format:
- Provide a clear, direct answer to the question
- Include citations in format: [Chapter Name](source_file) or [Section Name](source_file)
- If multiple passages are relevant, synthesize information from all of them

Remember: Your goal is to be accurate and helpful while staying strictly grounded in the provided book content."""

    FALLBACK_RESPONSE = "I cannot answer this from the book content provided."

    def __init__(self) -> None:
        """Initialize RAG agent."""
        self.chat_model = settings.chat_model
        self.base_url = settings.cohere_base_url

    async def chat(
        self,
        request: ChatRequest,
        mode: str = "full_book",
    ) -> Dict[str, Any]:
        """
        Generate response to user query.

        Args:
            request: Chat request
            mode: Chat mode ("full_book" or "selected_text")

        Returns:
            Response with answer, citations, and metadata
        """
        start_time = time.time()

        try:
            if mode == "selected_text" and request.selected_text:
                # Use selected-text mode
                result = await self._chat_with_selected_text(request)
            else:
                # Use full-book RAG mode
                result = await self._chat_full_book(request)

            latency_ms = (time.time() - start_time) * 1000

            return {
                "answer": result["answer"],
                "citations": result["citations"],
                "mode": mode,
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "latency_ms": latency_ms,
                "model_used": self.chat_model,
            }

        except Exception as e:
            print(f"RAG Error: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback on error
            return {
                "answer": self.FALLBACK_RESPONSE,
                "citations": [],
                "mode": mode,
                "chunks_retrieved": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "model_used": self.chat_model,
            }

    async def _chat_full_book(self, request: ChatRequest) -> Dict[str, Any]:
        """Chat with full-book RAG mode."""
        # Retrieve relevant chunks
        chunks = await retriever_agent.retrieve(
            query=request.query,
            book_id=request.book_id,
            top_k=request.max_chunks,
        )

        # Check if we have relevant chunks
        if not chunks:
            return {
                "answer": self.FALLBACK_RESPONSE,
                "citations": [],
                "chunks_retrieved": 0,
            }

        # Build context from chunks
        context = self._build_context(chunks)

        # Generate response
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"},
        ]

        answer = await cohere_service.chat(messages=messages)

        # Extract citations
        citations = self._extract_citations(chunks)

        return {
            "answer": answer,
            "citations": citations,
            "chunks_retrieved": len(chunks),
        }

    async def _chat_with_selected_text(self, request: ChatRequest) -> Dict[str, Any]:
        """Chat with selected-text mode."""
        # Use selected-text agent
        result = await selected_text_agent.answer_with_selected_text(
            query=request.query,
            selected_text=request.selected_text,
            book_id=request.book_id,
            retrieve_additional=True,
        )

        # Build context
        context = result["forced_context"]

        # Add additional chunks if available
        if result["additional_chunks"]:
            additional_context = "\n\n".join([
                f"[{ch['source']}] {ch['text']}"
                for ch in result["additional_chunks"]
            ])
            context += f"\n\n[Additional Relevant Content]\n{additional_context}"

        # Generate response
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"},
        ]

        answer = await cohere_service.chat(messages=messages)

        # Extract citations (mark selected text specially)
        citations = [
            Citation(
                chunk_id="selected",
                text=request.selected_text[:200] + "..." if len(request.selected_text) > 200 else request.selected_text,
                source="User Selection",
                chapter=None,
                section=None,
                score=1.0,
            )
        ]

        # Add citations from additional chunks
        for ch in result["additional_chunks"]:
            citations.append(
                Citation(
                    chunk_id=ch["chunk_id"],
                    text=ch["text"][:200] + "..." if len(ch["text"]) > 200 else ch["text"],
                    source=ch["source"],
                    chapter=ch.get("chapter"),
                    section=ch.get("section"),
                    score=ch["score"],
                )
            )

        return {
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "chunks_retrieved": len(result["additional_chunks"]) + 1,
        }

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = f"[{chunk.get('source_file', 'unknown')}"
            if chunk.get("chapter"):
                source += f" - {chunk['chapter']}"
            context_parts.append(f"{source}\n{chunk['text']}")

        return "\n\n".join(context_parts)

    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """Extract citations from retrieved chunks."""
        citations = []
        for chunk in chunks:
            citations.append(
                Citation(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    source=chunk["source_file"],
                    chapter=chunk.get("chapter"),
                    section=chunk.get("section"),
                    score=chunk.get("score", 0.0),
                )
            )
        return citations


# Global RAG agent instance
rag_agent = RAGAgent()
