"""
Chat API endpoints.

Provides REST API for RAG and simple chat.
"""
import asyncio
import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.chat import ChatRequest, ChatResponse, Citation
from app.services.cohere_service import cohere_service
from app.services.qdrant_service import qdrant_service


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# RAG system prompt - ensures answers are grounded in retrieved content
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided book passages.

CRITICAL RULES:
1. You MUST answer questions using ONLY the retrieved book passages provided in the context
2. If the passages do not contain information to answer the question, say "Based on the provided content, I don't have enough information to answer this question."
3. DO NOT use outside knowledge or information not present in the retrieved passages
4. DO NOT make up or hallucinate information

Response Format:
- Provide a clear, direct answer to the question
- Reference the passages you used in your answer
- Synthesize information from multiple passages if relevant"""

# Simple system prompt for general questions
SIMPLE_SYSTEM_PROMPT = """You are a helpful AI assistant.

Your role is to:
- Answer questions clearly and concisely
- Provide practical examples when relevant
- Be friendly and conversational
- Help users with software development, AI, and technical topics
- If you don't know something specific, be honest about it"""


async def retrieve_context(query: str, book_id: str = None, top_k: int = 5):
    """
    Retrieve relevant chunks from Qdrant for the query.

    Args:
        query: User question
        book_id: Optional book filter
        top_k: Number of chunks to retrieve

    Returns:
        List of retrieved chunks with metadata
    """
    # Generate query embedding
    query_embedding = await cohere_service.embed_text(query)

    # Search Qdrant
    results = await qdrant_service.search(
        query_vector=query_embedding,
        limit=top_k,
        book_id=book_id,
    )

    # Format results into chunks
    chunks = []
    for result in results:
        chunks.append({
            "chunk_id": result.payload.get("chunk_id"),
            "text": result.payload.get("text"),
            "source_file": result.payload.get("source_file"),
            "chapter": result.payload.get("chapter"),
            "section": result.payload.get("section"),
            "score": result.score,
        })

    return chunks


def build_context(chunks: list) -> str:
    """Build context string from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk.get('source_file', 'unknown')}"
        if chunk.get("chapter"):
            source += f" - {chunk['chapter']}"
        context_parts.append(f"{source}\n{chunk['text']}")

    return "\n\n".join(context_parts)


def extract_citations(chunks: list) -> list:
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


@router.post("/rag", response_model=ChatResponse)
async def rag_chat(request: ChatRequest):
    """
    RAG chat endpoint that retrieves relevant book content with fallback to simple chat.

    Uses Qdrant to find relevant passages. If passages are found with high confidence,
    generates grounded response. Otherwise, falls back to general AI response.
    """
    start_time = time.time()

    try:
        # Retrieve relevant chunks
        chunks = await retrieve_context(
            query=request.query,
            book_id=request.book_id,
            top_k=request.max_chunks or 5
        )

        # Filter chunks by relevance score (e.g., 0.7)
        # Only use chunks that are actually relevant to the question
        relevant_chunks = [c for c in chunks if c.get("score", 0) >= 0.4]

        # If we have relevant chunks, use RAG
        if relevant_chunks:
            # Build context from chunks
            context = build_context(relevant_chunks)

            # Generate response with context
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"},
            ]

            answer = await cohere_service.chat(messages=messages)

            # Extract citations
            citations = extract_citations(relevant_chunks)

            return ChatResponse(
                answer=answer,
                citations=citations,
                mode="rag",
                chunks_retrieved=len(relevant_chunks),
                latency_ms=(time.time() - start_time) * 1000,
                model_used=cohere_service.chat_model,
            )
        
        # Fallback to simple chat if no relevant context found
        messages = [
            {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": request.query},
        ]

        answer = await cohere_service.chat(messages=messages)

        return ChatResponse(
            answer=answer,
            citations=[],
            mode="simple",
            chunks_retrieved=0,
            latency_ms=(time.time() - start_time) * 1000,
            model_used=cohere_service.chat_model,
        )

    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()

        # Return error response
        return ChatResponse(
            answer=f"Sorry, I encountered an error: {str(e)}",
            citations=[],
            mode="rag",
            chunks_retrieved=0,
            latency_ms=(time.time() - start_time) * 1000,
            model_used=cohere_service.chat_model,
        )

    except Exception as e:
        print(f"RAG Chat error: {e}")
        import traceback
        traceback.print_exc()

        # Return error response
        return ChatResponse(
            answer=f"Sorry, I encountered an error: {str(e)}",
            citations=[],
            mode="rag",
            chunks_retrieved=0,
            latency_ms=(time.time() - start_time) * 1000,
            model_used=cohere_service.chat_model,
        )


@router.post("/simple", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    """
    Simple chat endpoint without RAG context.

    Directly queries the AI model without retrieving book content.
    """
    start_time = time.time()

    try:
        # Generate response using Cohere
        messages = [
            {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": request.query},
        ]

        answer = await cohere_service.chat(messages=messages)

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            answer=answer,
            citations=[],
            mode="simple",
            chunks_retrieved=0,
            latency_ms=latency_ms,
            model_used=cohere_service.chat_model,
        )

    except Exception as e:
        print(f"Simple Chat error: {e}")
        import traceback
        traceback.print_exc()

        # Return error response
        return ChatResponse(
            answer=f"Sorry, I encountered an error: {str(e)}",
            citations=[],
            mode="simple",
            chunks_retrieved=0,
            latency_ms=(time.time() - start_time) * 1000,
            model_used=cohere_service.chat_model,
        )


@router.post("", response_model=ChatResponse)
async def chat_unified(request: ChatRequest):
    """
    Main chat endpoint (uses RAG mode).

    Routes to RAG chat with book content retrieval.
    """
    return await rag_chat(request)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response as Server-Sent Events (SSE).

    This endpoint does NOT change the existing JSON contract at `/api/v1/chat`.
    It emits incremental `delta` events followed by a final `final` event
    containing citations and metadata.

    Note: the underlying model call is currently non-streaming; this endpoint
    simulates streaming by chunking the final answer.
    """

    async def event_generator():
        start_time = time.time()
        try:
            response: ChatResponse = await rag_chat(request)

            answer = response.answer or ""
            chunk_size = 24
            for i in range(0, len(answer), chunk_size):
                delta = answer[i : i + chunk_size]
                payload = {"type": "delta", "delta": delta}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            final_payload = {
                "type": "final",
                "citations": [c.model_dump() for c in response.citations],
                "mode": response.mode,
                "chunks_retrieved": response.chunks_retrieved,
                "latency_ms": (time.time() - start_time) * 1000,
                "model_used": response.model_used,
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            err_payload = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
