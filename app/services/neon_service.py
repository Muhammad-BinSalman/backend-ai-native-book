"""
Neon Postgres service for metadata persistence.

Provides CRUD operations for chunk metadata.
"""
from typing import List, Dict, Any, Optional
from app.db.connection import DatabaseConnection


class NeonService:
    """Neon Postgres service wrapper."""

    async def upsert_chunk_metadata(
        self,
        chunk_id: str,
        book_id: str,
        source_file: str,
        chapter: Optional[str],
        section: Optional[str],
        position: int,
        text: str,
        token_count: int,
    ) -> None:
        """Upsert chunk metadata to database."""
        query = """
            INSERT INTO chunks_metadata
            (chunk_id, book_id, source_file, chapter, section, position, text, token_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                chapter = EXCLUDED.chapter,
                section = EXCLUDED.section,
                position = EXCLUDED.position
        """

        await DatabaseConnection.execute(
            query,
            chunk_id,
            book_id,
            source_file,
            chapter,
            section,
            position,
            text,
            token_count,
        )

    async def get_chunk_metadata(
        self, chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific chunk."""
        query = """
            SELECT chunk_id, book_id, source_file, chapter, section, position, text, token_count, created_at
            FROM chunks_metadata
            WHERE chunk_id = $1
        """

        return await DatabaseConnection.execute(query, chunk_id, fetch="one")

    async def get_chunks_by_book(
        self, book_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a book."""
        query = """
            SELECT chunk_id, book_id, source_file, chapter, section, position, created_at
            FROM chunks_metadata
            WHERE book_id = $1
            ORDER BY position
            LIMIT $2 OFFSET $3
        """

        return await DatabaseConnection.execute(
            query, book_id, limit, offset, fetch="all"
        )

    async def delete_book_chunks(self, book_id: str) -> int:
        """Delete all chunks for a book."""
        query = "DELETE FROM chunks_metadata WHERE book_id = $1"

        # First get count
        count_query = "SELECT COUNT(*) FROM chunks_metadata WHERE book_id = $1"
        count = await DatabaseConnection.execute(count_query, book_id, fetch="val")

        # Then delete
        await DatabaseConnection.execute(query, book_id)

        return count


# Global Neon service instance
neon_service = NeonService()
