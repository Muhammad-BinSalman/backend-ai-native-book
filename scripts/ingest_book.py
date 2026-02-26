#!/usr/bin/env python3
"""
Book ingestion CLI script.

Processes markdown/text book files into searchable chunks with embeddings.
"""
import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import List, Tuple
import time

from app.config.settings import settings
from app.services.chunking import ChunkingService, Chunk
from app.services.cohere_service import cohere_service
from app.services.qdrant_service import qdrant_service, PointStruct
from app.db.connection import DatabaseConnection
from qdrant_client.models import PointStruct


class BookIngestionOrchestrator:
    """Orchestrates book ingestion workflow."""

    def __init__(self, book_path: Path):
        """Initialize orchestrator."""
        self.book_path = Path(book_path)
        self.chunking_service = ChunkingService()
        self.chunks: List[Chunk] = []
        self.book_id: str = ""
        self.total_files = 0
        self.total_chunks = 0

    async def ingest(self) -> dict:
        """Run full ingestion workflow."""
        start_time = time.time()

        print(f"Starting book ingestion from: {self.book_path}")

        # Step 1: Discover files
        files = self._discover_files()
        self.total_files = len(files)
        print(f"Found {self.total_files} files")

        # Step 2: Generate book ID
        self.book_id = self._generate_book_id()
        print(f"Book ID: {self.book_id}")

        # Step 3: Process files and create chunks
        await self._process_files(files)

        # Step 4: Generate embeddings
        print(f"\nGenerating embeddings for {len(self.chunks)} chunks...")
        await self._generate_embeddings()

        # Step 5: Store in Qdrant
        print(f"Storing vectors in Qdrant...")
        await self._store_in_qdrant()

        # Step 6: Store metadata in Neon (skip for now - Qdrant is enough)
        print(f"Skipping Neon metadata storage (Qdrant already has the data)...")
        # await self._store_metadata()

        elapsed = time.time() - start_time
        print(f"\nIngestion complete in {elapsed:.2f}s")
        print(f"   Files processed: {self.total_files}")
        print(f"   Chunks created: {self.total_chunks}")

        return {
            "book_id": self.book_id,
            "chunks_created": self.total_chunks,
            "status": "success",
            "processing_time_seconds": elapsed,
        }

    def _discover_files(self) -> List[Path]:
        """Discover all markdown and text files."""
        if self.book_path.is_file():
            files = [self.book_path]
        else:
            files = list(self.book_path.rglob("*.md")) + list(
                self.book_path.rglob("*.mdx")
            ) + list(self.book_path.rglob("*.txt"))

        return sorted(files)

    def _generate_book_id(self) -> str:
        """Generate unique book ID from path hash."""
        path_str = str(self.book_path.absolute())
        return hashlib.md5(path_str.encode()).hexdigest()

    async def _process_files(self, files: List[Path]) -> None:
        """Process all files and create chunks."""
        position = 0
        errors = []

        for file_path in files:
            try:
                print(f"  Processing: {file_path.name}")
                text = file_path.read_text(encoding="utf-8")

                # Chunk the text
                chunks = self.chunking_service.chunk_text(text, str(file_path))

                # Update positions
                for chunk in chunks:
                    chunk.position = position
                    position += 1

                self.chunks.extend(chunks)
                print(f"    Created {len(chunks)} chunks")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                errors.append(f"{file_path}: {e}")

        self.total_chunks = len(self.chunks)

        if errors:
            print(f"\n⚠️  Errors encountered:")
            for error in errors:
                print(f"  - {error}")

    async def _generate_embeddings(self) -> None:
        """Generate embeddings for all chunks."""
        batch_size = 10
        batch_texts = [
            chunk.text for chunk in self.chunks
        ]

        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i:i+batch_size]
            embeddings = await cohere_service.embed_batch(batch)

            # Store embeddings in chunks
            for j, embedding in enumerate(embeddings):
                chunk_idx = i + j
                self.chunks[chunk_idx].embedding = embedding

            print(f"  Processed {min(i+batch_size, len(batch_texts))}/{len(batch_texts)} chunks")

    async def _store_in_qdrant(self) -> None:
        """Store chunks as vectors in Qdrant."""
        # Initialize Qdrant service if needed
        await qdrant_service.initialize()

        # Check storage before ingestion (free-tier monitoring)
        collection_info = await qdrant_service.get_collection_info()
        current_count = collection_info.get("points_count", 0)
        vector_count = collection_info.get("vectors_count", 0)

        print(f"  Current collection state: {vector_count} vectors, {current_count} points")

        # Warn if approaching free-tier limit (1GB)
        # Rough estimate: 1K vectors with 1024-dim float32 ≈ 4MB
        # 250K vectors ≈ 1GB limit
        if vector_count > 200000:
            print(f"  ⚠️  WARNING: Approaching Qdrant free-tier storage limit!")
            print(f"     Current: {vector_count} vectors (≈{vector_count * 4 / 1024:.1f} MB)")
            print(f"     Free tier limit: ~250K vectors (1 GB)")

        # Create points
        points = []
        import uuid
        for i, chunk in enumerate(self.chunks):
            # Generate deterministic UUID for point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.book_id}-{i}"))
            point = PointStruct(
                id=point_id,
                vector=chunk.embedding,
                payload={
                    "book_id": self.book_id,
                    "chunk_id": f"{self.book_id}-{i}",
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "position": chunk.position,
                },
            )
            points.append(point)

        # Upsert to Qdrant in batches (smaller batches for reliability)
        batch_size = 50
        batches_stored = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            end_idx = min(i+batch_size, len(points))
            try:
                await qdrant_service.upsert_chunks(batch)
                print(f"  Stored batch {end_idx}/{len(points)} vectors")
                batches_stored += 1
            except Exception as e:
                print(f"  Error storing batch {i}-{end_idx}: {e}")
                # Retry once
                import asyncio
                await asyncio.sleep(2)
                try:
                    await qdrant_service.upsert_chunks(batch)
                    print(f"  Retry successful for batch {end_idx}/{len(points)}")
                    batches_stored += 1
                except Exception as e2:
                    print(f"  Retry failed: {e2}")
                    raise
            
        print(f"  Stored total {len(points)} vectors in collection: {qdrant_service.collection_name}")

    async def _store_metadata(self) -> None:
        """Store chunk metadata in Neon Postgres."""
        # Initialize connection pool
        await DatabaseConnection.create_pool()

        for i, chunk in enumerate(self.chunks):
            insert_query = """
                INSERT INTO chunks_metadata
                (chunk_id, book_id, source_file, chapter, section, position, text, token_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    chapter = EXCLUDED.chapter,
                    section = EXCLUDED.section
            """

            await DatabaseConnection.execute(
                insert_query,
                f"{self.book_id}-{i}",
                self.book_id,
                chunk.source_file,
                chunk.chapter,
                chunk.section,
                chunk.position,
                chunk.text,
                chunk.token_count,
            )

        print(f"  Stored {len(self.chunks)} chunk metadata records in Neon")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest book content into RAG chatbot knowledge base"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to book directory or file",
    )

    args = parser.parse_args()
    book_path = Path(args.path)

    if not book_path.exists():
        print(f"❌ Error: Path does not exist: {book_path}")
        return

    # Run ingestion
    orchestrator = BookIngestionOrchestrator(book_path)
    result = await orchestrator.ingest()

    print(f"\n📊 Summary:")
    print(f"  Book ID: {result['book_id']}")
    print(f"  Chunks: {result['chunks_created']}")
    print(f"  Time: {result['processing_time_seconds']:.2f}s")
    print(f"  Status: {result['status']}")


if __name__ == "__main__":
    asyncio.run(main())
