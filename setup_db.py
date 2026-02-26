#!/usr/bin/env python3
"""
One-time setup script to initialize the RAG system.

1. Creates database tables
2. Ingests book content into Qdrant
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.db.migrations import create_chunks_metadata_table
from app.db.connection import DatabaseConnection
from scripts.ingest_book import BookIngestionOrchestrator


async def main():
    """Run full setup."""
    print("=" * 60)
    print("RAG Chatbot Setup")
    print("=" * 60)

    # Step 1: Create database schema
    print("\nStep 1: Creating database schema...")
    try:
        await DatabaseConnection.create_pool()
        # Drop existing tables to recreate with correct schema
        from app.db.migrations import drop_all_tables
        await drop_all_tables()
        await create_chunks_metadata_table()
        await DatabaseConnection.close_pool()
        print("OK Database schema created")
    except Exception as e:
        print(f"ERROR Database error: {e}")
        return

    # Step 2: Ingest book content
    print("\nStep 2: Ingesting book content...")

    # Path to content directory (from backend folder, go up to parent, then content)
    content_path = Path(__file__).parent.parent / "content" / "chapters"

    if not content_path.exists():
        print(f"ERROR Content path not found: {content_path}")
        return

    print(f"Ingesting from: {content_path}")

    try:
        orchestrator = BookIngestionOrchestrator(content_path)
        result = await orchestrator.ingest()

        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"  Book ID: {result['book_id']}")
        print(f"  Chunks created: {result['chunks_created']}")
        print(f"  Processing time: {result['processing_time_seconds']:.2f}s")
        print("\nYour chatbot is now ready!")
        print("   Start the backend: python -m app.main")
        print("   Start the frontend: npm run dev")

    except Exception as e:
        print(f"\nERROR Ingestion error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
