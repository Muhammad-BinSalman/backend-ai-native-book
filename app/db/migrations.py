"""
Database migrations for chunk metadata table.

Run this file to create the necessary schema in Neon Postgres.
"""
from app.db.connection import DatabaseConnection


async def create_chunks_metadata_table() -> None:
    """Create chunks_metadata table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS books (
        book_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT,
        format TEXT NOT NULL,
        file_path TEXT NOT NULL,
        ingested_at TIMESTAMPTZ DEFAULT NOW(),
        total_chunks INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS chunks_metadata (
        book_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL PRIMARY KEY,
        source_file TEXT NOT NULL,
        chapter TEXT,
        section TEXT,
        position INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),

        CONSTRAINT fk_book FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks_metadata(book_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_source_position ON chunks_metadata(source_file, position);
    CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks_metadata(chapter);
    """

    await DatabaseConnection.execute(create_table_query)
    print("OK Database schema created successfully")


async def drop_all_tables() -> None:
    """Drop all tables (use with caution - for testing only)."""
    drop_query = """
    DROP TABLE IF EXISTS chunks_metadata CASCADE;
    DROP TABLE IF EXISTS books CASCADE;
    """
    await DatabaseConnection.execute(drop_query)
    print("WARNING All tables dropped")


async def main() -> None:
    """Run migrations."""
    import argparse

    parser = argparse.ArgumentParser(description="Database migrations")
    parser.add_argument("--drop", action="store_true", help="Drop all tables first")
    args = parser.parse_args()

    await DatabaseConnection.create_pool()

    try:
        if args.drop:
            await drop_all_tables()
        await create_chunks_metadata_table()
    finally:
        await DatabaseConnection.close_pool()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
