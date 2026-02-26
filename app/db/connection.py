"""
Async database connection management for Neon Postgres.

Provides connection pool and lifecycle management.
"""
import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from app.config.settings import settings


class DatabaseConnection:
    """Async database connection manager."""

    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def create_pool(cls) -> asyncpg.Pool:
        """Create connection pool for Neon Postgres."""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                settings.neon_database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
        return cls._pool

    @classmethod
    async def close_pool(cls) -> None:
        """Close connection pool."""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    @asynccontextmanager
    async def get_connection(cls) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get database connection from pool."""
        if cls._pool is None:
            await cls.create_pool()
        async with cls._pool.acquire() as connection:
            yield connection

    @classmethod
    async def execute(
        cls, query: str, *args, fetch: str = "none"
    ) -> Optional[list] | Optional[str]:
        """Execute SQL query with optional fetch."""
        async with cls.get_connection() as conn:
            if fetch == "all":
                result = await conn.fetch(query, *args)
                return [dict(row) for row in result]
            elif fetch == "one":
                result = await conn.fetchrow(query, *args)
                return dict(result) if result else None
            elif fetch == "val":
                result = await conn.fetchval(query, *args)
                return result
            else:
                await conn.execute(query, *args)
                return None
