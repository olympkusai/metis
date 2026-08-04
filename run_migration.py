"""Run database migration to create make_interval function.

Reads DATABASE_URL from environment (or .env via metis.config). Never commit
credentials to source control.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metis.config import get_settings
from metis.storage.migrations import run_migrations
from metis.storage.pool import create_pool


async def main() -> None:
    dsn = get_settings().database_url
    if not dsn:
        raise SystemExit(
            "DATABASE_URL is not configured. Set it in .env or as an environment variable."
        )

    pool = await create_pool(dsn)
    try:
        print("Running database migration to create make_interval function...")
        await run_migrations(pool)
        print("Migration completed successfully!")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
