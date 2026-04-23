"""Run database migration to create make_interval function."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.storage.pool import create_pool
from app.storage.migrations import run_migrations

async def main():
    dsn = "postgres://postgres:BGxE9aWYJP5Ai7rhLkGeQUcnt8Y4hvnq3IM282m7OgEtKIF4QjmMUbIND07qCBR9@88.99.66.165:5432/k0s_prd?sslmode=require"
    pool = await create_pool(dsn)
    
    print("Running database migration to create make_interval function...")
    await run_migrations(pool)
    print("Migration completed successfully!")
    
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
