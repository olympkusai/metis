from pathlib import Path
import sys

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

if __package__ in {None, ""}:
    # Support direct execution via `python app/main.py` by ensuring the
    # project root is searched before any installed `app` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import register_routes
from app.config import get_settings
from app.storage import DatabasePool
from app.storage.migrations import run_migrations
from app.tools import set_db_pool


# Global database pool
_db_pool: DatabasePool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for database pool initialization."""
    global _db_pool
    
    # Startup
    settings = get_settings()
    _db_pool = await DatabasePool.create(
        dsn=settings.database_url,
        min_size=10,
        max_size=50,
    )
    
    # Run migrations
    await run_migrations(_db_pool)
    
    # Set db pool for local tools
    set_db_pool(_db_pool)
    
    print(f"[MAIN] Database pool initialized with {settings.database_url}")
    print(f"[MAIN] Pool size: min=10, max=50")
    
    yield
    
    # Shutdown
    if _db_pool:
        await _db_pool.close()
        print("[MAIN] Database pool closed")


app = FastAPI(title="k0s - v0.0.1", lifespan=lifespan)

register_routes(app)

@app.get("/")
async def root():
    return {"message": "k0s - v0.0.1"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
