from pathlib import Path
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

if __package__ in {None, ""}:
    # Support direct execution via `python metis/main.py` by ensuring the
    # project root is searched before any installed `metis` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metis.api import register_routes
from metis.apollo_client import close_apollo_client
from metis.pluto_client import close_pluto_client
from metis.config import get_settings
from metis.storage import DatabasePool
from metis.storage.migrations import run_migrations, run_conversation_migrations
from metis.tools import set_db_pool
from metis.memory.conversation_history import set_conversation_db_pool


# Global database pools
_db_pool: DatabasePool | None = None
_conversation_db_pool: DatabasePool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for database pool initialization."""
    global _db_pool, _conversation_db_pool

    # Startup — db-metis Postgres (market data + calculator tables)
    settings = get_settings()
    _db_pool = await DatabasePool.create(
        dsn=settings.database_dsn,
        min_size=10,
        max_size=50,
    )
    await run_migrations(_db_pool)
    set_db_pool(_db_pool)
    print(f"[MAIN] Database pool initialized with {settings.database_dsn}")
    print(f"[MAIN] Pool size: min=10, max=50")

    # Startup — db-metis Postgres (conversations, chat_messages, notifications)
    _conversation_db_pool = await DatabasePool.create(
        dsn=settings.conversation_database_url,
        min_size=5,
        max_size=20,
    )
    await run_conversation_migrations(_conversation_db_pool)
    set_conversation_db_pool(_conversation_db_pool)
    print(f"[MAIN] Conversation database pool initialized (db-metis)")

    yield

    # Shutdown
    if _db_pool:
        await _db_pool.close()
        print("[MAIN] Database pool closed")
    if _conversation_db_pool:
        await _conversation_db_pool.close()
        print("[MAIN] Conversation database pool closed")
    await close_apollo_client()
    await close_pluto_client()


app = FastAPI(title="Metis", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://olympkusai.com",
        "https://www.olympkusai.com",
        "https://api.olympkusai.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

register_routes(app)

@app.get("/")
async def root():
    return {"message": "Metis"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("metis.main:app", host="0.0.0.0", port=8082, reload=False)
