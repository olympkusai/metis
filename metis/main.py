from pathlib import Path
import logging
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
from metis.api.deps import set_jwt_verifier
from metis.pluto_client import close_pluto_client
from metis.soter_client import close_soter_client
from metis.jwt_verifier import JWTVerifier
from metis.config import get_settings
from metis.storage import DatabasePool
from metis.storage.migrations import run_conversation_migrations
from metis.memory.conversation_history import set_conversation_db_pool
from metis.request_id import RequestIdMiddleware, RequestIdLogFilter


# Global resources
_conversation_db_pool: DatabasePool | None = None
_jwt_verifier: JWTVerifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for database pool and JWT verifier initialization."""
    global _conversation_db_pool, _jwt_verifier

    # Startup — db-metis Postgres (conversations, chat_messages, notifications)
    settings = get_settings()
    _conversation_db_pool = await DatabasePool.create(
        dsn=settings.conversation_database_url,
        min_size=5,
        max_size=20,
    )
    await run_conversation_migrations(_conversation_db_pool)
    set_conversation_db_pool(_conversation_db_pool)
    print(f"[MAIN] Conversation database pool initialized (db-metis)")

    # JWT verifier (JWKS from Soter)
    _jwt_verifier = JWTVerifier(
        jwks_url=settings.soter_jwks_url,
        issuer=settings.oidc_issuer,
        cache_ttl=settings.jwks_cache_ttl_seconds,
    )
    await _jwt_verifier.warm()
    set_jwt_verifier(_jwt_verifier)
    print("[MAIN] JWT verifier initialized (JWKS from Soter)")

    yield

    # Shutdown
    if _jwt_verifier:
        await _jwt_verifier.close()
        print("[MAIN] JWT verifier closed")
    if _conversation_db_pool:
        await _conversation_db_pool.close()
        print("[MAIN] Conversation database pool closed")
    await close_pluto_client()
    await close_soter_client()


app = FastAPI(title="Metis", lifespan=lifespan)

# Configure logging with request ID
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(name)s] [req=%(request_id)s] %(levelname)s: %(message)s",
)
logging.getLogger().addFilter(RequestIdLogFilter())

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

# Request ID middleware — must be after CORS so the header is visible
app.add_middleware(RequestIdMiddleware)

register_routes(app)

@app.get("/")
async def root():
    return {"message": "Metis"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("metis.main:app", host="0.0.0.0", port=8082, reload=False)
