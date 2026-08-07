"""API dependencies — DI for JWT verifier and shared state."""
from typing import Optional
from metis.jwt_verifier import JWTVerifier


# ── JWT verifier (set during lifespan) ──
_jwt_verifier: Optional[JWTVerifier] = None

def set_jwt_verifier(verifier: JWTVerifier) -> None:
    global _jwt_verifier
    _jwt_verifier = verifier

def get_jwt_verifier() -> JWTVerifier:
    if _jwt_verifier is None:
        raise RuntimeError("JWT verifier not initialized")
    return _jwt_verifier
