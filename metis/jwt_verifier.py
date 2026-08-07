"""JWT verifier — validates RS256 access tokens against Soter's JWKS.

Mirrors the Nike gateway's verifier pattern: cache public keys with a TTL,
refresh lazily on cache miss or staleness, and fall back to the cached key
if the refresh fails. No shared secret — only the public key is needed.

Usage:
    verifier = JWTVerifier(jwks_url, issuer)
    identity = await verifier.verify(token_string)  # → Identity or raises InvalidTokenError
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

logger = logging.getLogger(__name__)


class InvalidTokenError(Exception):
    """Raised when a JWT fails validation."""
    pass


@dataclass
class Identity:
    """Trusted subject extracted from a verified access token."""
    user_id: str
    email: str
    username: str
    user_type: str


class JWTVerifier:
    """Validates RS256 access tokens against a remote JWKS endpoint.

    Thread-safe via an asyncio.Lock around the refresh path. The key cache
    is a simple dict keyed by `kid` with a TTL — on stale or missing key,
    a refresh is attempted; if the refresh fails but the kid is known,
    the cached key is served (graceful degradation).
    """

    def __init__(
        self,
        jwks_url: str,
        issuer: str = "",
        audience: str = "",
        cache_ttl: float = 600.0,  # 10 minutes
        http_timeout: float = 10.0,
    ):
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.cache_ttl = cache_ttl
        self._client = httpx.AsyncClient(timeout=http_timeout)
        self._lock = asyncio.Lock()
        self._keys: dict[str, str] = {}  # kid → PEM-encoded public key
        self._fetched_at: float = 0.0

    async def verify(self, token_string: str) -> Identity:
        """Parse and cryptographically validate the token.

        Raises InvalidTokenError on any failure (expired, bad signature,
        wrong issuer, missing subject, etc.).
        """
        try:
            unverified_header = jwt.get_unverified_header(token_string)
            kid = unverified_header.get("kid", "")
            if not kid:
                raise InvalidTokenError("missing kid in token header")

            key_pem = await self._get_key(kid)
            if key_pem is None:
                raise InvalidTokenError(f"no key found for kid={kid}")

            options = {"verify_signature": True, "require": ["exp", "iat"]}
            kwargs: dict = {"algorithms": ["RS256"], "key": key_pem}
            if self.issuer:
                kwargs["issuer"] = self.issuer
            if self.audience:
                kwargs["audience"] = self.audience

            payload = jwt.decode(token_string, options=options, **kwargs)

            user_id = payload.get("sub", "")
            if not user_id:
                raise InvalidTokenError("missing subject in token")

            return Identity(
                user_id=user_id,
                email=payload.get("email", ""),
                username=payload.get("username", ""),
                user_type=payload.get("user_type", ""),
            )
        except jwt.ExpiredSignatureError:
            raise InvalidTokenError("token expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"invalid token: {e}")
        except InvalidTokenError:
            raise
        except Exception as e:
            raise InvalidTokenError(f"unexpected error: {e}")

    async def _get_key(self, kid: str) -> Optional[str]:
        """Get a cached key by kid, refreshing if stale or missing."""
        if kid in self._keys and not self._is_stale():
            return self._keys[kid]

        async with self._lock:
            if kid in self._keys and not self._is_stale():
                return self._keys[kid]

            try:
                await self._refresh()
            except Exception as e:
                logger.warning(f"[JWT] JWKS refresh failed: {e}")
                return self._keys.get(kid)

            return self._keys.get(kid)

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) > self.cache_ttl

    async def _refresh(self) -> None:
        """Fetch and cache the JWKS from the remote endpoint."""
        resp = await self._client.get(self.jwks_url)
        if resp.status_code != 200:
            raise RuntimeError(f"JWKS endpoint returned {resp.status_code}")

        body = resp.json()
        keys_raw = body.get("keys", [])
        if not keys_raw:
            raise RuntimeError("JWKS contained no keys")

        new_keys: dict[str, str] = {}
        for k in keys_raw:
            if k.get("kty") != "RSA" or not k.get("kid"):
                continue
            try:
                pem = RSAAlgorithm.from_jwk(k)
                new_keys[k["kid"]] = pem
            except Exception as e:
                logger.warning(f"[JWT] Failed to parse key kid={k.get('kid')}: {e}")
                continue

        if not new_keys:
            raise RuntimeError("JWKS contained no usable RSA keys")

        self._keys = new_keys
        self._fetched_at = time.monotonic()
        logger.info(f"[JWT] JWKS refreshed: {len(new_keys)} key(s) cached")

    async def warm(self) -> None:
        """Eagerly load the JWKS at startup. Failures are non-fatal."""
        try:
            async with self._lock:
                await self._refresh()
        except Exception as e:
            logger.warning(f"[JWT] Warm-up failed (will retry lazily): {e}")

    async def close(self) -> None:
        await self._client.aclose()
