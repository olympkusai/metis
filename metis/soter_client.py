"""HTTP client for the Soter auth/identity service.

Reads user personalization preferences (AI tone, display name, obfuscation
level, language, channel toggles) so that Metis can tailor its responses
to each user's personality settings.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

from metis.config import get_settings
from metis.request_id import get_request_id

logger = logging.getLogger(__name__)


class SoterApiError(RuntimeError):
    """Structured Soter client error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload

    @property
    def unauthorized(self) -> bool:
        return self.status_code in {401, 403}

    @property
    def not_found(self) -> bool:
        return self.status_code == 404


class SoterApiClient:
    """Async HTTP client for Soter preference endpoints.

    Every call carries the requesting user's bearer token (same pattern as
    PlutoApiClient). The connection pool is a shared singleton across
    concurrent users' requests.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self.base_url = (base_url or settings.soter_base_url).rstrip("/")
        self.timeout = timeout or settings.soter_request_timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    )
                    self._client = httpx.AsyncClient(
                        timeout=self.timeout,
                        limits=limits,
                        http2=False,
                    )
        return self._client

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    # ── Application discovery ──

    async def get_app_id_by_client_id(self, *, client_id: str, token: str) -> str:
        """Discover an application's UUID by its client_id slug (e.g. 'pluto')."""
        data = await self._request_json(
            "GET", f"/applications/by-client-id/{client_id}", token=token,
        )
        if isinstance(data, dict) and "id" in data:
            return data["id"]
        raise SoterApiError(f"Unexpected response from Soter: {data}")

    # ── Personalization preferences ──

    async def get_personalization(self, *, token: str, app_id: str) -> dict | None:
        """Fetch the user's AI personalization preference for a given app.

        Returns ``None`` if the preference does not exist yet (404).
        """
        try:
            return await self._request_json(
                "GET", f"/me/preferences/{app_id}/personalization", token=token,
            )
        except SoterApiError as e:
            if e.not_found:
                return None
            raise

    # ── Low-level request ──

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        token: str,
        json_body: Any | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        rid = get_request_id()
        if rid:
            headers["X-Request-ID"] = rid

        logger.debug(f"[SOTER] HTTP {method} {url}")

        for attempt in range(2):
            try:
                response = await client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=headers,
                )

                if response.is_error:
                    error_msg = self._extract_error_message(response)
                    logger.error(f"[SOTER] HTTP {response.status_code} | {error_msg}")
                    raise SoterApiError(
                        error_msg,
                        status_code=response.status_code,
                        payload=self._safe_json(response),
                    )

                return self._safe_json(response)
            except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError, httpx.ConnectError) as exc:
                logger.warning(f"[SOTER] Transport error attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                await self.close()
                client = await self._get_client()
            except SoterApiError:
                raise

        raise SoterApiError(f"Soter transport error after retries")

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw": response.text}

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        payload = SoterApiClient._safe_json(response)
        if isinstance(payload, dict):
            for key in ("detail", "message", "error", "raw"):
                value = payload.get(key)
                if value:
                    return f"Soter API error ({response.status_code}): {value}"
        return f"Soter API error ({response.status_code}): {response.text}"


_soter_client: SoterApiClient | None = None


def get_soter_client() -> SoterApiClient:
    """Get or create Soter API client singleton."""
    global _soter_client
    if _soter_client is None:
        _soter_client = SoterApiClient()
    return _soter_client


async def close_soter_client() -> None:
    """Close Soter client singleton."""
    global _soter_client
    if _soter_client is not None:
        await _soter_client.close()
        _soter_client = None
