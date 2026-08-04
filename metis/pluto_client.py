"""HTTP client for the Pluto personal-finance API (via the Nike gateway)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

from metis.config import get_settings

logger = logging.getLogger(__name__)


class PlutoApiError(RuntimeError):
    """Structured Pluto client error."""

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


class PlutoApiClient:
    """Async HTTP client with light retry logic for Pluto.

    Unlike ApolloApiClient, there is no fixed credential: every call carries
    the requesting user's own bearer token, so `token` is an explicit
    parameter on every method rather than something stored on the instance
    (this client's connection pool is a shared singleton across concurrent
    users' requests).
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self.base_url = (base_url or settings.pluto_base_url).rstrip("/")
        self.timeout = timeout or settings.pluto_request_timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
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

    # ── Financial profile & accounts (fetched unconditionally, "orientation" data) ──

    async def get_financial_profile(self, *, token: str) -> dict:
        return await self._request_json("GET", "/financial-profile", token=token)

    async def list_accounts(self, *, token: str) -> dict:
        return await self._request_json("GET", "/accounts", token=token)

    # ── Reports (fetched on-demand by finance tools) ──

    async def spending_by_category(self, *, token: str, date_from: str = "", date_to: str = "") -> dict:
        return await self._request_json(
            "GET", "/reports/spending-by-category", token=token,
            params=_prune({"date_from": date_from, "date_to": date_to}),
        )

    async def cashflow(self, *, token: str, date_from: str = "", date_to: str = "") -> dict:
        return await self._request_json(
            "GET", "/reports/cashflow", token=token,
            params=_prune({"date_from": date_from, "date_to": date_to}),
        )

    async def budget_progress(self, *, token: str) -> dict:
        return await self._request_json("GET", "/reports/budget-progress", token=token)

    async def goal_summary(self, *, token: str) -> dict:
        return await self._request_json("GET", "/reports/goal-summary", token=token)

    async def recurrences_due(self, *, token: str, within_days: int = 7) -> dict:
        return await self._request_json(
            "GET", "/reports/recurrences-due", token=token,
            params={"within_days": str(within_days)},
        )

    async def list_transactions(
        self,
        *,
        token: str,
        category_id: str = "",
        type: str = "",
        date_from: str = "",
        date_to: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        return await self._request_json(
            "GET", "/transactions", token=token,
            params=_prune({
                "category_id": category_id,
                "type": type,
                "date_from": date_from,
                "date_to": date_to,
                "page": str(page),
                "page_size": str(page_size),
            }),
        )

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
        last_error: Exception | None = None

        logger.debug(f"[PLUTO] HTTP {method} {url}")

        for attempt in range(2):
            try:
                logger.debug(f"[PLUTO] Tentativa {attempt + 1}/2")
                response = await client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=headers,
                )
                logger.debug(f"[PLUTO] Status HTTP: {response.status_code}")

                if response.is_error:
                    error_msg = self._extract_error_message(response)
                    logger.error(f"[PLUTO] ❌ Erro HTTP {response.status_code} | {error_msg}")
                    raise PlutoApiError(
                        error_msg,
                        status_code=response.status_code,
                        payload=self._safe_json(response),
                    )

                return self._safe_json(response)
            except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError, httpx.ConnectError) as exc:
                logger.warning(f"[PLUTO] ⚠️ Erro de transporte na tentativa {attempt + 1}: {type(exc).__name__}: {str(exc)}")
                last_error = exc
                await self.close()
                client = await self._get_client()
            except PlutoApiError:
                raise

        logger.error(f"[PLUTO] ❌ Falha após 2 tentativas: {last_error}")
        raise PlutoApiError(f"Pluto transport error: {last_error}")

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw": response.text}

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        payload = PlutoApiClient._safe_json(response)
        if isinstance(payload, dict):
            for key in ("detail", "message", "error", "raw"):
                value = payload.get(key)
                if value:
                    return f"Pluto API error ({response.status_code}): {value}"
        return f"Pluto API error ({response.status_code}): {response.text}"


def _prune(params: dict[str, str]) -> dict[str, str]:
    """Drops empty-string values so optional query params are omitted entirely."""
    return {k: v for k, v in params.items() if v}


_pluto_client: PlutoApiClient | None = None


def get_pluto_client() -> PlutoApiClient:
    """Get or create Pluto API client singleton."""
    global _pluto_client
    if _pluto_client is None:
        _pluto_client = PlutoApiClient()
    return _pluto_client


async def close_pluto_client() -> None:
    """Close Pluto client singleton."""
    global _pluto_client
    if _pluto_client is not None:
        await _pluto_client.close()
        _pluto_client = None
