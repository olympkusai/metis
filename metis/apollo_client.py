"""HTTP client for Apollo ML forecasting API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

from metis.agent.schemas import (
    ApolloBacktestOutput,
    ApolloPredictionOutput,
    ApolloTrainingOutput,
)
from metis.config import get_settings

logger = logging.getLogger(__name__)


class ApolloApiError(RuntimeError):
    """Structured Apollo client error."""

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
    def missing_model(self) -> bool:
        """Best-effort detection for model-not-trained failures."""
        text = str(self).lower()
        markers = (
            "model not found",
            "model not trained",
            "model unavailable",
            "no model",
            "not trained",
            "missing model",
            "file not found",
        )
        return self.status_code in {404, 409, 422} or any(marker in text for marker in markers)


class ApolloApiClient:
    """Async HTTP client with light retry logic for Apollo."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        settings = get_settings()
        self.base_url = (base_url or settings.apollo_base_url).rstrip("/")
        self.timeout = timeout
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

    async def predict(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> ApolloPredictionOutput:
        logger.info(f"[APOLLO] Iniciando predict para {symbol} | período: {start_date} até {end_date}")
        try:
            data = await self._request_json(
                "POST",
                "/ml/predict",
                json_body={
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            result = ApolloPredictionOutput.model_validate(data)
            logger.info(f"[APOLLO] ✅ Predict bem-sucedido | {symbol} | direção: {result.direction} | confiança: {result.confidence:.1%}")
            return result
        except Exception as e:
            logger.error(f"[APOLLO] ❌ Predict falhou para {symbol}: {str(e)}")
            raise

    async def train(
        self,
        *,
        symbol: str,
        lookback_days: int,
        use_walk_forward: bool = True,
        tft_max_epochs: int = 10,
        xgb_n_estimators: int = 200,
    ) -> ApolloTrainingOutput:
        logger.info(f"[APOLLO] Iniciando train para {symbol} | lookback: {lookback_days} dias")
        try:
            data = await self._request_json(
                "POST",
                "/ml/train",
                json_body={
                    "symbol": symbol,
                    "lookback_days": lookback_days,
                    "use_walk_forward": use_walk_forward,
                    "tft_max_epochs": tft_max_epochs,
                    "xgb_n_estimators": xgb_n_estimators,
                },
            )
            result = ApolloTrainingOutput.model_validate(data)
            logger.info(f"[APOLLO] ✅ Train iniciado | {symbol} | status: {result.status}")
            return result
        except Exception as e:
            logger.error(f"[APOLLO] ❌ Train falhou para {symbol}: {str(e)}")
            raise


    async def backtest(
        self,
        *,
        symbol: str,
        num_periods: int = 5,
    ) -> ApolloBacktestOutput:
        logger.info(f"[APOLLO] Iniciando backtest para {symbol} | períodos: {num_periods}")
        try:
            data = await self._request_json(
                "POST",
                "/ml/backtest",
                params={
                    "symbol": symbol,
                    "num_periods": str(num_periods),
                },
            )
            result = ApolloBacktestOutput.model_validate(data)
            logger.info(f"[APOLLO] ✅ Backtest concluído | {symbol} | períodos testados: {result.periods_tested}")
            return result
        except Exception as e:
            logger.error(f"[APOLLO] ❌ Backtest falhou para {symbol}: {str(e)}")
            raise

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        logger.debug(f"[APOLLO] HTTP {method} {url}")
        if json_body:
            logger.debug(f"[APOLLO] Body: {json.dumps(json_body, indent=2)}")
        if params:
            logger.debug(f"[APOLLO] Params: {params}")

        for attempt in range(2):
            try:
                logger.debug(f"[APOLLO] Tentativa {attempt + 1}/2")
                response = await client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers={"Content-Type": "application/json"},
                )
                logger.debug(f"[APOLLO] Status HTTP: {response.status_code}")

                if response.is_error:
                    error_msg = self._extract_error_message(response)
                    logger.error(f"[APOLLO] ❌ Erro HTTP {response.status_code} | {error_msg}")
                    raise ApolloApiError(
                        error_msg,
                        status_code=response.status_code,
                        payload=self._safe_json(response),
                    )

                result = self._safe_json(response)
                logger.debug(f"[APOLLO] ✅ Resposta bem-sucedida")
                return result
            except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError, httpx.ConnectError) as exc:
                logger.warning(f"[APOLLO] ⚠️ Erro de transporte na tentativa {attempt + 1}: {type(exc).__name__}: {str(exc)}")
                last_error = exc
                await self.close()
                client = await self._get_client()
            except ApolloApiError as e:
                logger.error(f"[APOLLO] ❌ Erro da API: {str(e)}")
                raise

        logger.error(f"[APOLLO] ❌ Falha após 2 tentativas: {last_error}")
        raise ApolloApiError(f"Apollo transport error: {last_error}")

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw": response.text}

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        payload = ApolloApiClient._safe_json(response)
        if isinstance(payload, dict):
            for key in ("detail", "message", "error", "raw"):
                value = payload.get(key)
                if value:
                    return f"Apollo API error ({response.status_code}): {value}"
        return f"Apollo API error ({response.status_code}): {response.text}"


_apollo_client: ApolloApiClient | None = None


def get_apollo_client() -> ApolloApiClient:
    """Get or create Apollo API client singleton."""
    global _apollo_client
    if _apollo_client is None:
        _apollo_client = ApolloApiClient()
    return _apollo_client


async def close_apollo_client() -> None:
    """Close Apollo client singleton."""
    global _apollo_client
    if _apollo_client is not None:
        await _apollo_client.close()
        _apollo_client = None
