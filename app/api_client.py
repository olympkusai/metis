"""
HTTP REST Client for k0s.app API
Replaces the gRPC client for OHLCV, features, and indicators data.
"""

import asyncio
import httpx
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.k0s.app/api/v1")


@dataclass
class OHLCVCandle:
    """Represents a single OHLCV candle."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: int  # milliseconds
    close_time: int  # milliseconds


@dataclass
class FeatureData:
    """Represents a single feature value with timestamp."""
    name: str
    value: float
    timestamp: int  # milliseconds


@dataclass
class IndicatorData:
    """Represents a single indicator value with timestamp."""
    name: str
    value: float
    timestamp: int  # milliseconds


@dataclass
class OHLCVResponse:
    """Response from OHLCV endpoint."""
    symbol: str
    interval: str
    candles: List[OHLCVCandle]


@dataclass
class FeaturesResponse:
    """Response from features endpoint."""
    symbol: str
    interval: str
    features: List[FeatureData]


@dataclass
class IndicatorsResponse:
    """Response from indicators endpoint."""
    symbol: str
    interval: str
    indicators: List[IndicatorData]


class K0sApiClient:
    """HTTP REST Client for k0s.app API"""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """Initialize the HTTP client

        Args:
            base_url: API base URL (default from env var API_BASE_URL)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or API_BASE_URL).rstrip('/')
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client (thread-safe)."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    # Configure connection limits to prevent exhaustion
                    limits = httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
                        keepalive_expiry=30.0
                    )
                    self._client = httpx.AsyncClient(
                        timeout=self.timeout,
                        limits=limits,
                        http2=False  # Disable HTTP/2 for better compatibility
                    )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        async with self._lock:
            if self._client:
                await self._client.aclose()
                self._client = None

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        from_time: int,  # milliseconds
        to_time: int,  # milliseconds
        limit: int = 100
    ) -> Optional[OHLCVResponse]:
        """Get OHLCV data from REST API

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1m", "1h", "1d")
            from_time: Start timestamp in milliseconds
            to_time: End timestamp in milliseconds
            limit: Maximum number of candles

        Returns:
            OHLCVResponse or None if error
        """
        client = await self._get_client()

        # Ensure timestamps are in milliseconds
        from_ms = self._ensure_milliseconds(from_time)
        to_ms = self._ensure_milliseconds(to_time)

        url = f"{self.base_url}/ohlcv"
        params = {
            "symbol": symbol,
            "interval": interval,
            "from": str(from_ms),
            "to": str(to_ms),
            "limit": str(limit)
        }

        print(f"[API CLIENT] Requesting OHLCV: {url}")
        print(f"[API CLIENT] Params: {params}")

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError) as e:
            # Connection closed error - recreate client and retry once
            print(f"[API CLIENT] Connection error in get_ohlcv, retrying: {e}")
            await self.close()
            client = await self._get_client()
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"[API CLIENT] Retry successful, got {len(data.get('candles', []))} candles")

        try:
            print(f"[API CLIENT] Response status: {response.status_code}")
            print(f"[API CLIENT] Response data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
            if isinstance(data, dict) and 'error' in data:
                print(f"[API CLIENT] API returned error: {data['error']}")
            # Handle response format: may be a list or a dict with 'candles' key
            if isinstance(data, dict) and "candles" in data:
                data = data["candles"]

            candles = []
            for item in data:
                # Handle both list format [open_time, open, high, low, close, volume, close_time]
                # and dict format {"open", "high", "low", "close", "volume", "open_time", "close_time"}
                if isinstance(item, list) and len(item) >= 6:
                    try:
                        candle = OHLCVCandle(
                            open=float(item[1]),
                            high=float(item[2]),
                            low=float(item[3]),
                            close=float(item[4]),
                            volume=float(item[5]),
                            open_time=int(item[0]),
                            close_time=int(item[6]) if len(item) > 6 else int(item[0])
                        )
                        # Validate required fields
                        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
                            continue
                        if candle.high < candle.low:  # Invalid: high should be >= low
                            continue
                        candles.append(candle)
                    except (ValueError, IndexError, TypeError):
                        continue
                elif isinstance(item, dict):
                    try:
                        candle = OHLCVCandle(
                            open=float(item.get("open", 0)),
                            high=float(item.get("high", 0)),
                            low=float(item.get("low", 0)),
                            close=float(item.get("close", 0)),
                            volume=float(item.get("volume", 0)),
                            open_time=int(item.get("open_time", 0)),
                            close_time=int(item.get("close_time", 0))
                        )
                        # Validate required fields
                        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
                            continue
                        if candle.high < candle.low:  # Invalid: high should be >= low
                            continue
                        candles.append(candle)
                    except (ValueError, TypeError):
                        continue
                else:
                    continue

            return OHLCVResponse(
                symbol=symbol,
                interval=interval,
                candles=candles
            )

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return None

    async def get_features(
        self,
        symbol: str,
        interval: str,
        features: List[str],
        from_time: int,  # milliseconds
        to_time: int  # milliseconds
    ) -> Optional[FeaturesResponse]:
        """Get features data from REST API

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            features: List of features to retrieve
            from_time: Start timestamp in milliseconds
            to_time: End timestamp in milliseconds

        Returns:
            FeaturesResponse or None if error
        """
        client = await self._get_client()

        # Ensure timestamps are in milliseconds
        from_ms = self._ensure_milliseconds(from_time)
        to_ms = self._ensure_milliseconds(to_time)

        url = f"{self.base_url}/features"
        params = {
            "symbol": symbol,
            "interval": interval,
            "from": str(from_ms),
            "to": str(to_ms),
            "features": ",".join(features)
        }

        print(f"[API CLIENT] Requesting Features: {url}")
        print(f"[API CLIENT] Params: {params}")

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError) as e:
            # Connection closed error - recreate client and retry once
            print(f"[API CLIENT] Connection error in get_features, retrying: {e}")
            await self.close()
            client = await self._get_client()
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"[API CLIENT] Retry successful, got {len(data.get('features', []))} features")

        try:
            print(f"[API CLIENT] Response status: {response.status_code}")
            print(f"[API CLIENT] Response data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
            if isinstance(data, dict) and 'error' in data:
                print(f"[API CLIENT] API returned error: {data['error']}")
            # Handle response format: may be a list or a dict with 'features' key
            if isinstance(data, dict) and "features" in data:
                data = data["features"]

            feature_data = []
            # Handle response format
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # API returns items with "name", "value", "timestamp" fields
                        if "name" in item and "value" in item:
                            # Parse timestamp from ISO format or use as-is if already ms
                            ts = item.get("timestamp")
                            if isinstance(ts, str):
                                from datetime import datetime, timezone
                                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                ts_ms = int(dt.timestamp() * 1000)
                            else:
                                ts_ms = int(ts) if ts else from_ms
                            
                            feature_data.append(FeatureData(
                                name=item["name"],
                                value=float(item["value"]),
                                timestamp=ts_ms
                            ))

            return FeaturesResponse(
                symbol=symbol,
                interval=interval,
                features=feature_data
            )

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Error fetching features: {e}")
            return None

    async def get_indicators(
        self,
        symbol: str,
        interval: str,
        indicators: List[str],
        from_time: int,  # milliseconds
        to_time: int  # milliseconds
    ) -> Optional[IndicatorsResponse]:
        """Get indicators data from REST API

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            indicators: List of indicators to retrieve
            from_time: Start timestamp in milliseconds
            to_time: End timestamp in milliseconds

        Returns:
            IndicatorsResponse or None if error
        """
        client = await self._get_client()

        # Ensure timestamps are in milliseconds
        from_ms = self._ensure_milliseconds(from_time)
        to_ms = self._ensure_milliseconds(to_time)

        url = f"{self.base_url}/indicators"
        params = {
            "symbol": symbol,
            "interval": interval,
            "from": str(from_ms),
            "to": str(to_ms),
            "indicators": ",".join(indicators)
        }

        print(f"[API CLIENT] Requesting Indicators: {url}")
        print(f"[API CLIENT] Params: {params}")

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError) as e:
            # Connection closed error - recreate client and retry once
            print(f"[API CLIENT] Connection error in get_indicators, retrying: {e}")
            await self.close()
            client = await self._get_client()
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"[API CLIENT] Retry successful, got {len(data.get('indicators', []))} indicators")

        try:
            print(f"[API CLIENT] Response status: {response.status_code}")
            print(f"[API CLIENT] Response data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
            if isinstance(data, dict) and 'error' in data:
                print(f"[API CLIENT] API returned error: {data['error']}")
            # Handle response format: may be a list or a dict with 'indicators' key
            if isinstance(data, dict) and "indicators" in data:
                data = data["indicators"]

            indicator_data = []
            # Handle response format
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # API returns items with "name", "value", "timestamp" fields
                        if "name" in item and "value" in item:
                            # Parse timestamp from ISO format or use as-is if already ms
                            ts = item.get("timestamp")
                            if isinstance(ts, str):
                                from datetime import datetime, timezone
                                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                ts_ms = int(dt.timestamp() * 1000)
                            else:
                                ts_ms = int(ts) if ts else from_ms
                            
                            indicator_data.append(IndicatorData(
                                name=item["name"],
                                value=float(item["value"]),
                                timestamp=ts_ms
                            ))

            return IndicatorsResponse(
                symbol=symbol,
                interval=interval,
                indicators=indicator_data
            )

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Error fetching indicators: {e}")
            return None

    @staticmethod
    def _ensure_milliseconds(timestamp: int) -> int:
        """Ensure timestamp is in milliseconds.

        If timestamp appears to be in seconds (less than 1 trillion),
        convert it to milliseconds.

        Args:
            timestamp: Timestamp value (could be seconds or milliseconds)

        Returns:
            Timestamp in milliseconds
        """
        # If timestamp is less than 1 trillion, it's likely in seconds
        # (1 trillion seconds = year ~33658)
        if timestamp < 1_000_000_000_000:
            return timestamp * 1000
        return timestamp

    @staticmethod
    def timestamp_now_ms() -> int:
        """Get current timestamp in milliseconds."""
        return int(datetime.now(timezone.utc).timestamp() * 1000)
