"""Database models for frequent calculations and market candles (read-only)."""
from datetime import datetime
from dataclasses import dataclass
from typing import Any


@dataclass
class MarketCandle:
    """Represents a market candle (READ-ONLY).
    
    Metis only reads from this table, never writes.
    """
    symbol: str = ""
    interval: str = ""
    open_time: datetime | None = None
    close_time: datetime | None = None
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    base_volume: float = 0.0
    quote_volume: float = 0.0
    closed: bool = False
    received_at: datetime | None = None
    
    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "MarketCandle":
        """Create from database record.
        
        Args:
            record: Database record
            
        Returns:
            MarketCandle instance
        """
        return cls(
            symbol=record.get("symbol", ""),
            interval=record.get("interval", ""),
            open_time=record.get("open_time"),
            close_time=record.get("close_time"),
            open_price=float(record.get("open_price", 0.0)),
            high_price=float(record.get("high_price", 0.0)),
            low_price=float(record.get("low_price", 0.0)),
            close_price=float(record.get("close_price", 0.0)),
            base_volume=float(record.get("base_volume", 0.0)),
            quote_volume=float(record.get("quote_volume", 0.0)),
            closed=record.get("closed", False),
            received_at=record.get("received_at"),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "base_volume": self.base_volume,
            "quote_volume": self.quote_volume,
            "closed": self.closed,
            "received_at": self.received_at,
        }


@dataclass
class FrequentCalculation:
    """Represents a frequently requested calculation."""
    id: int | None = None
    symbol: str = ""
    interval: str = ""
    calculation_type: str = ""  # 'feature' or 'indicator'
    name: str = ""  # e.g., 'ma_21', 'rsi_14'
    request_count: int = 0
    last_requested_at: datetime | None = None
    is_persisted: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    
    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "FrequentCalculation":
        """Create from database record.
        
        Args:
            record: Database record
            
        Returns:
            FrequentCalculation instance
        """
        return cls(
            id=record.get("id"),
            symbol=record.get("symbol", ""),
            interval=record.get("interval", ""),
            calculation_type=record.get("calculation_type", ""),
            name=record.get("name", ""),
            request_count=record.get("request_count", 0),
            last_requested_at=record.get("last_requested_at"),
            is_persisted=record.get("is_persisted", False),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "interval": self.interval,
            "calculation_type": self.calculation_type,
            "name": self.name,
            "request_count": self.request_count,
            "last_requested_at": self.last_requested_at,
            "is_persisted": self.is_persisted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class PersistedCalculation:
    """Represents a persisted calculation result."""
    id: int | None = None
    symbol: str = ""
    interval: str = ""
    calculation_type: str = ""
    name: str = ""
    data: bytes = b""
    expires_at: datetime | None = None
    created_at: datetime | None = None
    
    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PersistedCalculation":
        """Create from database record.
        
        Args:
            record: Database record
            
        Returns:
            PersistedCalculation instance
        """
        return cls(
            id=record.get("id"),
            symbol=record.get("symbol", ""),
            interval=record.get("interval", ""),
            calculation_type=record.get("calculation_type", ""),
            name=record.get("name", ""),
            data=record.get("data", b""),
            expires_at=record.get("expires_at"),
            created_at=record.get("created_at"),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "interval": self.interval,
            "calculation_type": self.calculation_type,
            "name": self.name,
            "data": self.data,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
        }
