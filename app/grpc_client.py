import grpc
import uuid
import os
from datetime import datetime, timezone
from google.protobuf.timestamp_pb2 import Timestamp
from app import helios_pb2
from app import helios_pb2_grpc


class CalculatorClient:
    """gRPC Client for Calculator Service"""
    
    def __init__(self, host='localhost', port=8081, api_key=None):
        """Initialize the gRPC client
        
        Args:
            host: gRPC server host
            port: gRPC server port
            api_key: API key for authentication (default from env var GRPC_API_KEY)
        """
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = helios_pb2_grpc.HeliosServiceStub(self.channel)
        self.api_key = api_key or os.getenv("GRPC_API_KEY", "test-key-1")
    
    def _get_metadata(self):
        """Get metadata for gRPC calls including API key"""
        return [('api_key', self.api_key)]
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()
    
    def calculate(self, symbol, interval, start_time, end_time, features=None, indicators=None):
        """Calculate features and indicators for a given symbol and time range
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1m", "1h", "1d")
            start_time: Start timestamp in Unix seconds
            end_time: End timestamp in Unix seconds
            features: List of features to calculate (e.g., ["returns", "ma_21"])
            indicators: List of indicators to calculate (e.g., ["rsi_14", "macd"])
        
        Returns:
            CalculationCompleted response or None if error
        """
        if features is None:
            features = []
        if indicators is None:
            indicators = []
        
        request = helios_pb2.CalculationRequest(
            request_id=str(uuid.uuid4()),
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            features=features,
            indicators=indicators
        )
        
        try:
            response = self.stub.Calculate(request, metadata=self._get_metadata())
            return response
        except grpc.RpcError as e:
            print(f"gRPC Error: {e.code()} - {e.details()}")
            return None
    
    def get_ohlcv(self, symbol, interval, from_time, to_time, limit=100):
        """Get OHLCV data
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            from_time: Start timestamp
            to_time: End timestamp
            limit: Maximum number of candles
        
        Returns:
            GetOHLCVResponse or None if error
        """
        request = helios_pb2.GetOHLCVRequest(
            symbol=symbol,
            interval=interval,
            **{'from': from_time},
            to=to_time,
            limit=limit
        )
        
        try:
            response = self.stub.GetOHLCV(request, metadata=self._get_metadata())
            return response
        except grpc.RpcError as e:
            print(f"gRPC Error: {e.code()} - {e.details()}")
            return None
    
    def get_features(self, symbol, interval, features, from_time, to_time):
        """Get features
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            features: List of features to retrieve
            from_time: Start timestamp
            to_time: End timestamp
        
        Returns:
            GetFeaturesResponse or None if error
        """
        request = helios_pb2.GetFeaturesRequest(
            symbol=symbol,
            interval=interval,
            features=features,
            **{'from': from_time},
            to=to_time
        )
        
        try:
            response = self.stub.GetFeatures(request, metadata=self._get_metadata())
            return response
        except grpc.RpcError as e:
            print(f"gRPC Error: {e.code()} - {e.details()}")
            return None
    
    def get_indicators(self, symbol, interval, indicators, from_time, to_time):
        """Get indicators
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            indicators: List of indicators to retrieve
            from_time: Start timestamp
            to_time: End timestamp
        
        Returns:
            GetIndicatorsResponse or None if error
        """
        request = helios_pb2.GetIndicatorsRequest(
            symbol=symbol,
            interval=interval,
            indicators=indicators,
            **{'from': from_time},
            to=to_time
        )
        
        try:
            response = self.stub.GetIndicators(request, metadata=self._get_metadata())
            return response
        except grpc.RpcError as e:
            print(f"gRPC Error: {e.code()} - {e.details()}")
            return None


def timestamp_to_pb(ts):
    """Convert Unix timestamp to protobuf Timestamp
    
    Args:
        ts: Unix timestamp in seconds
    
    Returns:
        protobuf Timestamp
    """
    if isinstance(ts, int):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        dt = ts
    pb_ts = Timestamp()
    pb_ts.FromDatetime(dt)
    return pb_ts


def pb_to_timestamp(pb_ts):
    """Convert protobuf Timestamp to Unix timestamp
    
    Args:
        pb_ts: protobuf Timestamp
    
    Returns:
        Unix timestamp in seconds
    """
    dt = pb_ts.ToDatetime()
    return int(dt.timestamp())
