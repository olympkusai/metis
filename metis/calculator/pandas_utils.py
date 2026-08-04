"""Pandas utilities for optimized calculations."""
import numpy as np
import pandas as pd
from .types import Candle, FeatureResult, IndicatorResult


def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    """Convert candles to pandas DataFrame for optimized operations.
    
    Args:
        candles: List of Candle objects
        
    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    data = {
        'open': [c.open_price for c in candles],
        'high': [c.high_price for c in candles],
        'low': [c.low_price for c in candles],
        'close': [c.close_price for c in candles],
        'volume': [c.base_volume for c in candles],
        'quote_volume': [c.quote_volume for c in candles],
        'timestamp': [c.close_time for c in candles],
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def dataframe_to_feature_results(df: pd.DataFrame, column_name: str, feature_name: str) -> list[FeatureResult]:
    """Convert DataFrame column to FeatureResult objects.
    
    Args:
        df: DataFrame with results
        column_name: Column name containing the values
        feature_name: Name for the feature
        
    Returns:
        List of FeatureResult objects
    """
    results = []
    for timestamp, value in df[column_name].items():
        results.append(FeatureResult(
            name=feature_name,
            value=float(value),
            timestamp=timestamp
        ))
    return results


def dataframe_to_indicator_results(df: pd.DataFrame, column_name: str, indicator_name: str) -> list[IndicatorResult]:
    """Convert DataFrame column to IndicatorResult objects.
    
    Args:
        df: DataFrame with results
        column_name: Column name containing the values
        indicator_name: Name for the indicator
        
    Returns:
        List of IndicatorResult objects
    """
    results = []
    for timestamp, value in df[column_name].items():
        results.append(IndicatorResult(
            name=indicator_name,
            value=float(value),
            timestamp=timestamp
        ))
    return results


def rolling_mean_pandas(df: pd.DataFrame, window: int, column: str = 'close') -> pd.Series:
    """Calculate rolling mean using pandas.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size
        column: Column to calculate mean for
        
    Returns:
        Series with rolling mean values
    """
    return df[column].rolling(window=window, min_periods=1).mean()


def rolling_std_pandas(df: pd.DataFrame, window: int, column: str = 'close') -> pd.Series:
    """Calculate rolling standard deviation using pandas.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size
        column: Column to calculate std for
        
    Returns:
        Series with rolling std values
    """
    return df[column].rolling(window=window, min_periods=1).std(ddof=0)


def ewma_pandas(df: pd.DataFrame, span: int, column: str = 'close') -> pd.Series:
    """Calculate exponentially weighted moving average using pandas.
    
    Args:
        df: DataFrame with OHLCV data
        span: Span for EWMA
        column: Column to calculate EWMA for
        
    Returns:
        Series with EWMA values
    """
    return df[column].ewm(span=span, adjust=False).mean()


def calculate_returns_pandas(df: pd.DataFrame, column: str = 'close') -> pd.Series:
    """Calculate returns using pandas.
    
    Args:
        df: DataFrame with OHLCV data
        column: Column to calculate returns for
        
    Returns:
        Series with returns (first value is 0)
    """
    returns = df[column].pct_change()
    returns.iloc[0] = 0.0
    return returns.fillna(0.0)


def calculate_log_returns_pandas(df: pd.DataFrame, column: str = 'close') -> pd.Series:
    """Calculate logarithmic returns using pandas.
    
    Args:
        df: DataFrame with OHLCV data
        column: Column to calculate log returns for
        
    Returns:
        Series with log returns (first value is 0)
    """
    log_returns = np.log(df[column] / df[column].shift(1))
    log_returns.iloc[0] = 0.0
    return log_returns.fillna(0.0)
