# Calculator Service API Documentation

## Overview

The Calculator Service provides technical analysis calculations for cryptocurrency market data through a message-based architecture using RabbitMQ.

## Architecture

- **Protocol**: RabbitMQ (AMQP)
- **Exchange**: `calculation.events`
- **Request Queue**: `q.calculation.requests`
- **Response Routing Keys**: 
  - `calculation.completed.v1` - Successful calculations
  - `calculation.failed.v1` - Failed calculations
- **Dead Letter Queue**: `q.calculation.requests.dlq`

## Request Format

### Message Structure

```json
{
  "request_id": "uuid-v4",
  "symbol": "BTCUSDT",
  "interval": "1m",
  "start_time": 1234567890,
  "end_time": 1234567950,
  "features": ["returns", "ma_21", "volatility_7"],
  "indicators": ["rsi_14", "macd"]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | Yes | Unique identifier for the request (UUID v4) |
| `symbol` | string | Yes | Trading pair symbol (e.g., "BTCUSDT") |
| `interval` | string | Yes | Candle interval (see supported intervals) |
| `start_time` | int64 | Yes | Start timestamp in Unix seconds |
| `end_time` | int64 | Yes | End timestamp in Unix seconds |
| `features` | array[string] | No | List of features to calculate |
| `indicators` | array[string] | No | List of indicators to calculate |

## Supported Time Intervals

| Interval | Description | Example |
|----------|-------------|---------|
| `1m` | 1 minute | High-frequency trading |
| `3m` | 3 minutes | Short-term analysis |
| `5m` | 5 minutes | Scalping |
| `15m` | 15 minutes | Day trading |
| `30m` | 30 minutes | Intraday |
| `1h` | 1 hour | Swing trading |
| `4h` | 4 hours | Position trading |
| `1d` | 1 day | Long-term analysis |
| `1w` | 1 week | Weekly analysis |

## Features

Features are calculated from price data and return a single value per candle.

### Returns

**Name**: `returns`

**Description**: Simple percentage return from previous close

**Formula**: `(ClosePrice - PreviousClosePrice) / PreviousClosePrice * 100`

**Minimum Candles**: 2

**Example Request**:
```json
{
  "request_id": "req-001",
  "symbol": "BTCUSDT",
  "interval": "1m",
  "start_time": 1713590400,
  "end_time": 1713591000,
  "features": ["returns"]
}
```

### Moving Averages

**Available Periods**:
- `ma_7` - 7-period Moving Average
- `ma_21` - 21-period Moving Average
- `ma_50` - 50-period Moving Average

**Description**: Simple Moving Average (SMA) of closing prices

**Formula**: `Average of last N closing prices`

**Minimum Candles**: N (period)

**Example Request**:
```json
{
  "request_id": "req-002",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start_time": 1713590400,
  "end_time": 1713676800,
  "features": ["ma_7", "ma_21", "ma_50"]
}
```

### Volatility

**Available Periods**:
- `volatility_7` - 7-period Volatility
- `volatility_21` - 21-period Volatility

**Description**: Standard deviation of returns over the period

**Formula**: `Standard Deviation of returns over last N periods`

**Minimum Candles**: N (period)

**Example Request**:
```json
{
  "request_id": "req-003",
  "symbol": "ETHUSDT",
  "interval": "15m",
  "start_time": 1713590400,
  "end_time": 1713633600,
  "features": ["volatility_7", "volatility_21"]
}
```

### EWMA (Exponentially Weighted Moving Average)

**Name**: `ewma_30d`

**Description**: 30-day Exponentially Weighted Moving Average

**Formula**: `EMA with exponential decay over 30 days`

**Minimum Candles**: 30

**Example Request**:
```json
{
  "request_id": "req-004",
  "symbol": "BTCUSDT",
  "interval": "1d",
  "start_time": 1712016000,
  "end_time": 1714694400,
  "features": ["ewma_30d"]
}
```

## Indicators

Indicators are technical analysis indicators that may require multiple calculations.

### RSI (Relative Strength Index)

**Name**: `rsi_14`

**Description**: 14-period Relative Strength Index

**Formula**: `RSI = 100 - (100 / (1 + RS))` where `RS = Average Gain / Average Loss`

**Range**: 0 to 100
- RSI > 70: Overbought
- RSI < 30: Oversold

**Minimum Candles**: 15 (period + 1)

**Example Request**:
```json
{
  "request_id": "req-005",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start_time": 1713590400,
  "end_time": 1713676800,
  "indicators": ["rsi_14"]
}
```

### MACD (Moving Average Convergence Divergence)

**Available Components**:
- `macd` - MACD line (12-period EMA - 26-period EMA)
- `macd_signal` - Signal line (9-period EMA of MACD)

**Parameters**:
- Fast Period: 12
- Slow Period: 26
- Signal Period: 9

**Minimum Candles**: 35 (slow + signal - 1)

**Example Request**:
```json
{
  "request_id": "req-006",
  "symbol": "ETHUSDT",
  "interval": "4h",
  "start_time": 1713590400,
  "end_time": 1713763200,
  "indicators": ["macd", "macd_signal"]
}
```

### Bollinger Bands

**Available Components**:
- `bb_upper` - Upper Band (MA + 2 * StdDev)
- `bb_lower` - Lower Band (MA - 2 * StdDev)
- `bb_width` - Band Width (Upper - Lower)

**Parameters**:
- Period: 20
- Standard Deviations: 2

**Minimum Candles**: 20

**Example Request**:
```json
{
  "request_id": "req-007",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start_time": 1713590400,
  "end_time": 1713676800,
  "indicators": ["bb_upper", "bb_lower", "bb_width"]
}
```

### Sharpe Ratio

**Name**: `share`

**Description**: Risk-adjusted return ratio

**Formula**: `(Mean Return - RiskFreeRate) / StdDev(Returns)`

**Parameters**:
- Period: Configurable (default 20)

**Minimum Candles**: Period + 1

**Example Request**:
```json
{
  "request_id": "req-008",
  "symbol": "BTCUSDT",
  "interval": "1d",
  "start_time": 1712016000,
  "end_time": 1714694400,
  "indicators": ["sharpe"]
}
```

### Calmar Ratio

**Name**: `calmar`

**Description**: Return vs Maximum Drawdown ratio

**Formula**: `Annual Return / Maximum Drawdown`

**Parameters**:
- Period: Configurable (default 20)

**Minimum Candles**: Period + 1

**Example Request**:
```json
{
  "request_id": "req-009",
  "symbol": "ETHUSDT",
  "interval": "1d",
  "start_time": 1712016000,
  "end_time": 1714694400,
  "indicators": ["calmar"]
}
```

### CVaR (Conditional Value at Risk)

**Name**: `cvar_95`

**Description**: 95th percentile Value at Risk

**Formula**: `Average of worst 5% returns`

**Minimum Candles**: 20

**Example Request**:
```json
{
  "request_id": "req-010",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start_time": 1713590400,
  "end_time": 1713676800,
  "indicators": ["cvar_95"]
}
```

## Response Format

### Success Response

**Routing Key**: `calculation.completed.v1`

```json
{
  "request_id": "req-001",
  "symbol": "BTCUSDT",
  "interval": "1m",
  "start_time": 1234567890,
  "end_time": 1234567950,
  "features": [
    {
      "name": "returns",
      "values": [
        {
          "timestamp": 1234567890,
          "value": 0.5
        },
        {
          "timestamp": 1234567950,
          "value": -0.3
        }
      ]
    }
  ],
  "indicators": [
    {
      "name": "rsi_14",
      "values": [
        {
          "timestamp": 1234567890,
          "value": 65.5
        }
      ]
    }
  ],
  "calculated_at": 1234567950
}
```

### Error Response

**Routing Key**: `calculation.failed.v1`

```json
{
  "request_id": "req-001",
  "error": "Not enough candles for calculation",
  "error_code": "INSUFFICIENT_DATA",
  "timestamp": 1234567950
}
```

## Complete Example

### Request: Multiple Features and Indicators

```json
{
  "request_id": "req-complete-001",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start_time": 1713590400,
  "end_time": 1713676800,
  "features": ["returns", "ma_21", "volatility_7"],
  "indicators": ["rsi_14", "macd", "bb_upper", "bb_lower"]
}
```

### Expected Response

```json
{
  "request_id": "req-complete-001",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start_time": 1713590400,
  "end_time": 1713676800,
  "features": [
    {
      "name": "returns",
      "values": [
        {
          "timestamp": 1713590400,
          "value": 0.0
        },
        {
          "timestamp": 1713594000,
          "value": 0.25
        }
      ]
    },
    {
      "name": "ma_21",
      "values": [
        {
          "timestamp": 1713673200,
          "value": 43250.5
        }
      ]
    },
    {
      "name": "volatility_7",
      "values": [
        {
          "timestamp": 1713666000,
          "value": 1.25
        }
      ]
    }
  ],
  "indicators": [
    {
      "name": "rsi_14",
      "values": [
        {
          "timestamp": 1713676800,
          "value": 58.5
        }
      ]
    },
    {
      "name": "macd",
      "values": [
        {
          "timestamp": 1713676800,
          "value": 125.5
        }
      ]
    },
    {
      "name": "bb_upper",
      "values": [
        {
          "timestamp": 1713676800,
          "value": 44500.0
        }
      ]
    },
    {
      "name": "bb_lower",
      "values": [
        {
          "timestamp": 1713676800,
          "value": 42000.0
        }
      ]
    }
  ],
  "calculated_at": 1713676805
}
```

## Error Codes

| Error Code | Description |
|------------|-------------|
| `INSUFFICIENT_DATA` | Not enough candles for the requested calculation |
| `INVALID_SYMBOL` | Symbol not found or invalid |
| `INVALID_INTERVAL` | Interval not supported |
| `INVALID_TIME_RANGE` | Time range is invalid or in the future |
| `CALCULATION_ERROR` | General calculation error |
| `TIMEOUT` | Calculation timed out |

## Performance

- **Worker Pool Size**: 5 (configurable via `CALCULATOR_WORKER_POOL_SIZE`)
- **Prefetch Count**: 50 (configurable via `CALCULATOR_PREFETCH_COUNT`)
- **Max Concurrent Calculations**: 10 (configurable via `CALCULATOR_MAX_CONCURRENT`)
- **Calculation Timeout**: 30s (configurable via `CALCULATOR_TIMEOUT_SECONDS`)
- **Cache TTL**: 24 hours

## Caching

Results are cached in Redis for 24 hours to improve performance. The cache key includes:
- Symbol
- Interval
- Start Time
- End Time

Frequent calculations are also persisted in TimescaleDB for faster retrieval.

## Health Checks

The Calculator Service provides health checks on the configured port (default 8081):

- `GET /health/live` - Liveness check
- `GET /health/ready` - Readiness check (includes database, cache, and messaging)

## Metrics

Prometheus metrics are available for monitoring:

- `calculation_duration_seconds` - Calculation duration by type
- `calculation_errors_total` - Total calculation errors by type
- `queue_depth` - Current queue depth by queue name
