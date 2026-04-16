# API Documentation

## Base URL
```
http://localhost:8080
```

## Authentication

### API Key Authentication
By default, all protected endpoints require API key authentication via header:
```
X-API-Key: dev_api_key_12345
```


### 6. Get Redis History
**GET /api/v1/history/:symbol**

Get historical candle data from Redis cache within a time range.

**Parameters:**
- `symbol` (path): Trading pair symbol
- `interval` (query, optional): Time interval (default: 1m; supports: 1m, 5m, 15m, 1h, 1d)
- `from` (query, required): Start timestamp in milliseconds
- `to` (query, required): End timestamp in milliseconds

**Response:**
```json
[
  {
    "t": 1776345600000,
    "et": 1776345659000,
    "o": "74924.64",
    "h": "74942.31",
    "l": "74908.16",
    "c": "74914.6",
    "v": "23.58566",
    "x": true
  }
]
```

**Example:**
```bash
curl -H "X-API-Key: dev_api_key_12345" "http://localhost:8080/api/v1/history/BTCUSDT?interval=1m&from=1776342000000&to=1776345000000"
```

---

### 7. Get Metrics
**GET /api/v1/metrics**

Get system metrics including ingestion counts, parse errors, shard status, queue depth, and DLQ messages.

**Response:**
```json
{
  "ingestion_counts": {
    "BTCUSDT": 1000,
    "ETHUSDT": 800
  },
  "parse_errors": {
    "BTCUSDT": 2
  },
  "shard_status": {
    "total_shards": 3,
    "shards": [
      {"id": 0, "status": "connected"},
      {"id": 1, "status": "connected"},
      {"id": 2, "status": "connected"}
    ]
  },
  "queue_depth": {
    "q.features.compute": 1500
  },
  "dlq_messages": 5
}
```

**Example:**
```bash
curl -H "X-API-Key: dev_api_key_12345" "http://localhost:8080/api/v1/metrics"
```

---

### 8. Get Features (All)
**GET /api/v1/features/:symbol**

Get feature values for a symbol. When called without a specific feature name, returns all features for the symbol.

**Parameters:**
- `symbol` (path): Trading pair symbol
- `interval` (query, optional): Time interval (default: 1m; supports: 1m, 5m, 15m, 1h, 1d)
- `limit` (query, optional): Number of records to return (default: 100)
- `from` (query, optional): Start timestamp in milliseconds
- `to` (query, optional): End timestamp in milliseconds

**Auto-recalculation:** If `from` and `to` are provided and no data exists in the requested window, the system will automatically recalculate features for that time range and persist the results.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "feature_name": "",
  "count": 0,
  "data": null
}
```

**Example:**
```bash
# Get latest features
curl -H "X-API-Key: dev_api_key_12345" "http://localhost:8080/api/v1/features/BTCUSDT?interval=1m&limit=100"

# Get features in a time range (with auto-recalculation if needed)
curl -H "X-API-Key: dev_api_key_12345" "http://localhost:8080/api/v1/features/BTCUSDT?interval=1m&from=1776342000000&to=1776345000000"
```

---

### 9. Recalculate Features
**POST /api/v1/features/:symbol/recalculate**

Recalculate and persist features for a specific symbol and time range.

**Parameters:**
- `symbol` (path): Trading pair symbol
- `interval` (query, optional): Time interval (default: 1m; supports: 1m, 5m, 15m, 1h, 1d)
- `from` (query, required): Start timestamp in milliseconds
- `to` (query, required): End timestamp in milliseconds
- `param` (query, optional): Comma-separated list of feature names to recalculate (e.g., "sharpe,ema_return_60"). If not provided, all features are recalculated.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "candles_processed": 51,
  "features_count": 32,
  "features": []
}
```

**Example:**
```bash
# Recalculate all features
curl -H "X-API-Key: dev_api_key_12345" \
  -X POST "http://localhost:8080/api/v1/features/BTCUSDT/recalculate?interval=1m&from=1776342000000&to=1776345000000"

# Recalculate specific features only
curl -H "X-API-Key: dev_api_key_12345" \
  -X POST "http://localhost:8080/api/v1/features/BTCUSDT/recalculate?interval=1m&from=1776342000000&to=1776345000000&param=sharpe,ema_return_60"
```

**Features Calculated:**
- Base features: ohlcv_open, ohlcv_high, ohlcv_low, ohlcv_close, ohlcv_volume, day_of_week, month
- Returns: returns_5d, returns_20d, log_returns_5d
- Moving averages: sma_5, sma_20, ewma_30
- Indicators: ema_return_60, volatility_20d, rsi_14, macd_line, macd_signal, macd_histogram
- Bollinger Bands: bb_upper, bb_middle, bb_lower
- Volume: volume_ratio
- Momentum: momentum_30d
- Encoding: encoder_60d
- Risk metrics: cvar_95, max_drawdown, sharpe, calmar
- Bootstrap: bootstrap_mean, bootstrap_std
- Target: target

---

### 10. Get Feature Values
**GET /api/v1/features/:symbol/:feature**

Get historical values for a specific feature.

**Parameters:**
- `symbol` (path): Trading pair symbol
- `feature` (path): Feature name (e.g., sharpe, ema_return_60, rsi_14)
- `interval` (query, optional): Time interval (default: 1m; supports: 1m, 5m, 15m, 1h, 1d)
- `limit` (query, optional): Number of records to return (default: 100)
- `from` (query, optional): Start timestamp in milliseconds
- `to` (query, optional): End timestamp in milliseconds

**Auto-recalculation:** If `from` and `to` are provided and no data exists in the requested window, the system will automatically recalculate the feature for that time range and persist the results.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "feature_name": "sharpe",
  "count": 5,
  "data": [
    {
      "symbol": "BTCUSDT",
      "interval": "1m",
      "timestamp": "2026-04-16T13:40:59Z",
      "feature_name": "sharpe",
      "feature_value": -12.663053188487298,
      "calculated_at": "2026-04-16T13:41:00.178119Z"
    }
  ]
}
```

**Example:**
```bash
# Get latest 5 values
curl -H "X-API-Key: dev_api_key_12345" \
  "http://localhost:8080/api/v1/features/BTCUSDT/sharpe?interval=1m&limit=5"

# Get values in a time range (with auto-recalculation if needed)
curl -H "X-API-Key: dev_api_key_12345" \
  "http://localhost:8080/api/v1/features/BTCUSDT/ema_return_60?interval=1m&from=1776342000000&to=1776345000000"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "from and to parameters are required"
}
```

### 404 Not Found
```json
{
  "error": "Price not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Failed to fetch historical data"
}
```

---

## Notes

- All timestamps are in milliseconds
- Features with insufficient data are stored as NULL
- `calculated_at` timestamp is automatically set when features are persisted
- The feature calculation pipeline supports all time windows (1m, 5m, 15m, 1h, 1d)
- `interval` is a query parameter with default value of `1m` for all endpoints
- GET feature endpoint supports auto-recalculation when data is missing in the requested time range
