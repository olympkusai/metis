# Feature Parameters and Time Windows

## Overview
This document describes all features, their mandatory/optional parameters, and the minimum time windows required for calculation.

## Features Configuration

### Base Features (No time window required)

| Feature Name | Parameters | Min Candles | Description |
|-------------|-----------|-------------|-------------|
| ohlcv_open | None | 1 | Opening price |
| ohlcv_high | None | 1 | Highest price |
| ohlcv_low | None | 1 | Lowest price |
| ohlcv_close | None | 1 | Closing price |
| ohlcv_volume | None | 1 | Trading volume |
| day_of_week | None | 1 | Day of week (0-6) |
| month | None | 1 | Month (1-12) |

### Returns Features

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| returns_5d | period: 5 | 6 | 5 candles | Simple return over 5 periods |
| returns_20d | period: 20 | 21 | 20 candles | Simple return over 20 periods |
| log_returns_5d | period: 5 | 6 | 5 candles | Log return over 5 periods |

### Moving Averages

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| sma_5 | period: 5 | 5 | 5 candles | Simple Moving Average (5 periods) |
| sma_20 | period: 20 | 20 | 20 candles | Simple Moving Average (20 periods) |
| ewma_30 | period: 30 | 30 | 30 candles | Exponentially Weighted Moving Average (30 periods) |

### Momentum & Trend Indicators

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| ema_return_60 | period: 60 | 61 | 60 candles | EMA of returns (60 periods) |
| momentum_30d | period: 30 | 31 | 30 candles | Price momentum (30 periods) |

### Volatility Indicators

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| volatility_20d | period: 20 | 21 | 20 candles | Volatility (std dev of returns, 20 periods) |

### Oscillators

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| rsi_14 | period: 14 | 15 | 14 candles | Relative Strength Index (14 periods) |
| macd_line | fast: 12, slow: 26, signal: 9 | 35 | 35 candles | MACD Line (12-26-9) |
| macd_signal | fast: 12, slow: 26, signal: 9 | 35 | 35 candles | MACD Signal Line (12-26-9) |
| macd_histogram | fast: 12, slow: 26, signal: 9 | 35 | 35 candles | MACD Histogram (12-26-9) |

### Bollinger Bands

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| bb_upper | period: 20, stdDev: 2 | 20 | 20 candles | Bollinger Band Upper (20, 2) |
| bb_middle | period: 20, stdDev: 2 | 20 | 20 candles | Bollinger Band Middle (SMA 20) |
| bb_lower | period: 20, stdDev: 2 | 20 | 20 candles | Bollinger Band Lower (20, 2) |

### Volume Features

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| volume_ratio | period: 20 | 20 | 20 candles | Volume ratio (current / avg 20) |

### Encoding Features

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| encoder_60d | None | 1 | N/A | Time encoding (60-day cycle) |

### Risk Metrics

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| cvar_95 | confidence: 0.95, period: 20 | 21 | 20 candles | Conditional Value at Risk (95%, 20 periods) |
| max_drawdown | None | 2 | N/A | Maximum drawdown (full history) |
| sharpe | riskFreeRate: 0.02, period: 60 | 61 | 60 candles | Sharpe ratio (2% risk-free, 60 periods) |
| calmar | None | 2 | N/A | Calmar ratio (full history) |

### Bootstrap Features

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| bootstrap_mean | period: 20 | 20 | 20 candles | Bootstrap mean (20 periods) |
| bootstrap_std | period: 20 | 20 | 20 candles | Bootstrap std dev (20 periods) |

### Target Feature

| Feature Name | Parameters | Min Candles | Time Window | Description |
|-------------|-----------|-------------|-------------|-------------|
| target | None | 2 | N/A | Target variable (next candle return) |

## Time Window Requirements by Interval

### Minimum Candles Required for All Features

To calculate ALL features simultaneously, you need the maximum of all minimum requirements:

**Maximum window required: 61 candles** (for ema_return_60 and sharpe)

### Per Interval Requirements

| Interval | 1m | 5m | 15m | 1h | 1d |
|----------|----|----|----|----|----|
| Min candles | 61 | 61 | 61 | 61 | 61 |
| Time span | 61 min | 305 min (5h 5m) | 915 min (15h 15m) | 61h | 61 days |

### Recommended Buffer

For production use, add a buffer of 10-20% to ensure stable calculations:

| Interval | Recommended Buffer | Total Candles |
|----------|-------------------|---------------|
| 1m | 10 candles | 71 candles (71 min) |
| 5m | 10 candles | 71 candles (355 min) |
| 15m | 10 candles | 71 candles (1065 min) |
| 1h | 10 candles | 71 candles (71h) |
| 1d | 10 candles | 71 candles (71 days) |

## Feature Dependencies

Some features depend on others:

1. **macd_line, macd_signal, macd_histogram** - All calculated together (MACD indicator)
2. **bb_upper, bb_middle, bb_lower** - All calculated together (Bollinger Bands)
3. **bootstrap_mean, bootstrap_std** - All calculated together (Bootstrap)
4. **sharpe, cvar_95** - Require returns calculation first
5. **ema_return_60** - Requires returns calculation first

## Implementation Notes

- Features with insufficient data return `null` in the database
- The pipeline automatically handles NULL values gracefully
- For real-time processing, ensure you have at least the minimum required candles in the buffer
- For backtesting, use the recommended buffer for more stable results
