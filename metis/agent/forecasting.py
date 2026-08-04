"""Deterministic helper logic for Apollo forecasting integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PredictionWindow:
    start_date: str
    end_date: str


@dataclass
class ForecastQuality:
    actionable: bool
    warnings: list[str] = field(default_factory=list)
    direction_bias: str = "neutral"


def build_prediction_window(
    *,
    reference_time: datetime | None = None,
    lookback_days: int = 90,
) -> PredictionWindow:
    """Build prediction window: ontem até N dias atrás (inclusivo).

    A janela cobre `lookback_days` dias completos terminando em `end_day` (último
    dia completo, i.e., ontem). Por exemplo, com `lookback_days=115` e
    referência em `2026-04-26`, a janela é `[2026-01-01, 2026-04-25]` — exatamente
    115 dias inclusivos.
    """
    now = reference_time.astimezone(UTC) if reference_time else datetime.now(UTC)
    end_day = (now - timedelta(days=1)).date()
    # `lookback_days - 1` porque o intervalo é inclusivo em ambos os extremos
    # (e.g., 115 dias = de end_day até end_day - 114 dias).
    start_day = end_day - timedelta(days=lookback_days - 1)

    start_dt = datetime(start_day.year, start_day.month, start_day.day, 0, 0, 0)
    end_dt = datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59)

    window = PredictionWindow(
        start_date=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        end_date=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    print(f"[WINDOW] {window.start_date} até {window.end_date} ({lookback_days}d)", flush=True)
    return window


def calculate_error_pct(current_price: float | None, predicted_price: float | None) -> float | None:
    """Return absolute percentage error."""
    if current_price is None or predicted_price is None or current_price <= 0:
        return None
    return abs(predicted_price - current_price) / current_price * 100.0


def calculate_return_pct(current_price: float | None, predicted_price: float | None) -> float | None:
    """Return forecasted percentage move."""
    if current_price is None or predicted_price is None or current_price <= 0:
        return None
    return (predicted_price - current_price) / current_price * 100.0


def assess_forecast_quality(
    *,
    confidence: float | None,
    model_mape: float | None,
    data_quality: str | None,
    data_points: int | None,
    predicted_return_pct: float | None,
    confidence_threshold: float,
    mape_threshold: float,
) -> ForecastQuality:
    """Classify whether the forecast is safe enough to influence the agent."""
    warnings: list[str] = []

    if data_points is None or data_points < 60:
        warnings.append(f"forecast com poucas observações ({data_points or 0})")
    if confidence is None or confidence < confidence_threshold:
        warnings.append(
            f"forecast com confiança baixa ({(confidence or 0.0):.1%} < {confidence_threshold:.0%})"
        )
    if model_mape is None or model_mape > mape_threshold:
        warnings.append(
            f"forecast com MAPE acima do limite ({model_mape if model_mape is not None else 'n/d'} > {mape_threshold:.2f}%)"
        )
    if data_quality and data_quality.lower() != "good":
        warnings.append(f"qualidade de dados do forecast = {data_quality}")
    if predicted_return_pct is None:
        warnings.append("forecast sem variação prevista utilizável")

    direction_bias = "neutral"
    if predicted_return_pct is not None:
        if predicted_return_pct > 0.5:
            direction_bias = "bullish"
        elif predicted_return_pct < -0.5:
            direction_bias = "bearish"

    # Forecast é acionável se:
    # 1. Tem uma direção clara (bullish/bearish), OU
    # 2. Passa nos thresholds principais (confiança e MAPE), MESMO COM outros warnings
    has_confidence_warning = any("confiança baixa" in w for w in warnings)
    has_mape_warning = any("MAPE acima do limite" in w for w in warnings)
    has_critical_warning = has_confidence_warning or has_mape_warning

    is_actionable = (
        direction_bias != "neutral" and not has_critical_warning
    ) or (
        # Alternativa: aceita sem avisos críticos, mesmo se direction_bias neutral
        not has_critical_warning and len([w for w in warnings if "confiança" in w or "MAPE" in w]) == 0
    )

    return ForecastQuality(
        actionable=is_actionable,
        warnings=warnings,
        direction_bias=direction_bias,
    )


def overlay_forecast_on_signal(
    *,
    signal: str,
    confidence: float,
    forecast_quality: ForecastQuality,
    predicted_return_pct: float | None,
) -> tuple[str, float, str]:
    """Apply a conservative forecast overlay to an existing trading signal."""
    if not forecast_quality.actionable or predicted_return_pct is None:
        reason = "; ".join(forecast_quality.warnings) if forecast_quality.warnings else "forecast ignorado"
        return signal, confidence, reason

    is_long_signal = signal in {"long", "conditional_long"}
    is_short_signal = signal in {"short", "conditional_short"}
    reason = "forecast alinhado"

    if is_long_signal and forecast_quality.direction_bias == "bearish":
        downgraded = "conditional_long" if signal == "long" else "wait"
        return downgraded, max(confidence * 0.75, 0.35), (
            f"forecast contradiz viés comprador ({predicted_return_pct:.2f}%)"
        )

    if is_short_signal and forecast_quality.direction_bias == "bullish":
        downgraded = "conditional_short" if signal == "short" else "wait"
        return downgraded, max(confidence * 0.75, 0.35), (
            f"forecast contradiz viés vendedor ({predicted_return_pct:.2f}%)"
        )

    if ((is_long_signal and forecast_quality.direction_bias == "bullish")
            or (is_short_signal and forecast_quality.direction_bias == "bearish")):
        boosted = min(confidence + 0.05, 0.95)
        return signal, boosted, f"forecast confirma direção ({predicted_return_pct:.2f}%)"

    return signal, confidence, reason
