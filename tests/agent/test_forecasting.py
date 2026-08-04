from datetime import UTC, datetime

from metis.agent.forecasting import (
    assess_forecast_quality,
    build_prediction_window,
    calculate_error_pct,
    calculate_return_pct,
    overlay_forecast_on_signal,
)


def test_build_prediction_window_uses_completed_day():
    window = build_prediction_window(
        reference_time=datetime(2026, 4, 26, 15, 0, 0, tzinfo=UTC),
        lookback_days=115,
    )
    assert window.start_date == "2026-01-01T00:00:00"
    assert window.end_date == "2026-04-25T23:59:59"


def test_calculate_error_pct():
    error = calculate_error_pct(100.0, 97.5)
    assert error == 2.5


def test_calculate_return_pct():
    predicted = calculate_return_pct(100.0, 103.0)
    assert predicted == 3.0


def test_assess_forecast_quality_rejects_poor_inputs():
    quality = assess_forecast_quality(
        confidence=0.55,
        model_mape=3.0,
        data_quality="fair",
        data_points=40,
        predicted_return_pct=-1.2,
        confidence_threshold=0.60,
        mape_threshold=2.5,
    )
    assert quality.actionable is False
    assert len(quality.warnings) >= 3
    assert quality.direction_bias == "bearish"


def test_overlay_forecast_downgrades_conflicting_long_signal():
    quality = assess_forecast_quality(
        confidence=0.75,
        model_mape=1.8,
        data_quality="good",
        data_points=115,
        predicted_return_pct=-3.0,
        confidence_threshold=0.60,
        mape_threshold=2.5,
    )
    signal, confidence, reason = overlay_forecast_on_signal(
        signal="long",
        confidence=0.8,
        forecast_quality=quality,
        predicted_return_pct=-3.0,
    )
    assert signal == "conditional_long"
    assert confidence < 0.8
    assert "contradiz" in reason
