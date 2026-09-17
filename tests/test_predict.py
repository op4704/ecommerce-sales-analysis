"""
Tests for predict.py

Uses a simple, deterministic fake "model" (predicts a fixed multiple of
Lag_1) so we can verify the recursive forecasting loop's mechanics -
correct future dates, no leakage, non-negative predictions - without
depending on a real trained scikit-learn model.
"""

import numpy as np
import pandas as pd
import pytest

from ecommerce_sales_analysis.predict import WARM_UP_DAYS, forecast_future


class FakeModel:
    """A deterministic stand-in model: predicts 1.0 * Lag_1 (i.e. 'repeat yesterday')."""

    def predict(self, X: pd.DataFrame):
        return X["Lag_1"].to_numpy()


class NegativeFakeModel:
    """A model that always predicts a negative number, to test clamping."""

    def predict(self, X: pd.DataFrame):
        return np.full(len(X), -50.0)


def make_history(n_days: int = WARM_UP_DAYS + 5, value: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"Revenue": [value] * n_days}, index=dates)


def test_forecast_future_returns_correct_number_of_days():
    history = make_history()
    forecast = forecast_future(FakeModel(), history, n_days=5)
    assert len(forecast) == 5


def test_forecast_future_uses_dates_immediately_after_history():
    history = make_history()
    last_history_date = history.index.max()
    forecast = forecast_future(FakeModel(), history, n_days=3)
    assert forecast.index.min() == last_history_date + pd.Timedelta(days=1)
    assert forecast.index.is_monotonic_increasing


def test_forecast_future_raises_on_insufficient_history():
    short_history = make_history(n_days=5)  # less than WARM_UP_DAYS
    with pytest.raises(ValueError):
        forecast_future(FakeModel(), short_history, n_days=3)


def test_forecast_future_clamps_negative_predictions_to_zero():
    history = make_history()
    forecast = forecast_future(NegativeFakeModel(), history, n_days=3)
    assert (forecast >= 0).all()


def test_forecast_future_repeat_model_stays_constant():
    """
    With a model that just echoes Lag_1, and constant historical revenue,
    every forecasted day should equal the constant value (sanity check that
    the recursive loop correctly feeds predictions back in as history).
    """
    history = make_history(value=100.0)
    forecast = forecast_future(FakeModel(), history, n_days=5)
    assert np.allclose(forecast.to_numpy(), 100.0)
