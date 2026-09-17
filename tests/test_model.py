"""
Tests for model.py

Uses small synthetic feature tables to verify split logic (no time leakage),
baseline correctness, and metric computation - without needing to train real
models on the full dataset (that's covered by an integration-style smoke
test at the end).
"""

import numpy as np
import pandas as pd
import pytest

from ecommerce_sales_analysis.model import (
    baseline_predictions,
    compute_metrics,
    evaluate_models,
    time_based_split,
    train_models,
)


def make_feature_df(n_rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n_rows, freq="D")
    rng = np.random.default_rng(42)
    revenue = rng.uniform(100, 200, size=n_rows)
    return pd.DataFrame(
        {
            "Revenue": revenue,
            "Lag_1": np.roll(revenue, 1),
            "DayOfWeek": dates.dayofweek,
            "Month": dates.month,
        },
        index=dates,
    )


def test_time_based_split_preserves_chronological_order():
    df = make_feature_df(20)
    X_train, X_test, y_train, y_test = time_based_split(df, test_size=0.2)

    assert len(X_train) == 16
    assert len(X_test) == 4
    # Every training date must be strictly before every test date - this is
    # the core guarantee: no future data leaks into training.
    assert X_train.index.max() < X_test.index.min()


def test_time_based_split_excludes_target_from_features():
    df = make_feature_df(20)
    X_train, X_test, y_train, y_test = time_based_split(df)
    assert "Revenue" not in X_train.columns
    assert "Revenue" not in X_test.columns


def test_baseline_predictions_uses_lag_1():
    df = make_feature_df(10)
    preds = baseline_predictions(df)
    assert (preds == df["Lag_1"]).all()


def test_compute_metrics_returns_expected_keys():
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = compute_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE"}
    assert metrics["MAE"] == pytest.approx(10.0)


def test_compute_metrics_handles_zero_actuals_without_crashing():
    y_true = pd.Series([0.0, 0.0, 100.0])
    y_pred = np.array([5.0, 10.0, 90.0])
    metrics = compute_metrics(y_true, y_pred)
    # MAPE computed only over the non-zero actual (100.0), should not be NaN or crash
    assert metrics["MAPE"] == pytest.approx(10.0)


def test_compute_metrics_all_zero_actuals_returns_nan_mape():
    y_true = pd.Series([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    metrics = compute_metrics(y_true, y_pred)
    assert np.isnan(metrics["MAPE"])


def test_train_and_evaluate_models_smoke_test():
    """
    Integration-style smoke test: train real (tiny) models on synthetic data
    and confirm evaluate_models runs end-to-end and returns a sane result
    including the baseline.
    """
    df = make_feature_df(60)
    X_train, X_test, y_train, y_test = time_based_split(df, test_size=0.3)

    fitted = train_models(X_train, y_train)
    assert set(fitted.keys()) == {"LinearRegression", "RandomForest", "GradientBoosting"}

    results = evaluate_models(fitted, X_test, y_test)
    assert "Baseline (naive lag-1)" in results.index
    assert set(results.columns) == {"MAE", "RMSE", "MAPE"}
    # Results should be sorted by RMSE ascending (best model first)
    assert results["RMSE"].is_monotonic_increasing
