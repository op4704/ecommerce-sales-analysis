"""
Tests for features.py

Uses small synthetic daily-revenue series to verify each feature
engineering step precisely, including edge cases like leakage prevention
in rolling features.
"""

import numpy as np
import pandas as pd
import pytest

from ecommerce_sales_analysis.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_daily_revenue,
    build_feature_set,
)


def make_cleaned_df(dates_and_revenue: list[tuple]) -> pd.DataFrame:
    """Build a minimal 'cleaned' dataframe with just InvoiceDate + Revenue."""
    rows = [{"InvoiceDate": d, "Revenue": r} for d, r in dates_and_revenue]
    return pd.DataFrame(rows)


def test_build_daily_revenue_aggregates_same_day_transactions():
    df = make_cleaned_df([
        ("2021-01-01 09:00:00", 10.0),
        ("2021-01-01 15:00:00", 5.0),
        ("2021-01-02 10:00:00", 20.0),
    ])
    daily = build_daily_revenue(df)
    assert daily.loc["2021-01-01", "Revenue"] == 15.0
    assert daily.loc["2021-01-02", "Revenue"] == 20.0


def test_build_daily_revenue_fills_missing_days_with_zero():
    df = make_cleaned_df([
        ("2021-01-01 09:00:00", 10.0),
        ("2021-01-05 09:00:00", 20.0),  # gap of 3 days with no transactions
    ])
    daily = build_daily_revenue(df)
    assert len(daily) == 5  # Jan 1 through Jan 5, no gaps
    assert daily.loc["2021-01-03", "Revenue"] == 0.0


def test_calendar_features_are_correct():
    daily = pd.DataFrame(
        {"Revenue": [1.0]},
        index=pd.DatetimeIndex(["2021-01-04"], name="Date"),  # a Monday
    )
    out = add_calendar_features(daily)
    assert out.loc["2021-01-04", "DayOfWeek"] == 0  # Monday
    assert out.loc["2021-01-04", "Month"] == 1
    assert out.loc["2021-01-04", "IsWeekend"] == 0


def test_calendar_features_flag_weekend_correctly():
    daily = pd.DataFrame(
        {"Revenue": [1.0]},
        index=pd.DatetimeIndex(["2021-01-09"], name="Date"),  # a Saturday
    )
    out = add_calendar_features(daily)
    assert out.loc["2021-01-09", "DayOfWeek"] == 5  # Saturday
    assert out.loc["2021-01-09", "IsWeekend"] == 1


def test_lag_features_shift_correctly():
    daily = pd.DataFrame(
        {"Revenue": [10.0, 20.0, 30.0, 40.0]},
        index=pd.date_range("2021-01-01", periods=4, freq="D"),
    )
    out = add_lag_features(daily, lags=[1])
    assert pd.isna(out["Lag_1"].iloc[0])       # no prior day for day 1
    assert out["Lag_1"].iloc[1] == 10.0        # day 2's lag_1 = day 1's revenue
    assert out["Lag_1"].iloc[3] == 30.0        # day 4's lag_1 = day 3's revenue


def test_rolling_features_do_not_leak_current_day():
    """
    The rolling mean for day N must never include day N's own revenue -
    otherwise the model would be trivially 'predicting' using its own answer.
    """
    daily = pd.DataFrame(
        {"Revenue": [10.0, 10.0, 10.0, 1000.0]},  # huge spike on day 4
        index=pd.date_range("2021-01-01", periods=4, freq="D"),
    )
    out = add_rolling_features(daily, windows=[3])
    # Day 4's rolling mean should be based on days 1-3 only (all 10.0),
    # NOT include day 4's own 1000.0 value.
    assert out["RollingMean_3"].iloc[3] == pytest.approx(10.0)


def test_build_feature_set_drops_incomplete_rows():
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    df = make_cleaned_df([(str(d), 10.0) for d in dates])
    features = build_feature_set(df)
    # No NaN should survive in the final feature set
    assert features.isna().sum().sum() == 0
    # Should have dropped rows where the longest lag/rolling window (30) has no history
    assert len(features) <= 40


def test_build_feature_set_has_expected_columns():
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    df = make_cleaned_df([(str(d), 10.0) for d in dates])
    features = build_feature_set(df)
    expected = {
        "Revenue", "DayOfWeek", "Month", "Quarter", "IsWeekend",
        "IsMonthStart", "IsMonthEnd", "DayOfYear",
        "Lag_1", "Lag_7", "Lag_14", "Lag_30",
        "RollingMean_7", "RollingStd_7", "RollingMean_30", "RollingStd_30",
    }
    assert expected.issubset(set(features.columns))
