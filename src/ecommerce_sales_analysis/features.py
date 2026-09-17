"""
Feature engineering utilities for the E-Commerce Sales Analysis & Prediction System.

Why a separate features module?
---------------------------------
Forecasting models can't use raw transaction rows directly - a "sale" is an
event, not a time series. We first have to aggregate transactions into a
regular time series (daily total revenue), then derive the actual model
inputs from that series: calendar signals, recent history (lags), and
smoothed trend (rolling averages).

Doing this in one documented module (instead of ad-hoc notebook cells) means
the exact same feature logic applies whether we're training or predicting.
"""

import pandas as pd


def build_daily_revenue(cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cleaned transaction-level data into a daily revenue time series.

    Why daily aggregation?
    -----------------------
    Individual transactions are noisy and include a handful of huge wholesale
    orders (see Notebook 03 EDA - revenue per line has a long right tail).
    Aggregating to daily totals smooths this out and gives us a regular time
    series a forecasting model can actually learn from. We also fill any
    missing calendar days with 0 revenue (no valid business happened) so the
    series has no gaps - forecasting models require a continuous index.

    Parameters
    ----------
    cleaned : pd.DataFrame
        Output of `cleaning.clean_online_retail()`. Must have 'InvoiceDate'
        and 'Revenue' columns.

    Returns
    -------
    pd.DataFrame
        Indexed by date (daily frequency, no gaps), with a single 'Revenue' column.
    """
    daily = (
        cleaned.assign(Date=pd.to_datetime(cleaned["InvoiceDate"]).dt.floor("D"))
        .groupby("Date")["Revenue"]
        .sum()
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "Date"
    return daily.to_frame(name="Revenue")


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features derived purely from the date index.

    Why these features?
    ---------------------
    Notebook 03 EDA showed clear month-of-year seasonality (this business
    sells gift-ware - Nov/Dec surge) and near-zero Saturday activity. Giving
    the model explicit calendar signals lets it learn these patterns instead
    of trying to infer them purely from lagged values.

    Parameters
    ----------
    daily : pd.DataFrame
        Output of `build_daily_revenue()`, indexed by date.

    Returns
    -------
    pd.DataFrame
        Same data with added columns: DayOfWeek, Month, Quarter, IsWeekend,
        IsMonthStart, IsMonthEnd, DayOfYear.
    """
    out = daily.copy()
    idx = out.index
    out["DayOfWeek"] = idx.dayofweek          # 0=Monday ... 6=Sunday
    out["Month"] = idx.month
    out["Quarter"] = idx.quarter
    out["IsWeekend"] = (idx.dayofweek >= 5).astype(int)
    out["IsMonthStart"] = idx.is_month_start.astype(int)
    out["IsMonthEnd"] = idx.is_month_end.astype(int)
    out["DayOfYear"] = idx.dayofyear
    return out


def add_lag_features(daily: pd.DataFrame, lags: list[int] = [1, 7, 14, 30]) -> pd.DataFrame:
    """
    Add lagged revenue features: "what was revenue N days ago?"

    Why lags?
    ----------
    Sales today are correlated with sales recently - lag_1 captures
    day-to-day momentum, lag_7 captures "same day last week" (accounting for
    weekly patterns), lag_30 captures "same point last month". These are
    some of the most predictive features in any sales forecasting problem.

    Parameters
    ----------
    daily : pd.DataFrame
        Must have a 'Revenue' column.
    lags : list[int]
        Which lag periods (in days) to create.

    Returns
    -------
    pd.DataFrame
        Same data with added columns Lag_{n} for each n in lags. The first
        `max(lags)` rows will have NaN in the longest lag columns - this is
        expected and handled by dropping incomplete rows before training.
    """
    out = daily.copy()
    for lag in lags:
        out[f"Lag_{lag}"] = out["Revenue"].shift(lag)
    return out


def add_rolling_features(daily: pd.DataFrame, windows: list[int] = [7, 30]) -> pd.DataFrame:
    """
    Add rolling average and rolling std features over past windows.

    Why rolling features?
    -----------------------
    A single lag value (e.g. yesterday) can be a noisy outlier. A rolling
    mean smooths recent history into a trend signal ("how has revenue been
    trending over the past week/month?"), and rolling std captures recent
    volatility. We shift by 1 day first so the window only ever looks at
    *past* data relative to the row being predicted - never leak the current
    day's own value into its own features.

    Parameters
    ----------
    daily : pd.DataFrame
        Must have a 'Revenue' column.
    windows : list[int]
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame
        Same data with added columns RollingMean_{n} and RollingStd_{n}.
    """
    out = daily.copy()
    shifted = out["Revenue"].shift(1)  # never include the current day itself
    for window in windows:
        out[f"RollingMean_{window}"] = shifted.rolling(window=window).mean()
        out[f"RollingStd_{window}"] = shifted.rolling(window=window).std()
    return out


def build_feature_set(cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline: aggregate -> calendar -> lags -> rolling.

    Rows with any NaN (from the longest lag/rolling window at the start of
    the series) are dropped, since a model needs complete rows to train on.

    Parameters
    ----------
    cleaned : pd.DataFrame
        Output of `cleaning.clean_online_retail()`.

    Returns
    -------
    pd.DataFrame
        Model-ready feature table indexed by date, target column 'Revenue',
        all other columns are features.
    """
    daily = build_daily_revenue(cleaned)
    daily = add_calendar_features(daily)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    return daily.dropna()
