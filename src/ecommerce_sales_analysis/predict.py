"""
Prediction utilities for the E-Commerce Sales Analysis & Prediction System.

Why a separate predict module?
---------------------------------
Forecasting *future* days is fundamentally different from evaluating a model
on a held-out test set. In evaluation (Notebook 05/06), lag and rolling
features for the test period could be computed directly from real historical
data, because that data already existed. For real future predictions, days
2, 3, 4... don't have real "yesterday" values yet - each new prediction has
to become part of the history used to predict the day after it. This module
implements that iterative (recursive) forecasting loop explicitly, instead
of leaving it as an easy-to-get-wrong ad-hoc notebook loop.
"""

import pandas as pd

from ecommerce_sales_analysis.features import add_calendar_features, add_lag_features, add_rolling_features

LAGS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 30]
WARM_UP_DAYS = max(max(LAGS), max(ROLLING_WINDOWS))


def forecast_future(model, daily_history: pd.DataFrame, n_days: int) -> pd.Series:
    """
    Forecast revenue for the next `n_days` beyond the end of `daily_history`,
    one day at a time, feeding each prediction back in as history for the next.

    Why recursive/iterative forecasting?
    ---------------------------------------
    To predict day T+2, the model needs Lag_1 (day T+1's revenue) as a
    feature - but day T+1's revenue doesn't exist yet until *we* predict it.
    So we predict one day, treat that prediction as if it were the real
    value, recompute lag/rolling features including it, then predict the
    next day. This is standard practice for multi-step time series
    forecasting with lag-based features, and it means errors can compound
    over longer horizons - a real limitation we should be upfront about
    (see the warning printed by this function for horizons beyond 7 days).

    Parameters
    ----------
    model : a fitted scikit-learn regressor
        Must have been trained on features shaped like
        `features.build_feature_set()` output.
    daily_history : pd.DataFrame
        Daily revenue history with a 'Revenue' column and a DatetimeIndex,
        e.g. `features.build_daily_revenue()` output. Must have at least
        `WARM_UP_DAYS` days of history so lag/rolling features are valid
        from day 1 of the forecast.
    n_days : int
        Number of future days to forecast.

    Returns
    -------
    pd.Series
        Forecasted revenue, indexed by the new future dates.
    """
    if len(daily_history) < WARM_UP_DAYS:
        raise ValueError(
            f"Need at least {WARM_UP_DAYS} days of history to compute lag/rolling "
            f"features, got {len(daily_history)}."
        )

    history = daily_history[["Revenue"]].copy()
    last_date = history.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days, freq="D")

    predictions = []
    for date in future_dates:
        # Append a placeholder row for "today" so calendar features can be
        # computed for it; its Revenue value is unused (lag/rolling features
        # only ever look at *past* rows via shift()).
        history.loc[date] = 0.0

        featured = add_calendar_features(history)
        featured = add_lag_features(featured, lags=LAGS)
        featured = add_rolling_features(featured, windows=ROLLING_WINDOWS)

        feature_cols = [c for c in featured.columns if c != "Revenue"]
        today_features = featured.loc[[date], feature_cols]

        pred = model.predict(today_features)[0]
        pred = max(pred, 0.0)  # revenue can't be negative

        history.loc[date, "Revenue"] = pred
        predictions.append(pred)

    if n_days > 7:
        print(
            f"Note: forecasting {n_days} days ahead using recursive lag features. "
            "Accuracy typically degrades the further out you predict, since each "
            "day's forecast is partly built on the previous day's forecast rather "
            "than real data. Treat predictions beyond ~7 days as directional, not precise."
        )

    return pd.Series(predictions, index=future_dates, name="PredictedRevenue")
