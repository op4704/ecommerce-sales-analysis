"""Model training, baseline comparison, and evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

TARGET_COL = "Revenue"


def time_based_split(features: pd.DataFrame, test_size: float = 0.2):
    """
    Split a time-indexed feature table into train/test sets WITHOUT shuffling.

    Why not a random split?
    -------------------------
    Standard `train_test_split(shuffle=True)` would let the model "see the
    future" during training - e.g. training on a random day in December and
    testing on a random day in October, when December is chronologically
    after October. That leaks future information and gives an unrealistically
    good score that would never hold up in real deployment, where we only
    ever have access to the past.

    Instead we split by date: the earliest `1 - test_size` fraction of days
    is training data, and the most recent `test_size` fraction is the test
    set - mimicking how the model will actually be used (train on history,
    predict the near future).

    Parameters
    ----------
    features : pd.DataFrame
        Output of `features.build_feature_set()`, indexed by date, sorted
        ascending, with a 'Revenue' target column.
    test_size : float
        Fraction of the most recent rows to hold out as the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    features = features.sort_index()
    split_idx = int(len(features) * (1 - test_size))

    train = features.iloc[:split_idx]
    test = features.iloc[split_idx:]

    feature_cols = [c for c in features.columns if c != TARGET_COL]
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]
    return X_train, X_test, y_train, y_test


def baseline_predictions(X: pd.DataFrame) -> pd.Series:
    """
    Naive baseline: predict today's revenue = yesterday's revenue (Lag_1).

    Why bother with a naive baseline?
    ------------------------------------
    Any "real" ML model is only useful if it beats this trivial rule. If a
    Random Forest can't outperform "just predict yesterday's number", the
    extra complexity isn't earning its keep. This is the standard sanity
    floor in every forecasting problem.

    Parameters
    ----------
    X : pd.DataFrame
        Must include a 'Lag_1' column (yesterday's revenue).

    Returns
    -------
    pd.Series
        Naive predictions, same index as X.
    """
    return X["Lag_1"]


def get_models() -> dict:
    """
    Return the set of candidate models to train and compare.

    Why these three?
    ------------------
    - LinearRegression: simplest possible real model, fast, interpretable
      coefficients - our sanity check against the baseline.
    - RandomForestRegressor: handles non-linear relationships and feature
      interactions (e.g. "December AND weekday" behaving differently than
      either alone) without needing manual interaction terms.
    - GradientBoostingRegressor: typically the strongest tabular-data
      performer of the three, learns from residual errors iteratively.

    All use fixed random_state for reproducibility - re-running training
    gives identical results, which matters for comparing runs fairly.
    """
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit every candidate model on the training set.

    Returns
    -------
    dict
        {model_name: fitted_model}
    """
    fitted = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression/forecasting error metrics.

    Why these three metrics?
    ---------------------------
    - MAE (Mean Absolute Error): average error in the same units as revenue
      (GBP) - easy to interpret ("on average we're off by £X").
    - RMSE (Root Mean Squared Error): like MAE but penalizes large errors
      more heavily - useful because a few very wrong predictions matter more
      in a business context than many small ones.
    - MAPE (Mean Absolute Percentage Error): error as a percentage of the
      actual value - lets us compare performance across days with very
      different revenue scales (a slow Tuesday vs a big pre-Christmas day).

    Returns
    -------
    dict
        {"MAE": ..., "RMSE": ..., "MAPE": ...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE is undefined for zero actuals; guard against days with 0 revenue
    # (e.g. quiet Saturdays) by only computing it over non-zero actuals.
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[nonzero_mask], np.asarray(y_pred)[nonzero_mask]) * 100
    else:
        mape = float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_models(fitted_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate every fitted model (plus the naive baseline) on the test set.

    Returns
    -------
    pd.DataFrame
        One row per model with MAE/RMSE/MAPE columns, sorted by RMSE ascending
        (best model first).
    """
    results = {}

    baseline_pred = baseline_predictions(X_test)
    results["Baseline (naive lag-1)"] = compute_metrics(y_test, baseline_pred)

    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        results[name] = compute_metrics(y_test, preds)

    return pd.DataFrame(results).T.sort_values("RMSE")
