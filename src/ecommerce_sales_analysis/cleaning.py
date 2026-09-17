"""Cleaning logic for the raw Online Retail II transaction data."""

import pandas as pd


def clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Online Retail II dataframe.

    Cleaning decisions (and why):
    ------------------------------
    1. Drop rows with missing 'Customer ID' -> we can't link them to a
       customer, and for sales forecasting we care about aggregate revenue,
       but for any customer-level features we need a valid ID. We keep the
       decision explicit rather than silently ignoring these rows.
    2. Remove cancelled invoices (InvoiceNo starting with 'C') -> these
       represent order cancellations, not completed sales; including them
       would double-count or invert revenue.
    3. Remove non-positive Quantity or Price -> a completed sale must have
       a positive quantity and a positive price. Zero/negative values here
       are returns, data entry errors, or adjustment entries, not real sales.
    4. Drop exact duplicate rows -> duplicate transaction lines inflate
       revenue and unit counts.
    5. Add a computed 'Revenue' column (Quantity * Price) -> this is our
       core sales metric used throughout EDA and forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by `data_loader.load_raw_data()`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, ready for EDA and feature engineering.
    """
    cleaned = df.copy()

    # Normalize column names in case of whitespace/casing differences
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # 1. Drop rows with missing Customer ID
    customer_col = "Customer ID" if "Customer ID" in cleaned.columns else "CustomerID"
    cleaned = cleaned.dropna(subset=[customer_col])

    # 2. Remove cancelled invoices (InvoiceNo starting with 'C')
    cleaned["InvoiceNo"] = cleaned["Invoice"].astype(str) if "Invoice" in cleaned.columns else cleaned["InvoiceNo"].astype(str)
    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C")]

    # 3. Remove non-positive Quantity or Price
    price_col = "Price" if "Price" in cleaned.columns else "UnitPrice"
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned[price_col] > 0)]

    # 4. Drop exact duplicate rows
    cleaned = cleaned.drop_duplicates()

    # 5. Parse dates and compute Revenue
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["Revenue"] = cleaned["Quantity"] * cleaned[price_col]

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def cleaning_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Return a small before/after summary, useful for the final report."""
    return {
        "raw_rows": len(raw),
        "cleaned_rows": len(cleaned),
        "rows_removed": len(raw) - len(cleaned),
        "pct_removed": round((len(raw) - len(cleaned)) / len(raw) * 100, 2),
    }
