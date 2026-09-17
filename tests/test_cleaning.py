"""
Tests for cleaning.py

These use small synthetic dataframes (not the real 1M-row dataset) so we can
construct exact edge cases and assert precisely what the cleaning pipeline
does with each one.
"""

import pandas as pd
import pytest

from ecommerce_sales_analysis.cleaning import clean_online_retail, cleaning_summary


def make_raw_df(rows: list[dict]) -> pd.DataFrame:
    """Build a raw dataframe matching the real dataset's column names."""
    base_cols = {
        "Invoice": None,
        "StockCode": "85048",
        "Description": "TEST ITEM",
        "Quantity": 1,
        "InvoiceDate": "2010-01-01 10:00:00",
        "Price": 1.0,
        "Customer ID": 12345.0,
        "Country": "United Kingdom",
    }
    full_rows = []
    for r in rows:
        row = {**base_cols, **r}
        full_rows.append(row)
    return pd.DataFrame(full_rows)


def test_drops_rows_with_missing_customer_id():
    df = make_raw_df([
        {"Invoice": "1", "Customer ID": 111.0},
        {"Invoice": "2", "Customer ID": None},
    ])
    cleaned = clean_online_retail(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["InvoiceNo"] == "1"


def test_removes_cancelled_invoices():
    df = make_raw_df([
        {"Invoice": "489434", "Customer ID": 111.0},
        {"Invoice": "C489435", "Customer ID": 111.0},
    ])
    cleaned = clean_online_retail(df)
    assert len(cleaned) == 1
    assert not cleaned["InvoiceNo"].str.startswith("C").any()


def test_removes_non_positive_quantity_and_price():
    df = make_raw_df([
        {"Invoice": "1", "Quantity": 5, "Price": 2.0},
        {"Invoice": "2", "Quantity": -3, "Price": 2.0},   # return
        {"Invoice": "3", "Quantity": 5, "Price": 0.0},    # free/adjustment
        {"Invoice": "4", "Quantity": 0, "Price": 2.0},    # zero qty
    ])
    cleaned = clean_online_retail(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["InvoiceNo"] == "1"


def test_drops_exact_duplicate_rows():
    df = make_raw_df([
        {"Invoice": "1", "Customer ID": 111.0},
        {"Invoice": "1", "Customer ID": 111.0},  # exact duplicate
    ])
    cleaned = clean_online_retail(df)
    assert len(cleaned) == 1


def test_computes_revenue_correctly():
    df = make_raw_df([
        {"Invoice": "1", "Quantity": 4, "Price": 2.5},
    ])
    cleaned = clean_online_retail(df)
    assert cleaned.iloc[0]["Revenue"] == pytest.approx(10.0)


def test_invoice_date_is_parsed_to_datetime():
    df = make_raw_df([{"Invoice": "1"}])
    cleaned = clean_online_retail(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"])


def test_cleaning_summary_reports_correct_counts():
    df = make_raw_df([
        {"Invoice": "1", "Customer ID": 111.0},
        {"Invoice": "2", "Customer ID": None},  # will be dropped
    ])
    cleaned = clean_online_retail(df)
    summary = cleaning_summary(df, cleaned)
    assert summary["raw_rows"] == 2
    assert summary["cleaned_rows"] == 1
    assert summary["rows_removed"] == 1
    assert summary["pct_removed"] == 50.0


def test_no_rows_survive_edge_case_returns_empty_df_not_error():
    df = make_raw_df([
        {"Invoice": "C1", "Customer ID": None},
    ])
    cleaned = clean_online_retail(df)
    assert len(cleaned) == 0
    assert "Revenue" in cleaned.columns
