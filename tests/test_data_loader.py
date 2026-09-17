"""
Tests for data_loader.py

We don't want unit tests hitting the network / UCI servers every run, so
these tests use monkeypatching to verify the caching behavior and the
sheet-combining logic in isolation, without requiring an actual download.
"""

import pandas as pd
import pytest

from ecommerce_sales_analysis import data_loader


def test_load_raw_data_uses_cache_when_present(tmp_path, monkeypatch):
    fake_csv = tmp_path / "online_retail_ii.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(fake_csv, index=False)

    monkeypatch.setattr(data_loader, "RAW_CSV", fake_csv)
    monkeypatch.setattr(data_loader, "RAW_DIR", tmp_path)

    def fail_if_called():
        raise AssertionError("download_online_retail_ii should not be called when cache exists")

    monkeypatch.setattr(data_loader, "download_online_retail_ii", lambda: fail_if_called())

    result = data_loader.load_raw_data()
    assert len(result) == 2
    assert list(result.columns) == ["a", "b"]


def test_load_raw_data_force_download_ignores_cache(tmp_path, monkeypatch):
    fake_csv = tmp_path / "online_retail_ii.csv"
    pd.DataFrame({"a": [1]}).to_csv(fake_csv, index=False)

    monkeypatch.setattr(data_loader, "RAW_CSV", fake_csv)
    monkeypatch.setattr(data_loader, "RAW_DIR", tmp_path)

    fresh_df = pd.DataFrame({"a": [9, 9, 9]})
    monkeypatch.setattr(data_loader, "download_online_retail_ii", lambda: fresh_df)

    result = data_loader.load_raw_data(force_download=True)
    assert len(result) == 3


def test_download_combines_both_year_sheets(tmp_path, monkeypatch):
    """
    The real dataset ships as an xlsx with two sheets (one per year).
    download_online_retail_ii() must concatenate them into one dataframe,
    not silently drop one.
    """
    fake_xlsx = tmp_path / "online_retail_II.xlsx"
    sheet_2009 = pd.DataFrame({"Invoice": ["1", "2"]})
    sheet_2010 = pd.DataFrame({"Invoice": ["3", "4", "5"]})

    with pd.ExcelWriter(fake_xlsx) as writer:
        sheet_2009.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        sheet_2010.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    monkeypatch.setattr(data_loader, "RAW_XLSX", fake_xlsx)
    monkeypatch.setattr(data_loader, "RAW_DIR", tmp_path)

    result = data_loader.download_online_retail_ii()
    assert len(result) == 5  # 2 + 3 rows combined, not just one sheet
