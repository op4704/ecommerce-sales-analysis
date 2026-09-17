"""
Loads the Online Retail II dataset and caches it locally as CSV.

Note: the `ucimlrepo` package's API for this dataset (id=502) returns
"not available for import" even though the dataset page is live, so we
pull the static file bundle directly from UCI's archive instead.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_ZIP = RAW_DIR / "online_retail_ii.zip"
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
RAW_CSV = RAW_DIR / "online_retail_ii.csv"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def download_online_retail_ii() -> pd.DataFrame:
    """
    Download the 'Online Retail II' dataset directly from UCI's static file archive.

    The dataset ships as a single .xlsx with two sheets — one per year of
    transactions ("Year 2009-2010" and "Year 2010-2011"). We combine both
    into one continuous dataframe covering 01/12/2009 - 09/12/2011.

    Returns
    -------
    pd.DataFrame
        Raw, uncleaned, combined two-year transaction data.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_XLSX.exists():
        response = requests.get(UCI_ZIP_URL, timeout=120)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(RAW_DIR)

    sheets = pd.read_excel(RAW_XLSX, sheet_name=None)  # dict of {sheet_name: df}
    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load the raw Online Retail II dataset, using a local CSV cache when available.

    Parameters
    ----------
    force_download : bool
        If True, re-download/re-parse from the source .xlsx even if a
        cached CSV already exists.

    Returns
    -------
    pd.DataFrame
        Raw transaction-level e-commerce data, untouched (no cleaning applied).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists() and not force_download:
        return pd.read_csv(RAW_CSV, encoding="utf-8")

    df = download_online_retail_ii()
    df.to_csv(RAW_CSV, index=False)
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(f"Loaded {len(data):,} rows, {len(data.columns)} columns")
    print(data.head())
    print(data.dtypes)
