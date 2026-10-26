from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]
VALID_INVOICE_PATTERN = r"^\d{6}$"
VALID_STOCKCODE_PATTERN = r"^(?:\d{5}|\d{5}[A-Za-z]+|PADS)$"


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load the raw Online Retail II transactions dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_excel(dataset_path, sheet_name=0)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's documented data cleaning policy.

    The raw dataset preserves all historical transactions. For segmentation, we keep
    only records with valid customer IDs, a 6-digit invoice number, a valid stock
    code, a positive price, and positive quantity. This removes accounting entries,
    cancellations, and returns from the modeling set while keeping the original data
    untouched.
    """
    if df.empty:
        raise ValueError("Transaction dataset is empty.")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["Customer ID", "InvoiceDate"]).copy()
    cleaned = cleaned[cleaned["Invoice"].str.match(VALID_INVOICE_PATTERN, na=False)].copy()
    cleaned = cleaned[cleaned["StockCode"].str.match(VALID_STOCKCODE_PATTERN, na=False)].copy()
    cleaned = cleaned[cleaned["Price"].notna() & (cleaned["Price"] > 0)].copy()
    cleaned = cleaned[cleaned["Quantity"].notna() & (cleaned["Quantity"] > 0)].copy()
    cleaned["SalesLineTotal"] = cleaned["Price"] * cleaned["Quantity"]
    return cleaned
