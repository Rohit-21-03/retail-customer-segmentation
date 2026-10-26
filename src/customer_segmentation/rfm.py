from __future__ import annotations

import pandas as pd


def compute_rfm(
    cleaned_transactions: pd.DataFrame,
    reference_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Compute the customer-level RFM features.

    Definitions:
    - Recency: days since the customer's most recent purchase relative to the
      reference date.
    - Frequency: number of distinct invoices per customer.
    - MonetaryValue: sum of sales line totals per customer.
    """
    if cleaned_transactions.empty:
        raise ValueError("No transactions remain after cleaning.")

    if reference_date is None:
        reference_date = pd.Timestamp(cleaned_transactions["InvoiceDate"].max())
    else:
        reference_date = pd.Timestamp(reference_date)

    customer_summary = (
        cleaned_transactions.groupby("Customer ID", as_index=False)
        .agg(
            MonetaryValue=("SalesLineTotal", "sum"),
            Frequency=("Invoice", "nunique"),
            LastPurchaseDate=("InvoiceDate", "max"),
        )
        .sort_values("Customer ID")
        .reset_index(drop=True)
    )

    customer_summary["Recency"] = (
        reference_date - customer_summary["LastPurchaseDate"]
    ).dt.days.astype(int)
    customer_summary["Recency"] = customer_summary["Recency"].clip(lower=0)

    feature_frame = customer_summary[["Recency", "Frequency", "MonetaryValue"]]
    if feature_frame.isna().any().any():
        raise ValueError("RFM features contain missing values after aggregation.")
    if (feature_frame <= 0).all().all():
        raise ValueError("RFM features are invalid; all values are non-positive.")

    return customer_summary
