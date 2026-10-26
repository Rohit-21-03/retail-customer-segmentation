from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from customer_segmentation import (
    build_cluster_profiles,
    clean_transactions,
    compare_clustering_algorithms,
    compare_preprocessing_variants,
    compute_rfm,
    evaluate_k_values,
    generate_diagnostic_plots,
    load_transactions,
    select_k_for_segmentation,
    summarize_data_quality,
)

DATA_PATH = "data/online_retail_II.xlsx"


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["536365", "536365", "536366", "C536367", "536368", "536369"],
            "StockCode": ["85123A", "71053", "84406B", "85123A", "21833", "22086"],
            "Description": [
                "White Metal Lantern",
                "White Metal Lantern",
                "Cream Cupid Hearts Co",
                "White Metal Lantern",
                "Pillows",
                "Pillows",
            ],
            "Quantity": [2, 3, 1, -1, 5, 2],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-12-01 08:26:00",
                    "2010-12-01 08:26:00",
                    "2010-12-01 09:00:00",
                    "2010-12-01 10:00:00",
                    "2010-12-01 09:30:00",
                    "2010-12-01 11:00:00",
                ]
            ),
            "Price": [2.5, 3.0, 4.0, 2.5, 5.0, 6.0],
            "Customer ID": [1001.0, 1001.0, 1002.0, 1002.0, 1003.0, None],
            "Country": ["United Kingdom"] * 6,
        }
    )


def test_load_transactions_has_expected_schema() -> None:
    df = load_transactions(DATA_PATH)
    required = {"Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"}
    assert required.issubset(df.columns)
    assert len(df) > 0


def test_clean_transactions_removes_invalid_rows(sample_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(sample_transactions)
    assert len(cleaned) == 4
    assert set(cleaned["Customer ID"].unique()) == {1001.0, 1002.0, 1003.0}
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["Price"] > 0).all()
    assert cleaned["Invoice"].str.match(r"^\d{6}$").all()


def test_compute_rfm_uses_reference_date_and_has_valid_values(sample_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(sample_transactions)
    rfm = compute_rfm(cleaned, reference_date="2010-12-01")
    assert set(rfm.columns) >= {"Customer ID", "MonetaryValue", "Frequency", "LastPurchaseDate", "Recency"}
    assert len(rfm) == 3
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["MonetaryValue"] > 0).all()
    assert (rfm["Frequency"] > 0).all()


def test_evaluate_k_values_returns_metrics_for_valid_k() -> None:
    df = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4, 5],
            "Recency": [10, 20, 30, 40, 50],
            "Frequency": [5, 4, 3, 2, 1],
            "MonetaryValue": [100, 80, 60, 40, 20],
        }
    )
    metrics = evaluate_k_values(df, [2, 3])
    assert list(metrics.columns) == [
        "k",
        "inertia",
        "silhouette_score",
        "cluster_sizes",
        "min_cluster_size",
        "max_cluster_size",
        "cluster_imbalance_ratio",
    ]
    assert set(metrics["k"]) == {2, 3}


def test_cluster_profiles_and_k_selection_stay_valid() -> None:
    df = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4],
            "Recency": [5, 10, 8, 12],
            "Frequency": [2, 3, 4, 5],
            "MonetaryValue": [100, 200, 300, 400],
            "Cluster": [0, 0, 1, 1],
        }
    )
    profiles = build_cluster_profiles(df)
    assert set(profiles.columns) >= {"Cluster", "customer_count", "customer_pct", "avg_recency", "revenue_pct"}
    assert profiles["customer_count"].sum() == 4

    selected_k, metrics = select_k_for_segmentation(df[["Recency", "Frequency", "MonetaryValue"]], k_values=[2, 3])
    assert selected_k in {2, 3}
    assert "silhouette_score" in metrics.columns


def test_preprocessing_comparison_runs_for_valid_rfm() -> None:
    df = pd.DataFrame(
        {
            "Recency": [1, 2, 5, 10],
            "Frequency": [2, 3, 4, 5],
            "MonetaryValue": [50, 60, 90, 200],
        }
    )
    comparison = compare_preprocessing_variants(df, k_values=[2, 3])
    assert set(comparison["variant"]) == {"baseline", "log1p"}
    assert set(comparison["k"]) == {2, 3}


def test_clustering_algorithm_comparison_runs() -> None:
    df = pd.DataFrame(
        {
            "Recency": [1, 2, 3, 15, 16, 17],
            "Frequency": [2, 3, 4, 7, 8, 9],
            "MonetaryValue": [50, 60, 80, 220, 260, 300],
        }
    )
    comparison = compare_clustering_algorithms(df, k_values=[2, 3])
    assert set(comparison["algorithm"]) == {"kmeans", "agglomerative"}
    assert set(comparison["k"]) == {2, 3}


def test_data_quality_summary_counts_invalid_record_classes(sample_transactions: pd.DataFrame) -> None:
    summary = summarize_data_quality(sample_transactions)
    summary_by_metric = dict(zip(summary["metric"], summary["value"]))
    assert summary_by_metric["total_rows"] == 6
    assert summary_by_metric["missing_customer_id"] == 1
    assert summary_by_metric["cancelled_invoices"] == 1
    assert summary_by_metric["negative_quantity"] == 1
    assert summary_by_metric["rows_after_policy"] >= 3


def test_generate_diagnostic_plots_writes_files(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4, 5, 6],
            "Recency": [1, 2, 3, 15, 16, 17],
            "Frequency": [2, 3, 4, 7, 8, 9],
            "MonetaryValue": [50, 60, 80, 220, 260, 300],
        }
    )
    result = generate_diagnostic_plots(df, k_values=[2, 3], output_dir=tmp_path)
    assert set(result) == {"elbow_curve", "silhouette_curve"}
    assert Path(result["elbow_curve"]).exists()
    assert Path(result["silhouette_curve"]).exists()


def test_compute_rfm_rejects_empty_input() -> None:
    empty = pd.DataFrame(columns=["Customer ID", "Invoice", "SalesLineTotal", "InvoiceDate"])
    with pytest.raises(ValueError):
        compute_rfm(empty)
