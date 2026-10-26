from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .clustering import MODEL_FEATURES, evaluate_k_values, scale_rfm_features


def build_cluster_profiles(assignments: pd.DataFrame) -> pd.DataFrame:
    """Create a cluster profile summary for business interpretation."""
    required = {"Customer ID", "Recency", "Frequency", "MonetaryValue", "Cluster"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignments are missing required columns: {sorted(missing)}")

    cluster_summary = (
        assignments.groupby("Cluster", dropna=False)
        .agg(
            customer_count=("Customer ID", "count"),
            avg_recency=("Recency", "mean"),
            median_recency=("Recency", "median"),
            avg_frequency=("Frequency", "mean"),
            median_frequency=("Frequency", "median"),
            avg_monetary=("MonetaryValue", "mean"),
            median_monetary=("MonetaryValue", "median"),
            total_revenue=("MonetaryValue", "sum"),
        )
        .reset_index()
    )

    total_customers = cluster_summary["customer_count"].sum()
    total_revenue = cluster_summary["total_revenue"].sum()
    if total_customers == 0:
        raise ValueError("Cluster assignments contain no customers.")
    if total_revenue == 0:
        cluster_summary["customer_pct"] = cluster_summary["customer_count"] / total_customers
        cluster_summary["revenue_pct"] = 0.0
    else:
        cluster_summary["customer_pct"] = cluster_summary["customer_count"] / total_customers
        cluster_summary["revenue_pct"] = cluster_summary["total_revenue"] / total_revenue

    cluster_summary = cluster_summary.sort_values("Cluster").reset_index(drop=True)
    return cluster_summary


def summarize_data_quality(transactions: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw data-quality issues and the selected retention policy."""
    if transactions.empty:
        raise ValueError("Transaction dataset is empty.")

    raw = transactions.copy()
    raw["Invoice"] = raw["Invoice"].astype(str)
    raw["StockCode"] = raw["StockCode"].astype(str)

    invoice_valid = raw["Invoice"].str.match(r"^\d{6}$", na=False)
    stock_valid = raw["StockCode"].str.match(r"^(?:\d{5}|\d{5}[A-Za-z]+|PADS)$", na=False)
    valid_policy_mask = (
        raw["Customer ID"].notna()
        & invoice_valid
        & stock_valid
        & raw["InvoiceDate"].notna()
        & (raw["Price"] > 0)
        & (raw["Quantity"] > 0)
    )

    summary = pd.DataFrame(
        [
            {"metric": "total_rows", "value": int(len(raw))},
            {"metric": "missing_customer_id", "value": int(raw["Customer ID"].isna().sum())},
            {"metric": "duplicate_rows", "value": int(raw.duplicated().sum())},
            {"metric": "invalid_invoice_pattern", "value": int((~invoice_valid).sum())},
            {"metric": "cancelled_invoices", "value": int(raw["Invoice"].str.startswith("C").sum())},
            {"metric": "accounting_invoices", "value": int(raw["Invoice"].str.startswith("A").sum())},
            {"metric": "invalid_stock_code", "value": int((~stock_valid).sum())},
            {"metric": "negative_quantity", "value": int((raw["Quantity"] < 0).sum())},
            {"metric": "zero_quantity", "value": int((raw["Quantity"] == 0).sum())},
            {"metric": "negative_price", "value": int((raw["Price"] < 0).sum())},
            {"metric": "zero_price", "value": int((raw["Price"] == 0).sum())},
            {"metric": "rows_after_policy", "value": int(valid_policy_mask.sum())},
        ]
    )
    return summary


def generate_diagnostic_plots(
    rfm_df: pd.DataFrame,
    k_values: list[int] | range | None = None,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, str]:
    """Save elbow and silhouette plots for the evaluated K values."""
    metrics = evaluate_k_values(rfm_df, k_values=k_values, random_state=random_state)
    output_path = Path(output_dir) if output_dir is not None else Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_path / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    elbow_path = diagnostic_dir / "elbow_curve.png"
    silhouette_path = diagnostic_dir / "silhouette_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(metrics["k"], metrics["inertia"], marker="o")
    plt.title("Elbow curve for KMeans on standardized RFM features")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics["k"], metrics["silhouette_score"], marker="o", color="tab:green")
    plt.title("Silhouette score by K")
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(silhouette_path, dpi=200)
    plt.close()

    return {"elbow_curve": str(elbow_path), "silhouette_curve": str(silhouette_path)}


def compare_preprocessing_variants(
    rfm_df: pd.DataFrame,
    k_values: list[int] | range | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare baseline and log-transformed feature preprocessing on the same data."""
    if not set(MODEL_FEATURES).issubset(rfm_df.columns):
        raise ValueError(f"RFM dataframe must include {MODEL_FEATURES}")

    if k_values is None:
        k_values = range(2, 11)
    k_values = list(k_values)

    def evaluate_variant(name: str, matrix: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        scaled = scale_rfm_features(matrix)
        for k in k_values:
            if k >= len(rfm_df):
                continue
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(scaled)
            sizes = pd.Series(labels).value_counts().sort_index()
            rows.append(
                {
                    "variant": name,
                    "k": int(k),
                    "inertia": float(model.inertia_),
                    "silhouette_score": float(silhouette_score(scaled, labels)),
                    "cluster_sizes": dict(sizes.to_dict()),
                    "min_cluster_size": int(sizes.min()),
                    "max_cluster_size": int(sizes.max()),
                }
            )
        return pd.DataFrame(rows)

    baseline = evaluate_variant("baseline", rfm_df[MODEL_FEATURES].copy())
    log_matrix = rfm_df[MODEL_FEATURES].astype(float).clip(lower=0)
    log_variant = evaluate_variant("log1p", pd.DataFrame(np.log1p(log_matrix), columns=MODEL_FEATURES))
    return pd.concat([baseline, log_variant], ignore_index=True)


def compare_clustering_algorithms(
    rfm_df: pd.DataFrame,
    k_values: list[int] | range | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare KMeans against a simple agglomerative baseline on the same features."""
    if not set(MODEL_FEATURES).issubset(rfm_df.columns):
        raise ValueError(f"RFM dataframe must include {MODEL_FEATURES}")

    if k_values is None:
        k_values = range(2, 8)
    k_values = list(k_values)
    scaled_data = scale_rfm_features(rfm_df)
    results = []

    for algorithm in ["kmeans", "agglomerative"]:
        for k in k_values:
            if k >= len(rfm_df):
                continue
            if algorithm == "kmeans":
                model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = model.fit_predict(scaled_data)
                inertia = float(model.inertia_)
            else:
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(scaled_data)
                inertia = None
            sizes = pd.Series(labels).value_counts().sort_index()
            results.append(
                {
                    "algorithm": algorithm,
                    "k": int(k),
                    "inertia": inertia,
                    "silhouette_score": float(silhouette_score(scaled_data, labels)),
                    "cluster_sizes": dict(sizes.to_dict()),
                    "min_cluster_size": int(sizes.min()),
                    "max_cluster_size": int(sizes.max()),
                }
            )
    return pd.DataFrame(results)
