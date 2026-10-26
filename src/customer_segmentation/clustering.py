from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MODEL_FEATURES = ["Recency", "Frequency", "MonetaryValue"]


def scale_rfm_features(rfm_df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Standardize RFM features while preserving DataFrame columns."""
    columns = feature_columns or MODEL_FEATURES
    if not set(columns).issubset(rfm_df.columns):
        raise ValueError(f"RFM dataframe must include: {columns}")

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(rfm_df[columns])
    return pd.DataFrame(scaled_matrix, columns=columns, index=rfm_df.index)


def fit_kmeans(data: pd.DataFrame, n_clusters: int, random_state: int = 42) -> KMeans:
    """Fit a K-Means model on the RFM feature matrix."""
    if n_clusters < 2:
        raise ValueError("K-Means requires at least two clusters.")
    if data.empty:
        raise ValueError("RFM feature matrix is empty.")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(data)
    return model


def evaluate_k_values(
    rfm_df: pd.DataFrame,
    k_values: range | list[int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate a set of K values using inertia and silhouette score."""
    if not set(MODEL_FEATURES).issubset(rfm_df.columns):
        raise ValueError(f"RFM dataframe must include: {MODEL_FEATURES}")

    if k_values is None:
        k_values = range(2, 11)
    k_values = list(k_values)
    if not k_values or min(k_values) < 2:
        raise ValueError("K values must start at 2 or higher.")

    scaled_matrix = scale_rfm_features(rfm_df)
    rows = []

    for k in k_values:
        if k >= len(rfm_df):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled_matrix)
        sizes = pd.Series(labels).value_counts().sort_index()
        silhouette = silhouette_score(scaled_matrix, labels)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette),
                "cluster_sizes": dict(sizes.to_dict()),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "cluster_imbalance_ratio": float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf"),
            }
        )

    if not rows:
        raise ValueError("No valid K values were available for evaluation.")
    return pd.DataFrame(rows)


def select_k_for_segmentation(
    rfm_df: pd.DataFrame,
    k_values: range | list[int] | None = None,
    random_state: int = 42,
    min_cluster_pct: float = 0.02,
) -> tuple[int, pd.DataFrame]:
    """Select a K value while guarding against tiny or degenerate clusters.

    The customer dataset naturally produces a highly imbalanced K=2 structure, so a
    minimum cluster share filter may exclude all candidates. In that case, the
    highest-silhouette result is preserved as the statistical baseline while the
    cluster-balance diagnostics remain visible to the caller.
    """
    metrics = evaluate_k_values(rfm_df, k_values=k_values, random_state=random_state)
    if metrics.empty:
        raise ValueError("No metrics were generated for model selection.")

    total_customers = len(rfm_df)
    metrics["max_cluster_pct"] = metrics["cluster_sizes"].map(
        lambda sizes: max(sizes.values()) / total_customers if sizes else 0.0
    )
    metrics["min_cluster_pct"] = metrics["cluster_sizes"].map(
        lambda sizes: min(sizes.values()) / total_customers if sizes else 0.0
    )
    valid = metrics[metrics["min_cluster_pct"] >= min_cluster_pct].copy()
    if valid.empty:
        valid = metrics.copy()

    chosen = valid.sort_values(["silhouette_score", "inertia"], ascending=[False, True]).iloc[0]
    return int(chosen["k"]), metrics
