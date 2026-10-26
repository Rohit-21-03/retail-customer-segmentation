from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler

from customer_segmentation import (
    DEFAULT_DATA_PATH,
    build_cluster_profiles,
    clean_transactions,
    compare_clustering_algorithms,
    compare_preprocessing_variants,
    compute_rfm,
    ensure_output_dir,
    evaluate_k_values,
    fit_kmeans,
    generate_diagnostic_plots,
    load_transactions,
    resolve_repo_path,
    select_k_for_segmentation,
    summarize_data_quality,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the K-Means customer segmentation workflow.")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the Online Retail II Excel file. Relative paths are resolved from the repository root.",
    )
    parser.add_argument("--reference-date", default=None, help="Explicit reference date for Recency calculations (ISO format).")
    parser.add_argument("--max-k", type=int, default=10, help="Largest K value to evaluate.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated summary and profile outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.data_path)
    output_dir = ensure_output_dir(args.output_dir)
    transactions = load_transactions(dataset_path)
    cleaned = clean_transactions(transactions)
    rfm = compute_rfm(cleaned, args.reference_date)

    k_values = list(range(2, min(args.max_k, len(rfm) - 1) + 1))
    metric_table = evaluate_k_values(rfm, k_values)
    selected_k, _ = select_k_for_segmentation(rfm, k_values=k_values, min_cluster_pct=0.02)

    feature_matrix = rfm[["Recency", "Frequency", "MonetaryValue"]]
    scaler = StandardScaler()
    scaled_feature_frame = pd.DataFrame(
        scaler.fit_transform(feature_matrix),
        columns=feature_matrix.columns,
        index=feature_matrix.index,
    )
    model = fit_kmeans(scaled_feature_frame, selected_k)
    labels = model.predict(scaled_feature_frame)
    rfm["Cluster"] = labels

    profile_summary = build_cluster_profiles(rfm)
    baseline_comparison = compare_preprocessing_variants(rfm, k_values=k_values)
    algorithm_comparison = compare_clustering_algorithms(rfm, k_values=k_values)
    quality_summary = summarize_data_quality(transactions)
    diagnostic_paths = generate_diagnostic_plots(rfm, k_values=k_values, output_dir=output_dir)

    summary = {
        "selected_k": selected_k,
        "rows_after_cleaning": len(cleaned),
        "customers_in_rfm": len(rfm),
        "best_silhouette": float(metric_table["silhouette_score"].max()),
        "selected_silhouette": float(metric_table.loc[metric_table["k"] == selected_k, "silhouette_score"].iloc[0]),
        "cluster_sizes": {str(k): int(v) for k, v in rfm["Cluster"].value_counts().sort_index().items()},
        "cluster_profiles": profile_summary.to_dict(orient="records"),
        "diagnostic_plots": diagnostic_paths,
        "selection_note": (
            "K=2 remains the strongest statistical baseline for this dataset, but the cluster is extremely "
            "imbalanced and should be treated as a business baseline rather than a final operational segmentation."
        ),
    }

    with (output_dir / "segmentation_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    with (output_dir / "cluster_profiles.csv").open("w", encoding="utf-8", newline="") as csv_file:
        profile_summary.to_csv(csv_file, index=False)
    with (output_dir / "k_evaluation.csv").open("w", encoding="utf-8", newline="") as csv_file:
        metric_table.to_csv(csv_file, index=False)
    with (output_dir / "preprocessing_comparison.csv").open("w", encoding="utf-8", newline="") as csv_file:
        baseline_comparison.to_csv(csv_file, index=False)
    with (output_dir / "algorithm_comparison.csv").open("w", encoding="utf-8", newline="") as csv_file:
        algorithm_comparison.to_csv(csv_file, index=False)
    with (output_dir / "data_quality_summary.csv").open("w", encoding="utf-8", newline="") as csv_file:
        quality_summary.to_csv(csv_file, index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
