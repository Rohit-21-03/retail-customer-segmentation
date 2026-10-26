"""Customer segmentation package for RFM analysis and K-Means clustering."""

from .clustering import evaluate_k_values, fit_kmeans, scale_rfm_features, select_k_for_segmentation
from .config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT, ensure_output_dir, resolve_repo_path
from .data_loader import clean_transactions, load_transactions
from .reporting import (
    build_cluster_profiles,
    compare_clustering_algorithms,
    compare_preprocessing_variants,
    generate_diagnostic_plots,
    summarize_data_quality,
)
from .rfm import compute_rfm

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_DIR",
    "PROJECT_ROOT",
    "build_cluster_profiles",
    "clean_transactions",
    "compare_clustering_algorithms",
    "compare_preprocessing_variants",
    "compute_rfm",
    "ensure_output_dir",
    "evaluate_k_values",
    "fit_kmeans",
    "generate_diagnostic_plots",
    "load_transactions",
    "resolve_repo_path",
    "scale_rfm_features",
    "select_k_for_segmentation",
    "summarize_data_quality",
]
