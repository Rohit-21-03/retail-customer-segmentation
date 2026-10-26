from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "online_retail_II.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def resolve_repo_path(path: str | Path | None) -> Path:
    """Resolve repository-local paths without depending on the current working directory."""
    if path is None or path == "":
        return DEFAULT_DATA_PATH

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def ensure_output_dir(path: str | Path | None = None) -> Path:
    """Create the project output directory if it does not already exist."""
    output_dir = resolve_repo_path(path) if path is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
