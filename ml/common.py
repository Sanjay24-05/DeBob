"""Shared helpers for DeBob's file-risk ML workflow."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Literal

NUMERIC_FEATURES = [
    "churn_score",
    "author_count",
    "import_fan_in",
    "import_fan_out",
    "lines_of_code",
]
LAYER_FEATURE = "layer"
LABEL = "risky"
OPERATIONAL_MODEL_NAME = "random_forest_without_churn"
DIAGNOSTIC_MODEL_NAME = "random_forest_with_churn_diagnostic"
DEFAULT_TOP_K = 5
CandidateScope = Literal["source", "all"]
FILE_CATEGORIES = (
    "source",
    "config",
    "documentation",
    "dependency_metadata",
    "generated_or_artifact",
)
SOURCE_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".py"}
DOCUMENTATION_EXTENSIONS = {".md", ".mdx", ".html", ".txt"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini", ".cfg"}
GENERATED_PARTS = {".debob", "dist", "build", "coverage", "__pycache__", "artifacts"}
DEPENDENCY_FILES = {
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "requirements.txt",
    "poetry.lock",
    "pipfile.lock",
}


def normalise_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def classify_file(path: str) -> str:
    """Classify a repository file, prioritising generated content over extensions."""
    normalised = normalise_path(path)
    parts = set(normalised.lower().split("/"))
    name = Path(normalised).name.lower()
    suffix = Path(name).suffix
    if parts & GENERATED_PARTS or name.endswith(".min.js") or name.endswith(".generated.ts"):
        return "generated_or_artifact"
    if name in DEPENDENCY_FILES or name.endswith("-lock.json"):
        return "dependency_metadata"
    if "docs" in parts or suffix in DOCUMENTATION_EXTENSIONS:
        return "documentation"
    if name.startswith("tsconfig") or name.endswith((".config.js", ".config.ts", ".config.cjs", ".config.mjs")) or suffix in CONFIG_EXTENSIONS:
        return "config"
    if suffix in SOURCE_EXTENSIONS:
        return "source"
    return "config"


def is_candidate_scope(category: str, scope: CandidateScope) -> bool:
    return scope == "all" or category == "source"


def open_db(repo: Path, db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or repo / ".debob" / "context.db"
    if not path.exists():
        raise FileNotFoundError(f"DeBob database not found: {path}. Run debob init first.")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metadata_value(raw: str | None, key: str, default: object = 0) -> object:
    if not raw:
        return default
    try:
        return json.loads(raw).get(key, default)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
