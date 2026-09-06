"""Shared helpers for DeBob's file-risk ML workflow."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable

NUMERIC_FEATURES = [
    "churn_score",
    "author_count",
    "import_fan_in",
    "import_fan_out",
    "lines_of_code",
]
LAYER_FEATURE = "layer"
LABEL = "risky"


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
