"""Apply DeBob's documented weak risk label to exported features."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from common import read_csv, write_csv

BUG_WORDS = re.compile(r"\b(fix|bug|hotfix|patch)\b", re.IGNORECASE)


def label_features(repo: Path, features_path: Path, output: Path, db_path: Path | None = None) -> int:
    rows = read_csv(features_path)
    if not rows:
        raise ValueError("No feature rows were found; cannot create labels.")
    churn = sorted(
        float(row.get("churn_score", 0) or 0)
        for row in rows
        if float(row.get("churn_score", 0) or 0) > 0
    )
    threshold = churn[max(0, int(0.75 * len(churn)) - 1)] if churn else 0.0
    connection = sqlite3.connect(db_path or repo / ".debob" / "context.db")
    try:
        touched_by_fix: set[str] = set()
        for subject, files_json in connection.execute("SELECT subject, files_changed_json FROM git_commits"):
            if not BUG_WORDS.search(subject or ""):
                continue
            try:
                touched_by_fix.update(json.loads(files_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    finally:
        connection.close()

    for row in rows:
        path = row["file_path"].replace("\\", "/")
        churn_score = float(row.get("churn_score", 0) or 0)
        row["risky"] = int((churn_score > 0 and churn_score >= threshold) or path in touched_by_fix)
    fields = list(rows[0].keys())
    if "risky" not in fields:
        fields.append("risky")
    write_csv(output, rows, fields)
    print(f"Labeled {len(rows)} files; churn threshold={threshold:g}; output={output}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--features", type=Path, default=Path("ml/artifacts/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/labeled_features.csv"))
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    label_features(args.repo.resolve(), args.features, args.output, args.db)


if __name__ == "__main__":
    main()
