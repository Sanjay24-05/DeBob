"""Export one ML feature row per DeBob file node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import NUMERIC_FEATURES, open_db, metadata_value, write_csv


def export_features(repo: Path, output: Path, db_path: Path | None = None) -> int:
    connection = open_db(repo, db_path)
    try:
        rows = connection.execute(
            """
            SELECT n.id AS file_path, n.layer, n.metadata_json,
                   COALESCE(g.commit_count, 0) AS commit_count,
                   COALESCE(g.churn_score, 0) AS churn_score,
                   COALESCE(g.author_count, 0) AS author_count,
                   COALESCE(g.last_modified_at, '') AS last_modified_at
            FROM nodes n
            LEFT JOIN git_file_stats g ON g.file_path = n.id
            WHERE n.type = 'file'
            ORDER BY n.id
            """
        ).fetchall()

        fan_in = dict(connection.execute(
            """SELECT target, COUNT(DISTINCT source) FROM edges
               WHERE type = 'imports' GROUP BY target"""
        ).fetchall())
        fan_out = dict(connection.execute(
            """SELECT source, COUNT(DISTINCT target) FROM edges
               WHERE type = 'imports' GROUP BY source"""
        ).fetchall())
        manifest = repo / ".debob" / "manifest.json"
        manifest_data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}

        exported = []
        for row in rows:
            exported.append({
                "file_path": row["file_path"],
                "churn_score": row["churn_score"],
                "commit_count": row["commit_count"],
                "author_count": row["author_count"],
                "import_fan_in": fan_in.get(row["file_path"], 0),
                "import_fan_out": fan_out.get(row["file_path"], 0),
                "lines_of_code": metadata_value(row["metadata_json"], "linesOfCode", 0),
                "layer": row["layer"] or "unclassified",
                "last_modified_at": row["last_modified_at"],
                "db_schema_version": manifest_data.get("schemaVersion", ""),
                "db_head_commit": manifest_data.get("headCommit", ""),
            })
        fields = ["file_path", *NUMERIC_FEATURES[:1], "commit_count", "author_count",
                  "import_fan_in", "import_fan_out", "lines_of_code", "layer",
                  "last_modified_at", "db_schema_version", "db_head_commit"]
        write_csv(output, exported, fields)
        print(f"Exported {len(exported)} file features to {output}")
        return len(exported)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/features.csv"))
    args = parser.parse_args()
    export_features(args.repo.resolve(), args.output, args.db)


if __name__ == "__main__":
    main()
