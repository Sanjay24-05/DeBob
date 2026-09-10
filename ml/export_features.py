"""Export one ML feature row per DeBob file node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    CandidateScope,
    NUMERIC_FEATURES,
    classify_file,
    is_candidate_scope,
    metadata_value,
    normalise_path,
    open_db,
    write_csv,
)


def physical_lines(repo: Path, file_path: str) -> int:
    path = repo / file_path
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def export_features(
    repo: Path,
    output: Path,
    db_path: Path | None = None,
    scope: CandidateScope = "source",
) -> int:
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

        all_rows = []
        for row in rows:
            file_path = normalise_path(row["file_path"])
            category = classify_file(file_path)
            lines = metadata_value(row["metadata_json"], "linesOfCode", 0)
            if not lines:
                lines = physical_lines(repo, file_path)
            fan_in_value = fan_in.get(row["file_path"], 0)
            fan_out_value = fan_out.get(row["file_path"], 0)
            all_rows.append({
                "file_path": file_path,
                "churn_score": row["churn_score"],
                "commit_count": row["commit_count"],
                "author_count": row["author_count"],
                "import_fan_in": fan_in_value,
                "import_fan_out": fan_out_value,
                "lines_of_code": lines,
                "layer": row["layer"] or "unclassified",
                "file_category": category,
                "meaningful_code": int(category == "source" and (float(lines) > 0 or fan_in_value > 0 or fan_out_value > 0)),
                "last_modified_at": row["last_modified_at"],
                "db_schema_version": manifest_data.get("schemaVersion", ""),
                "db_head_commit": manifest_data.get("headCommit", ""),
            })
        exported = [row for row in all_rows if is_candidate_scope(row["file_category"], scope)]
        fields = ["file_path", *NUMERIC_FEATURES[:1], "commit_count", "author_count",
                  "import_fan_in", "import_fan_out", "lines_of_code", "layer",
                  "file_category", "meaningful_code", "last_modified_at", "db_schema_version", "db_head_commit"]
        write_csv(output, exported, fields)
        excluded = [row for row in all_rows if row not in exported]
        excluded_by_category = {}
        for row in excluded:
            excluded_by_category[row["file_category"]] = excluded_by_category.get(row["file_category"], 0) + 1
        metadata = {
            "candidate_scope": scope,
            "total_file_nodes": len(all_rows),
            "included_rows": len(exported),
            "excluded_rows": len(excluded),
            "excluded_by_category": excluded_by_category,
            "db_schema_versions": sorted({str(row["db_schema_version"]) for row in all_rows if row["db_schema_version"]}),
            "db_head_commits": sorted({str(row["db_head_commit"]) for row in all_rows if row["db_head_commit"]}),
        }
        output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Exported {len(exported)} file features to {output}")
        return len(exported)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/features.csv"))
    parser.add_argument("--scope", choices=("source", "all"), default="source")
    args = parser.parse_args()
    export_features(args.repo.resolve(), args.output, args.db, args.scope)


if __name__ == "__main__":
    main()
