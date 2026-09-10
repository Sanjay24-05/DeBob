"""Score current DeBob files with a saved joblib model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from common import CandidateScope, LAYER_FEATURE, OPERATIONAL_MODEL_NAME, DEFAULT_TOP_K


def reason_label(feature: str) -> str:
    labels = {
        "churn_score": "high commit-touch churn",
        "author_count": "many contributors",
        "import_fan_in": "high import fan-in",
        "import_fan_out": "high import fan-out",
        "lines_of_code": "large module",
        "layer": "architectural layer signal",
    }
    return labels.get(feature, feature.replace("_", " "))


def predict(repo: Path, model_path: Path, output: Path, top: int, scope: CandidateScope = "source") -> None:
    bundle = joblib.load(model_path)
    if bundle.get("diagnostic_only") or bundle.get("leakage_warning"):
        import sys
        warning = bundle.get("leakage_warning") or "This model is a leakage diagnostic and should not be used for operational ranking."
        print(f"WARNING: {warning}", file=sys.stderr)
        print(f"WARNING: Use the without-churn model for operational predictions.", file=sys.stderr)
    expected = bundle.get("features")
    if not isinstance(expected, list) or not expected or LAYER_FEATURE not in expected:
        raise ValueError("Model artifact does not contain a valid feature schema.")
    model_scope = bundle.get("candidate_scope", "source")
    if model_scope != scope:
        raise ValueError(f"Model candidate scope is {model_scope!r}, but prediction scope is {scope!r}.")
    temporary = output.parent / ".predict_features.csv"
    from export_features import export_features
    export_features(repo, temporary, scope=scope)
    try:
        data = pd.read_csv(temporary)
        metadata_path = temporary.with_suffix(temporary.suffix + ".meta.json")
        export_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if "file_category" not in data or "meaningful_code" not in data:
            raise ValueError("Exported features are missing file scope columns.")
        data = data[data["meaningful_code"].astype(str) == "1"].copy()
        if data.empty:
            raise ValueError("No meaningful source files are available for prediction.")
        missing = [feature for feature in expected if feature not in data.columns]
        if missing:
            raise ValueError(f"Exported features are missing model columns: {', '.join(missing)}")
        scores = bundle["model"].predict_proba(data[expected])[:, 1]
        importance = bundle.get("feature_importance", {})
        thresholds = bundle.get("feature_thresholds", {})
        reasons = []
        for index, row in data.iterrows():
            row_reasons = []
            for feature, feature_importance in sorted(importance.items(), key=lambda item: item[1], reverse=True):
                if feature not in row or feature == LAYER_FEATURE:
                    continue
                value = float(row[feature])
                threshold = float(thresholds.get(feature, 0))
                if threshold > 0 and value >= threshold:
                    row_reasons.append({"feature": feature, "label": reason_label(feature), "value": value, "importance": round(float(feature_importance), 4)})
                if len(row_reasons) == 3:
                    break
            reasons.append(row_reasons)
        result = pd.DataFrame({"file_path": data["file_path"], "risk_probability": scores, "predicted_risky": (scores >= 0.5).astype(int), "risk_reasons": [json.dumps(value) for value in reasons], "model": bundle["model_name"]}).sort_values("risk_probability", ascending=False).head(top)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")
        else:
            result.to_csv(output, index=False)
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "model_name": bundle.get("model_name"),
            "label_caveat": bundle.get("label_caveat"),
            "leakage_warning": bundle.get("leakage_warning"),
            "model_provenance": bundle.get("provenance", {}),
            "prediction_rows": len(data),
            "candidate_scope": scope,
            "total_file_nodes": export_metadata.get("total_file_nodes", len(data)),
            "candidate_count": len(data),
            "excluded_count": export_metadata.get("excluded_rows", 0),
            "excluded_by_category": export_metadata.get("excluded_by_category", {}),
            "top": top,
            "db_schema_versions": sorted(str(value) for value in data["db_schema_version"].dropna().unique()) if "db_schema_version" in data else [],
            "db_head_commits": sorted(str(value) for value in data["db_head_commit"].dropna().unique()) if "db_head_commit" in data else [],
        }
        output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for row in result.itertuples(index=False):
            print(f"{row.risk_probability:.3f} {row.file_path}")
            for reason in json.loads(row.risk_reasons):
                print(f"  - {reason['label']} ({reason['value']:g})")
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(temporary.suffix + ".meta.json").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=Path(f"ml/artifacts/models/{OPERATIONAL_MODEL_NAME}.joblib"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/predictions.csv"))
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--scope", choices=("source", "all"), default="source")
    args = parser.parse_args()
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    predict(args.repo.resolve(), args.model, args.output, args.top, args.scope)


if __name__ == "__main__":
    main()
