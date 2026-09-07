"""Score current DeBob files with a saved joblib model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from common import LAYER_FEATURE


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


def predict(repo: Path, model_path: Path, output: Path, top: int) -> None:
    bundle = joblib.load(model_path)
    expected = bundle.get("features")
    if not isinstance(expected, list) or not expected or LAYER_FEATURE not in expected:
        raise ValueError("Model artifact does not contain a valid feature schema.")
    temporary = output.parent / ".predict_features.csv"
    from export_features import export_features
    export_features(repo, temporary)
    try:
        data = pd.read_csv(temporary)
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
            "model_provenance": bundle.get("provenance", {}),
            "prediction_rows": len(data),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/predictions.csv"))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    predict(args.repo.resolve(), args.model, args.output, args.top)


if __name__ == "__main__":
    main()
