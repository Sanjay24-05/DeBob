"""Score current DeBob files with a saved joblib model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from common import NUMERIC_FEATURES, LAYER_FEATURE, open_db


def predict(repo: Path, model_path: Path, output: Path, top: int) -> None:
    bundle = joblib.load(model_path)
    expected = [*NUMERIC_FEATURES, LAYER_FEATURE]
    if bundle.get("features") != expected:
        raise ValueError("Model feature schema does not match this DeBob ML version.")
    temporary = output.parent / ".predict_features.csv"
    from export_features import export_features
    export_features(repo, temporary)
    try:
        data = pd.read_csv(temporary)
        scores = bundle["model"].predict_proba(data[expected])[:, 1]
        result = pd.DataFrame({"file_path": data["file_path"], "risk_probability": scores, "predicted_risky": (scores >= 0.5).astype(int), "model": bundle["model_name"]}).sort_values("risk_probability", ascending=False).head(top)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")
        else:
            result.to_csv(output, index=False)
        for row in result.itertuples(index=False):
            print(f"{row.risk_probability:.3f} {row.file_path}")
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
