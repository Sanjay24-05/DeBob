"""Train and compare four supervised file-risk classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from common import NUMERIC_FEATURES, LAYER_FEATURE

SEED = 42


def train(input_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(input_path)
    required = [*NUMERIC_FEATURES, LAYER_FEATURE, "risky"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(data) < 10 or data["risky"].nunique() < 2:
        raise ValueError("At least 10 rows and both risk classes are required for stratified training.")

    x = data[[*NUMERIC_FEATURES, LAYER_FEATURE]]
    y = data["risky"].astype(int)
    if y.value_counts().min() < 2:
        raise ValueError("Each risk class needs at least two rows for a stratified split.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("layer", categorical, [LAYER_FEATURE])])
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=SEED),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probabilities)
        except ValueError:
            auc = None
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        metrics.append({"model": name, "accuracy": accuracy_score(y_test, predicted), "precision": precision_score(y_test, predicted, zero_division=0), "recall": recall_score(y_test, predicted, zero_division=0), "f1": f1_score(y_test, predicted, zero_division=0), "roc_auc": auc})
        (output_dir / f"confusion_{name}.json").write_text(json.dumps(matrix.tolist(), indent=2), encoding="utf-8")
        joblib.dump({"model": pipeline, "features": [*NUMERIC_FEATURES, LAYER_FEATURE], "seed": SEED, "model_name": name}, output_dir / f"{name}.joblib")

    metrics.sort(key=lambda row: row["f1"], reverse=True)
    (output_dir / "metrics.json").write_text(json.dumps({"seed": SEED, "primary_metric": "f1", "rows": metrics}, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    print(f"Trained {len(models)} models; winner by F1: {metrics[0]['model']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ml/artifacts/labeled_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/models"))
    args = parser.parse_args()
    train(args.input, args.output_dir)


if __name__ == "__main__":
    main()
