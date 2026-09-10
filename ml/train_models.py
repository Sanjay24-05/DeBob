"""Compare supervised file-risk models with five-fold cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn

from common import NUMERIC_FEATURES, LAYER_FEATURE, OPERATIONAL_MODEL_NAME, DIAGNOSTIC_MODEL_NAME, DEFAULT_TOP_K

SEED = 42
FOLDS = 5
LABEL_CAVEAT = "Labels are proxy heuristics, not observed defects; results are not general accuracy estimates."


def make_pipeline(estimator: object, numeric_features: list[str]) -> Pipeline:
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("layer", categorical, [LAYER_FEATURE]),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def model_factories() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced"),
        "svm_rbf": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", probability=False, random_state=SEED),
            cv=3,
            ensemble=False,
        ),
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    }


def metric_values(actual: pd.Series, predicted: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    try:
        roc_auc = float(roc_auc_score(actual, probabilities))
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": roc_auc,
    }


def ranking_values(actual: list[int], probabilities: list[float]) -> dict[str, float]:
    actual_array = np.asarray(actual)
    probability_array = np.asarray(probabilities)
    order = np.argsort(-probability_array, kind="stable")
    positive_count = int(actual_array.sum())
    values: dict[str, float] = {}
    for k in (5, 10):
        selected = order[:min(k, len(order))]
        hits = int(actual_array[selected].sum())
        values[f"precision_at_{k}"] = hits / len(selected) if len(selected) else 0.0
        values[f"recall_at_{k}"] = hits / positive_count if positive_count else 0.0
    return values


def summarize(
    experiment: str,
    model_name: str,
    fold_metrics: list[dict[str, float]],
    matrix: np.ndarray,
    actual: list[int],
    candidate_scope: str,
) -> dict[str, object]:
    result: dict[str, object] = {"experiment": experiment, "model": model_name, "folds": len(fold_metrics)}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = np.array([row[metric] for row in fold_metrics], dtype=float)
        result[f"{metric}_mean"] = float(np.nanmean(values))
        result[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    result["confusion_matrix"] = matrix.tolist()
    result["candidate_scope"] = candidate_scope
    result["class_counts"] = {"0": actual.count(0), "1": actual.count(1)}
    return result


def evaluate_experiment(
    data: pd.DataFrame,
    numeric_features: list[str],
    experiment: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidate_scope: str,
) -> tuple[list[dict[str, object]], dict[str, Pipeline]]:
    x = data[[*numeric_features, LAYER_FEATURE]]
    y = data["risky"].astype(int)
    results: list[dict[str, object]] = []
    final_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in model_factories().items():
        fold_metrics: list[dict[str, float]] = []
        pooled_actual: list[int] = []
        pooled_predicted: list[int] = []
        pooled_probabilities: list[float] = []
        for train_indices, test_indices in folds:
            fold_estimator = clone(estimator)
            pipeline = make_pipeline(fold_estimator, numeric_features)
            pipeline.fit(x.iloc[train_indices], y.iloc[train_indices])
            predicted = pipeline.predict(x.iloc[test_indices])
            probabilities = pipeline.predict_proba(x.iloc[test_indices])[:, 1]
            fold_metrics.append(metric_values(y.iloc[test_indices], predicted, probabilities))
            pooled_actual.extend(y.iloc[test_indices].tolist())
            pooled_predicted.extend(predicted.tolist())
            pooled_probabilities.extend(probabilities.tolist())

        matrix = confusion_matrix(pooled_actual, pooled_predicted, labels=[0, 1])
        result = summarize(experiment, model_name, fold_metrics, matrix, pooled_actual, candidate_scope)
        result.update(ranking_values(pooled_actual, pooled_probabilities))
        results.append(result)
        final_pipeline = make_pipeline(estimator, numeric_features)
        final_pipeline.fit(x, y)
        final_pipelines[model_name] = final_pipeline

    churn_values = data["churn_score"].astype(float).to_numpy()
    baseline_folds: list[dict[str, float]] = []
    pooled_actual = []
    pooled_predicted = []
    pooled_probabilities = []
    for train_indices, test_indices in folds:
        train_churn = np.sort(churn_values[train_indices])
        threshold = train_churn[max(0, int(0.75 * len(train_churn)) - 1)]
        predicted = ((churn_values[test_indices] > 0) & (churn_values[test_indices] >= threshold)).astype(int)
        probabilities = np.clip(churn_values[test_indices] / max(threshold, 1), 0, 1)
        actual = y.iloc[test_indices]
        baseline_folds.append(metric_values(actual, predicted, probabilities))
        pooled_actual.extend(actual.tolist())
        pooled_predicted.extend(predicted.tolist())
        pooled_probabilities.extend(probabilities.tolist())
    baseline = summarize(
        experiment,
        "raw_churn_heuristic",
        baseline_folds,
        confusion_matrix(pooled_actual, pooled_predicted, labels=[0, 1]),
        pooled_actual,
        candidate_scope,
    )
    baseline.update(ranking_values(pooled_actual, pooled_probabilities))
    results.append(baseline)
    return results, final_pipelines


def train(input_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(input_path)
    required = [*NUMERIC_FEATURES, LAYER_FEATURE, "file_category", "meaningful_code", "risky"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if set(data["file_category"].dropna().unique()) != {"source"} or not (data["meaningful_code"].astype(str) == "1").all():
        raise ValueError("Training input must contain only meaningful source files. Regenerate features with --scope source.")
    if len(data) < 10 or data["risky"].nunique() < 2:
        raise ValueError("At least 10 rows and both risk classes are required for training.")
    if int(data["risky"].value_counts().min()) < FOLDS:
        raise ValueError(f"Each risk class needs at least {FOLDS} rows for {FOLDS}-fold stratified validation.")

    y = data["risky"].astype(int)
    small_data_warning = (
        f"Small dataset: {len(data)} source candidates with class counts "
        f"0={int((y == 0).sum())}, 1={int((y == 1).sum())}; metrics are not general accuracy estimates."
        if len(data) < 50 or int(y.value_counts().min()) < 10
        else None
    )
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(data, y))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    provenance = {
        "generated_at": generated_at,
        "input_sha256": input_sha256,
        "training_rows": len(data),
        "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "seed": SEED,
        "folds": FOLDS,
        "splitter": "StratifiedKFold(shuffle=True)",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "db_schema_versions": sorted(str(value) for value in data["db_schema_version"].dropna().unique()) if "db_schema_version" in data else [],
        "db_head_commits": sorted(str(value) for value in data["db_head_commit"].dropna().unique()) if "db_head_commit" in data else [],
    }

    inclusive_results, inclusive_pipelines = evaluate_experiment(data, list(NUMERIC_FEATURES), "with_churn", folds, "source")
    no_churn_features = [feature for feature in NUMERIC_FEATURES if feature != "churn_score"]
    no_churn_results, no_churn_pipelines = evaluate_experiment(data, no_churn_features, "without_churn", folds, "source")
    results = inclusive_results + no_churn_results

    for result in results:
        safe_name = f"{result['experiment']}_{result['model']}"
        (output_dir / f"confusion_{safe_name}.json").write_text(json.dumps(result["confusion_matrix"], indent=2), encoding="utf-8")

    def bundle(pipeline: Pipeline, features: list[str], model_name: str, experiment: str) -> dict[str, object]:
        transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        aggregated: dict[str, float] = {feature: 0.0 for feature in features}
        for name, importance in zip(transformed_names, importances):
            feature = "layer" if "layer" in name else name.split("__")[-1]
            aggregated[feature] = aggregated.get(feature, 0.0) + float(importance)
        return {
            "model": pipeline,
            "features": [*features, LAYER_FEATURE],
            "seed": SEED,
            "model_name": model_name,
            "experiment": experiment,
            "candidate_scope": "source",
            "small_data_warning": small_data_warning,
            "label_caveat": LABEL_CAVEAT,
            "leakage_warning": "churn_score is present in both features and the proxy label; use the without_churn bundle for operational ranking." if experiment == "with_churn" else None,
            "training_rows": len(data),
            "feature_importance": dict(sorted(aggregated.items(), key=lambda item: item[1], reverse=True)),
            "feature_thresholds": {
                feature: float(data[feature].astype(float).quantile(0.75))
                for feature in features
            },
            "provenance": {**provenance, "features": [*features, LAYER_FEATURE], "experiment": experiment, "model_name": model_name},
        }

    joblib.dump({**bundle(inclusive_pipelines["random_forest"], list(NUMERIC_FEATURES), "random_forest", "with_churn"), "diagnostic_only": True}, output_dir / f"{DIAGNOSTIC_MODEL_NAME}.joblib")
    joblib.dump(bundle(no_churn_pipelines["random_forest"], no_churn_features, "random_forest_without_churn", "without_churn"), output_dir / f"{OPERATIONAL_MODEL_NAME}.joblib")

    frame = pd.DataFrame([{key: value for key, value in result.items() if key != "confusion_matrix"} for result in results])
    frame.sort_values(["experiment", "f1_mean"], ascending=[True, False]).to_csv(output_dir / "metrics.csv", index=False)
    serializable = [{key: value for key, value in result.items() if key != "confusion_matrix"} | {"confusion_matrix": result["confusion_matrix"]} for result in results]
    experiments = {
        experiment: [
            row for row in serializable
            if row["experiment"] == experiment and row["model"] != "raw_churn_heuristic"
        ]
        for experiment in ("with_churn", "without_churn")
    }
    baselines = next(
        (row for row in serializable if row["model"] == "raw_churn_heuristic"),
        None,
    )
    # Select the recommended operational model: precision@5 first, then F1, then recall
    operational = frame[
        (frame["experiment"] == "without_churn") &
        (frame["model"].isin(["logistic_regression", "decision_tree", "random_forest", "svm_rbf"]))
    ].copy()
    operational = operational.sort_values(
        ["precision_at_5", "f1_mean", "recall_mean"],
        ascending=[False, False, False],
    )
    recommended = operational.iloc[0]
    recommended_model = {
        "model": str(recommended["model"]),
        "experiment": "without_churn",
        "precision_at_5": float(recommended["precision_at_5"]),
        "f1_mean": float(recommended["f1_mean"]),
        "recall_mean": float(recommended["recall_mean"]),
        "rationale": "precision@5 is the primary metric because the product ranks a short top-list for human review.",
    }
    metrics_payload = {
        "seed": SEED,
        "folds": FOLDS,
        "primary_metrics": ["precision_at_5", "f1_mean", "recall_mean"],
        "recommended_model": recommended_model,
        "experiments": {
            "without_churn": experiments["without_churn"],
        },
        "diagnostic_experiments": {
            "with_churn": experiments["with_churn"],
        },
        "baselines": {"raw_churn_heuristic": baselines} if baselines else {},
        "candidate_scope": "source",
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "small_data_warning": small_data_warning,
        "label_caveat": LABEL_CAVEAT,
        "leakage_warning": "with_churn metrics are label-leakage diagnostics, not evidence of predictive accuracy; use without_churn for operational ranking.",
    }
    metrics_payload["provenance"] = provenance
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")

    learned = frame[frame["model"].isin(["logistic_regression", "decision_tree", "random_forest", "svm_rbf"])]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title in zip(axes, ("f1_mean", "recall_mean", "precision_at_5"), ("F1 by feature set", "Recall by feature set", "Precision@5 by feature set")):
        chart_data = learned.pivot(index="model", columns="experiment", values=metric)
        chart_data.plot(kind="bar", ax=axis, rot=25, title=title)
        axis.set_ylabel("score")
        axis.set_ylim(0, 1)
        axis.legend(title="experiment")
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_metrics.png", dpi=150)
    plt.close(figure)

    winners = frame[frame["model"].isin(["logistic_regression", "decision_tree", "random_forest", "svm_rbf"])]
    winners = winners.sort_values(["experiment", "precision_at_5", "f1_mean", "recall_mean"], ascending=[True, False, False, False]).groupby("experiment").first()
    print("Cross-validation complete")
    print(f"Recommended operational model: {recommended_model['model']} (precision@5={recommended_model['precision_at_5']:.2f})")
    for experiment, row in winners.iterrows():
        p_at_5 = row['precision_at_5'] if 'precision_at_5' in row else float('nan')
        print(f"  {experiment}: {row['model']} p@5={p_at_5:.2f}, F1={row['f1_mean']:.3f} +/- {row['f1_std']:.3f}, recall={row['recall_mean']:.3f} +/- {row['recall_std']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ml/artifacts/labeled_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/models"))
    args = parser.parse_args()
    train(args.input, args.output_dir)


if __name__ == "__main__":
    main()
