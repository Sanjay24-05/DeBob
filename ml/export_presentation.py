"""Export presentation-ready PNG charts from saved DeBob ML artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["logistic_regression", "decision_tree", "random_forest", "svm_rbf"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm_rbf": "Calibrated SVM",
}


def load_metrics(models_dir: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for experiment, experiment_rows in payload.get("experiments", {}).items():
        rows.extend(experiment_rows)
    for experiment, experiment_rows in payload.get("diagnostic_experiments", {}).items():
        rows.extend(experiment_rows)
    return payload, pd.DataFrame(rows)


def save_model_comparison(frame: pd.DataFrame, output: Path) -> None:
    learned = frame[frame["model"].isin(MODEL_ORDER)]
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, metric, title in zip(
        axes,
        ("f1_mean", "recall_mean", "roc_auc_mean", "precision_at_5"),
        ("Risky-class F1", "Risky-class recall", "ROC-AUC", "Precision@5"),
    ):
        chart = learned.pivot(index="model", columns="experiment", values=metric).reindex(MODEL_ORDER)
        chart.index = [MODEL_LABELS[name] for name in chart.index]
        chart.plot(kind="bar", ax=axis, rot=30, color=["#1769aa", "#e07a5f"])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_ylabel("score")
        axis.legend(title="feature set")
    figure.suptitle("DeBob model comparison (recommended: Random Forest without churn)", fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_ablation(frame: pd.DataFrame, output: Path) -> None:
    learned = frame[frame["model"].isin(MODEL_ORDER)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, title in zip(axes, ("f1_mean", "recall_mean"), ("F1 ablation", "Recall ablation")):
        chart = learned.pivot(index="model", columns="experiment", values=metric).reindex(MODEL_ORDER)
        chart.index = [MODEL_LABELS[name] for name in chart.index]
        chart.plot(kind="bar", ax=axis, rot=30, color=["#1769aa", "#e07a5f"])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_ylabel("score")
        axis.legend(title="feature set")
    figure.suptitle("Effect of removing churn_score", fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_top_k(frame: pd.DataFrame, baseline: dict, output: Path) -> None:
    rows = frame[frame["model"].isin(MODEL_ORDER)].copy()
    rows["label"] = rows["experiment"].str.replace("_", " ") + " / " + rows["model"].map(MODEL_LABELS)
    baseline_row = pd.DataFrame([baseline])
    baseline_row["label"] = "Raw churn heuristic"
    rows = pd.concat([rows, baseline_row], ignore_index=True)
    labels = rows["label"].tolist()
    positions = np.arange(len(labels))
    width = 0.19
    figure, axis = plt.subplots(figsize=(14, 6))
    for offset, metric, title, color in [
        (-width, "precision_at_5", "Precision@5", "#2a9d8f"),
        (0, "recall_at_5", "Recall@5", "#e9c46a"),
        (width, "precision_at_10", "Precision@10", "#457b9d"),
        (2 * width, "recall_at_10", "Recall@10", "#e76f51"),
    ]:
        axis.bar(positions + offset, rows[metric].astype(float), width, label=title, color=color)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title("Review-priority ranking quality (★ = recommended)")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_confusion(models_dir: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, experiment, title in zip(axes, ("with_churn", "without_churn"), ("With churn", "Without churn")):
        path = models_dir / f"confusion_{experiment}_random_forest.json"
        matrix = np.asarray(json.loads(path.read_text(encoding="utf-8")))
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1], ["Stable", "Risky"])
        axis.set_yticks([0, 1], ["Stable", "Risky"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    figure.suptitle("Random Forest confusion matrices", fontweight="bold")
    figure.savefig(output_dir / "confusion_matrices_random_forest.png", dpi=180)
    plt.close(figure)


def save_feature_importance(models_dir: Path, output: Path) -> None:
    # Prefer the diagnostic bundle (has churn feature importance); fall back to operational
    diagnostic_path = models_dir / "random_forest_with_churn_diagnostic.joblib"
    operational_path = models_dir / "random_forest_without_churn.joblib"
    fallback_path = models_dir / "random_forest.joblib"
    if diagnostic_path.exists():
        bundle = joblib.load(diagnostic_path)
    elif fallback_path.exists():
        bundle = joblib.load(fallback_path)
    else:
        bundle = joblib.load(operational_path)
    importance = pd.Series(bundle["feature_importance"]).sort_values()
    axis = importance.plot(kind="barh", figsize=(8, 4), color="#1769aa", title="Random Forest global feature importance")
    axis.set_xlabel("importance")
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=180)
    plt.close(axis.figure)


def save_clustering(clusters_dir: Path, output: Path) -> None:
    metrics = pd.read_csv(clusters_dir / "cluster_metrics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics.plot.bar(x="model", y="silhouette", ax=axes[0], legend=False, color="#2a9d8f", title="Silhouette score")
    metrics.plot.bar(x="model", y="davies_bouldin", ax=axes[1], legend=False, color="#e76f51", title="Davies-Bouldin index")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("higher is better")
    axes[1].set_ylabel("lower is better")
    figure.suptitle("Unsupervised module clustering", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def export(models_dir: Path, clusters_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, frame = load_metrics(models_dir)
    baseline = payload["baselines"]["raw_churn_heuristic"]
    save_model_comparison(frame, output_dir / "01_model_comparison.png")
    save_ablation(frame, output_dir / "02_churn_ablation.png")
    save_top_k(frame, baseline, output_dir / "03_top_k_ranking.png")
    save_confusion(models_dir, output_dir)
    save_feature_importance(models_dir, output_dir / "05_feature_importance.png")
    save_clustering(clusters_dir, output_dir / "06_clustering_metrics.png")
    dendrogram = clusters_dir / "agglomerative_dendrogram.png"
    if dendrogram.exists():
        shutil.copyfile(dendrogram, output_dir / "07_agglomerative_dendrogram.png")
    print(f"Presentation assets exported to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("ml/artifacts/models"))
    parser.add_argument("--clusters", type=Path, default=Path("ml/artifacts/clusters"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/presentation"))
    args = parser.parse_args()
    export(args.models, args.clusters, args.output)


if __name__ == "__main__":
    main()
