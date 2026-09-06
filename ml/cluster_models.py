"""Compare K-Means, DBSCAN, and Agglomerative file clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import NUMERIC_FEATURES, LAYER_FEATURE

SEED = 42


def cluster(input_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(input_path)
    if len(data) < 3:
        raise ValueError("At least three rows are required for clustering.")
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    matrix = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("layer", categorical, [LAYER_FEATURE])]).fit_transform(data)
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {"kmeans": KMeans(n_clusters=min(3, len(data) - 1), random_state=SEED, n_init=10), "agglomerative": AgglomerativeClustering(n_clusters=min(3, len(data) - 1)), "dbscan": DBSCAN(eps=1.5, min_samples=2)}
    summaries = []
    for name, model in models.items():
        labels = model.fit_predict(matrix)
        data[f"cluster_{name}"] = labels
        valid = set(labels) - {-1}
        scores = {"silhouette": None, "davies_bouldin": None}
        if len(valid) >= 2 and len(valid) < len(labels):
            mask = labels != -1
            scores = {"silhouette": silhouette_score(matrix[mask], labels[mask]), "davies_bouldin": davies_bouldin_score(matrix[mask], labels[mask])}
        elif len(valid) >= 2:
            scores = {"silhouette": silhouette_score(matrix, labels), "davies_bouldin": davies_bouldin_score(matrix, labels)}
        summaries.append({"model": name, **scores, "clusters": len(valid), "noise_points": int(sum(labels == -1))})
        data[["file_path", "layer", "risky", f"cluster_{name}"]].to_csv(output_dir / f"assignments_{name}.csv", index=False)
        pd.crosstab(data[f"cluster_{name}"], data["layer"]).to_csv(output_dir / f"crosstab_{name}_layer.csv")
        pd.crosstab(data[f"cluster_{name}"], data["risky"]).to_csv(output_dir / f"crosstab_{name}_risk.csv")

    hierarchy = linkage(matrix, method="ward")
    plt.figure(figsize=(8, 4))
    plt.title("DeBob module hierarchy")
    dendrogram(hierarchy, no_labels=True)
    plt.xlabel("files")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig(output_dir / "agglomerative_dendrogram.png")
    plt.close()
    (output_dir / "cluster_metrics.json").write_text(json.dumps({"seed": SEED, "rows": summaries}, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "cluster_metrics.csv", index=False)
    print(f"Compared {len(models)} clustering algorithms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ml/artifacts/labeled_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/clusters"))
    args = parser.parse_args()
    cluster(args.input, args.output_dir)


if __name__ == "__main__":
    main()
