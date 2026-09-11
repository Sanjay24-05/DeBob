# DeBob ML Risk Report

This report documents DeBob's proof-of-concept file-risk prediction experiment.

All figures below were reproduced locally from `ml/artifacts/models/metrics.json` and
`ml/artifacts/clusters/cluster_metrics.csv`. Run provenance: 43 training rows, class counts
`{0: 25, 1: 18}`, seed `42`, 5 folds, Python 3.11.9, scikit-learn 1.9.1, graph head commit
`9da5a73`. Regenerating the graph changes these numbers — `layer` is a model feature and
churn moves with new commits — so re-run the pipeline after any `debob init` before quoting
them.

## Dataset and label

Each row is one **source** file node from `.debob/context.db`. `classify_file` in
`ml/common.py` buckets every file into `source`, `config`, `documentation`,
`dependency_metadata`, or `generated_or_artifact`, and only `source` rows enter training.
This matters: the repository's highest-churn files are `PROGRESS.md`, `README.md`, and
`package-lock.json`, all of which the proxy label would otherwise mark risky, teaching the
model to recognise changelogs and lockfiles rather than code. Of 69 file nodes, 43 are
source; 21 documentation, 3 dependency metadata, and 2 config are excluded.

Features are churn score, author count, import fan-in, import fan-out, physical lines of
code, and architectural layer. The risk label is a proxy: a meaningful source file is risky
when it sits at or above the 75th-percentile churn threshold (3.0 in this run), or when a
Git commit whose subject contains `fix`, `bug`, `hotfix`, or `patch` touched it.

This is not ground-truth bug data. With 43 rows the fold-to-fold variance is large, and the
standard deviations below should be read as the headline result rather than the means.

## Reproduction

See [ml/README.md](../ml/README.md). Note that `cluster_models.py` must run before
`export_presentation.py`, which reads `ml/artifacts/clusters/cluster_metrics.csv`.

## Supervised results

Shuffled, stratified 5-fold cross-validation, seed `42`. Values are mean +/- standard
deviation across folds; ROC-AUC and the ranking metrics are pooled out-of-fold.

Precision@5 is the primary operational metric because the product ranks a short list for
human review. It is also a five-item statistic on a 43-row dataset — see the caveat below.

### Operational models (without churn)

`churn_score` is excluded because it is part of the label definition. These are the only
numbers that describe predictive behaviour.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | P@5 | P@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated SVM | 0.861 +/- 0.092 | 0.850 +/- 0.137 | **0.833 +/- 0.156** | **0.831 +/- 0.104** | 0.880 | 0.80 | 0.70 |
| Logistic Regression | 0.811 +/- 0.107 | 0.767 +/- 0.137 | 0.767 +/- 0.137 | 0.767 +/- 0.137 | **0.923** | **1.00** | **0.90** |
| Decision Tree | 0.792 +/- 0.144 | 0.783 +/- 0.217 | 0.733 +/- 0.181 | 0.748 +/- 0.161 | 0.782 | 0.60 | 0.80 |
| Random Forest | 0.792 +/- 0.144 | 0.750 +/- 0.144 | 0.733 +/- 0.253 | 0.733 +/- 0.189 | 0.827 | **1.00** | 0.80 |
| DummyClassifier | 0.583 +/- 0.038 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.500 | 0.60 | 0.40 |

### Model selection rationale

The pipeline recommends **Logistic Regression**: it ties Random Forest on precision@5 (1.00)
and wins the F1 tiebreak, and it also has the best ROC-AUC (0.923) and precision@10 (0.90).

That recommendation deserves scepticism. The Calibrated SVM is better on F1 (0.831 vs
0.767), recall (0.833 vs 0.767), and accuracy (0.861 vs 0.811), and loses only on
precision@5 — a metric computed over five files. The dummy classifier, which predicts one
class for everything and has zero recall, scores precision@5 = 0.60 purely from tie
ordering. A single file changing position flips the ranking. Precision@5 is the right
*objective* for a review-list product, but on 43 rows it is not a reliable *discriminator*
between models, and the honest reading is that SVM and Logistic Regression are not
separated by this dataset.

### Leakage diagnostic (with churn)

Retained only to demonstrate label leakage. These are not peer experiments and must not be
compared against the operational table.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | P@5 |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 | 1.00 |
| Random Forest | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 | 1.00 |
| Logistic Regression | 0.906 +/- 0.053 | 0.910 +/- 0.124 | 0.883 +/- 0.162 | 0.881 +/- 0.074 | 0.957 | 1.00 |
| Calibrated SVM | 0.861 +/- 0.092 | 0.850 +/- 0.137 | 0.833 +/- 0.156 | 0.831 +/- 0.104 | 0.950 | 1.00 |

Two models reach a perfect score across every fold. That is the expected result when a
feature reconstructs the label, not evidence of predictive power, and it is why
`random_forest_with_churn_diagnostic.joblib` is tagged `diagnostic_only` and prints a
warning when loaded for prediction.

### Baselines

| Baseline | Accuracy | Precision | Recall | F1 | ROC-AUC | P@5 | P@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw churn heuristic | 0.925 +/- 0.168 | 0.800 +/- 0.447 | 0.800 +/- 0.447 | 0.800 +/- 0.447 | 1.000 | 1.00 | 1.00 |
| DummyClassifier | 0.583 +/- 0.038 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.500 | 0.60 | 0.40 |

The raw churn heuristic scores higher accuracy (0.925) than any learned model and a perfect
ROC-AUC. **This is not evidence that a one-line rule beats machine learning.** The heuristic
thresholds churn, and churn defines the upper-quartile half of the label, so it has partial
access to the answer — the same leakage as the `with_churn` experiments, and it belongs with
them rather than alongside the operational models. Its +/- 0.447 standard deviation shows how
unstable it is despite the high mean: near-perfect on some folds, poor on others.

The fair comparison for the operational models is the **DummyClassifier**, which achieves
0.583 accuracy with zero recall on this 58/42 split. Against that floor, operational F1
scores of 0.73-0.83 represent real signal carried by author count, import fan-in/fan-out,
lines of code, and layer.

### Review-priority ranking

Pooled out-of-fold, operational (without-churn) models:

| Model | Precision@5 | Recall@5 | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.00 | 0.278 | 0.90 | 0.500 |
| Random Forest | 1.00 | 0.278 | 0.80 | 0.444 |
| Calibrated SVM | 0.80 | 0.222 | 0.70 | 0.389 |
| Decision Tree | 0.60 | 0.167 | 0.80 | 0.444 |
| DummyClassifier | 0.60 | 0.167 | 0.40 | 0.222 |

These answer a practical question — how many proxy-risk files appear near the top of a short
review list — not how the model would perform on a future repository state.

## Unsupervised results

| Algorithm | Silhouette | Davies-Bouldin | Clusters | Noise |
|---|---:|---:|---:|---:|
| DBSCAN | **0.342** | **0.907** | 3 | 12 |
| K-Means | 0.305 | 1.275 | 3 | 0 |
| Agglomerative | 0.294 | 1.239 | 3 | 0 |

DBSCAN shows the best separation scores, but it assigns 12 of 43 files to noise and those
points are excluded from its silhouette calculation — it is scoring a cleaner subset than the
other two, so the comparison is not like-for-like.

K-Means separates risk substantially better than earlier runs on the unfiltered dataset:

| K-Means cluster | Stable | Risky |
|---|---:|---:|
| 0 | 23 | 2 |
| 1 | 2 | 11 |
| 2 | 0 | 5 |

Cluster 0 is almost entirely stable and clusters 1 and 2 are almost entirely risky, so the
graph features carry risk signal without ever seeing the label.

The layer crosstab tells a different story. Cluster 1 is mostly `data` (8) and `infra` (3),
but cluster 0 mixes `business` (10), `test` (7), and `unclassified` (5). Clusters partially
track architecture but do not reproduce it one-cluster-per-layer. This supports using `layer`
as a supervised feature rather than treating clusters as architecture labels — a negative
result that justifies a design decision.

## Limitations

- **43 rows, 18 positive.** Fold-to-fold standard deviations reach +/- 0.25. No result here
  separates models with confidence.
- **The label is a proxy**, and churn defines most of it. Every number measures agreement
  with that heuristic, not observed defects.
- **Single snapshot, no temporal validation.** `last_modified_at` is exported but unused. All
  rows come from one point in the repository's history, so this is not future-defect prediction.
- **One repository.** No cross-repository test, so nothing here generalises.
- **Precision@5 is a five-item statistic** and is not a reliable discriminator at this scale.

A production version would use historical snapshots with later bug-fix commits as labels,
temporal rather than stratified validation, and multiple repositories with
leave-one-repository-out testing. See the [model card](ml-model-card.md) for intended use and
explanation caveats.
