# DeBob ML Risk Report

This report documents DeBob's proof-of-concept file-risk prediction experiment.

## Dataset and label

Each row represents one file node from `.debob/context.db`. Features are churn score,
author count, import fan-in, import fan-out, physical lines of code, and architectural
layer. The risk label is a proxy: churn at or above the 75th percentile, or a file touched
by a commit whose subject contains `fix`, `bug`, `hotfix`, or `patch`.

This is not ground-truth bug data. The repository is small, and the results should not be
generalized to other projects without larger, time-aware validation.

## Reproduction

See [ml/README.md](../ml/README.md). Generated metrics, confusion matrices, model bundles,
cluster assignments, crosstabs, and plots are written under `ml/artifacts/`.

## Supervised results

The run used seed `42` and an 80/20 stratified split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 0.90 | 1.00 | 0.75 | 0.857 | 1.00 |
| RBF SVM | 0.80 | 0.75 | 0.75 | 0.75 | 0.958 |

Decision Tree is selected as the winner by the configured primary metric, F1. Random
Forest ties it on this small split. The perfect tree and forest scores must not be read as
evidence of general defect-prediction accuracy: `churn_score` is both an input feature and
part of the proxy-label rule, so the models can reconstruct the label directly. A stronger
follow-up experiment should exclude label-defining features or use time-based validation.

## Unsupervised results

| Algorithm | Silhouette | Davies-Bouldin | Clusters | Noise |
|---|---:|---:|---:|---:|
| K-Means | 0.271 | 1.346 | 3 | 0 |
| Agglomerative | 0.267 | 1.337 | 3 | 0 |
| DBSCAN | 0.344 | 0.797 | 8 | 6 |

DBSCAN produced the strongest separation scores, but it also identified six noise files
and eight small clusters. K-Means and Agglomerative produced three broad clusters and did
not cleanly rediscover architectural layers. For example, K-Means cluster 1 contains most
of the stable files (21 stable versus 2 risky), while cluster 0 contains mostly risky files
(12 risky versus 4 stable), suggesting that the graph features carry some risk signal.
The layer crosstabs show mixed architectural membership rather than a one-cluster-per-layer
structure. This supports using `layer` as a feature, not treating unsupervised clusters as
replacement architecture labels.

Generated confusion matrices, crosstabs, model bundles, and plots are in `ml/artifacts/`.
