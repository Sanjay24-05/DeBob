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

The revised run uses shuffled, stratified 5-fold cross-validation with seed `42`. Values
below are mean +/- standard deviation across folds. Risky-class F1 and recall are the
headline metrics; accuracy is included for context.

### With churn feature

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 |
| Random Forest | 0.978 +/- 0.050 | 1.000 +/- 0.000 | 0.900 +/- 0.224 | 0.933 +/- 0.149 | 1.000 |
| Logistic Regression | 0.958 +/- 0.058 | 1.000 +/- 0.000 | 0.833 +/- 0.236 | 0.893 +/- 0.153 | 0.971 |
| RBF SVM | 0.933 +/- 0.061 | 1.000 +/- 0.000 | 0.733 +/- 0.253 | 0.827 +/- 0.167 | 0.986 |
| DummyClassifier | 0.740 +/- 0.053 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.500 |

### Without churn feature

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.716 +/- 0.067 | 0.533 +/- 0.447 | 0.333 +/- 0.204 | 0.360 +/- 0.207 | 0.742 |
| RBF SVM | 0.673 +/- 0.080 | 0.067 +/- 0.149 | 0.100 +/- 0.224 | 0.080 +/- 0.179 | 0.290 |
| Random Forest | 0.693 +/- 0.099 | 0.597 +/- 0.372 | 0.533 +/- 0.274 | 0.461 +/- 0.094 | 0.739 |
| Decision Tree | 0.629 +/- 0.104 | 0.107 +/- 0.153 | 0.200 +/- 0.274 | 0.137 +/- 0.192 | 0.653 |
| DummyClassifier | 0.740 +/- 0.053 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.500 |

### Baselines

| Baseline | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Raw churn heuristic | 0.867 +/- 0.145 | 0.730 +/- 0.277 | 1.000 +/- 0.000 | 0.819 +/- 0.195 | 0.912 |

The churn-inclusive tree and forest scores are expected to be extremely strong because
`churn_score` is both an input and part of the proxy-label rule. The raw churn baseline
also performs strongly for the same reason. Removing `churn_score` controls for that direct
feature leakage: the best remaining model is Random Forest at risky-class F1
`0.461 +/- 0.094` and recall `0.533 +/- 0.274`. This comparison makes the leakage visible
and demonstrates that the non-churn features carry some, but substantially weaker, signal.

The cross-validation change improves variance estimation over the prior 80/20 split, but it
does not establish future defect prediction: all rows still come from one repository snapshot.
A future version should use historical snapshots and later bug-fix commits as labels.

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
