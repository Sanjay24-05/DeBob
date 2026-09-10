# DeBob ML workflow

This directory adds file-level risk prediction and clustering using `.debob/context.db`.
The operational ranking is source-only by default. TypeScript, JavaScript, and Python files are
classified as `source`; Markdown and HTML are `documentation`; package manifests and lockfiles
are `dependency_metadata`; config files are `config`; and generated, build, coverage, `.debob`,
and artifact paths are `generated_or_artifact`. Non-source categories are excluded from the
default training and prediction candidate set. Use `--scope all` only for diagnostics.

The label is a proxy, not ground truth: a meaningful source file is risky when it was touched by
a Git commit whose subject contains `fix`, `bug`, `hotfix`, or `patch`, or when it is in the
high-churn group. Churn alone cannot label empty or non-source files. Labels include the matching
reason columns so the training target can be audited.

On Windows, install dependencies with:

```powershell
powershell -ExecutionPolicy Bypass -File .\ml\install.ps1
```

Then export and label the current graph:

```powershell
.\ml\.venv\Scripts\python.exe .\ml\export_features.py
.\ml\.venv\Scripts\python.exe .\ml\label.py
.\ml\.venv\Scripts\python.exe .\ml\train_models.py
.\ml\.venv\Scripts\python.exe .\ml\predict_risk.py --output .\ml\artifacts\predictions.json
.\ml\.venv\Scripts\python.exe .\ml\export_presentation.py
```

### Diagnostic model

Training also saves `random_forest_with_churn_diagnostic.joblib`. This model includes
`churn_score` as a feature, which is part of the proxy label definition. It exists only to
detect label leakage — loading it for prediction prints a warning. Do not use it for
operational ranking; use the default `random_forest_without_churn.joblib` instead.

The exporter writes `features.csv.meta.json` with candidate scope, total graph file nodes, and
excluded counts by category. Prediction writes the same scope and exclusion information in its
sidecar metadata. Models refuse to train on mixed categories, and prediction refuses a model whose
candidate scope does not match the requested scope.

Training writes five-fold metrics, pooled precision, recall, F1, precision@5, recall@5, a raw
churn baseline, class counts, candidate scope, and a small-data warning when the source dataset is
too small for reliable generalization. The `with_churn` experiment is retained as a leakage
diagnostic only; the `without_churn` model is the operational default because churn is part of the
proxy label. Primary metrics are precision@5, F1, and recall. These metrics measure agreement
with the proxy label, not real defect prediction accuracy. A small repository should not make
general accuracy claims from these results.
Prediction rows include influential signal reasons; each output gets a `.meta.json` sidecar
with model and graph provenance. Reasons are not causal explanations. Generated CSVs and
model artifacts belong under `ml/artifacts/` and are ignored by Git. The presentation
export creates model comparison, churn ablation, top-K ranking, confusion matrix, feature
importance, clustering, and dendrogram PNGs under `ml/artifacts/presentation/`.
