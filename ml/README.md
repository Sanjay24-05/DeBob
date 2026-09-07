# DeBob ML workflow

This directory adds file-level risk prediction and clustering using `.debob/context.db`.
The label is a proxy: a file is risky when its churn is at or above the 75th-percentile
threshold, or when a matching Git commit subject contains `fix`, `bug`, `hotfix`, or `patch`.
It is not ground-truth defect data.

On Windows, install dependencies with:

```powershell
powershell -ExecutionPolicy Bypass -File .\ml\install.ps1
```

Then export and label the current graph:

```powershell
.\ml\.venv\Scripts\python.exe .\ml\export_features.py
.\ml\.venv\Scripts\python.exe .\ml\label.py
.\ml\.venv\Scripts\python.exe .\ml\train_models.py
.\ml\.venv\Scripts\python.exe .\ml\predict_risk.py --model .\ml\artifacts\models\random_forest.joblib --output .\ml\artifacts\predictions.json --top 10
.\ml\.venv\Scripts\python.exe .\ml\export_presentation.py
```

Training writes five-fold metrics, pooled top-K ranking metrics, and an ablation chart.
Prediction rows include influential signal reasons; each output gets a `.meta.json` sidecar
with model and graph provenance. Reasons are not causal explanations. Generated CSVs and
model artifacts belong under `ml/artifacts/` and are ignored by Git. The presentation
export creates model comparison, churn ablation, top-K ranking, confusion matrix, feature
importance, clustering, and dendrogram PNGs under `ml/artifacts/presentation/`.
