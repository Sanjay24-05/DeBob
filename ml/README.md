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
```

Generated CSVs and model artifacts belong under `ml/artifacts/` and are ignored by Git.
