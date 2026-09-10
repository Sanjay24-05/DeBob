# DeBob ML Risk Model Card

## Intended use

Rank repository files for review and testing prioritization using signals already present in
DeBob's knowledge graph. The output is a triage aid for developers, not an automated defect
verdict.

## Non-intended use

Do not interpret a score as the probability that a file contains a bug. Do not use it for
personnel evaluation, release approval, or unattended code changes. The model has not been
validated across repositories or against a production bug tracker.

## Data and target

The current dataset has one row per file from `.debob/context.db`. Features include commit-touch
churn, author count, static import fan-in/fan-out, physical lines of code, and architectural
layer. A weak label marks files in the positive-churn upper quartile or files touched by commits
whose subject contains `fix`, `bug`, `hotfix`, or `patch`.

## Evaluation

Training uses shuffled, stratified five-fold cross-validation with seed `42`. Precision@5 is
the primary metric because the product ranks a short list for human review. Risky-class F1
and recall are secondary. The experiment is reported both with and without `churn_score`. The
churn-inclusive experiment is a leakage diagnostic only; the without-churn Random Forest is the
operational model.

## Output explanations

Prediction rows include influential signals such as high churn, high fan-in, many contributors,
or a large module. These are global Random Forest importance plus threshold-based indicators;
they are not causal explanations of an individual file's risk.

## Provenance

Model bundles record the input hash, feature schema, training row count, class counts, graph
schema/head metadata, package versions, seed, and fold count. Prediction commands write a
metadata sidecar next to the CSV or JSON output.

## Example

```text
0.985 bin/debob.ts
  - high import fan-out (8)
  - many contributors (4)
0.920 src/engine/index.ts
  - high import fan-in (6)
  - large module (340)
```

The `with_churn` diagnostic model (`random_forest_with_churn_diagnostic.joblib`) is retained
for leakage detection. Loading it for prediction will print a warning.

The current repository is small, so high variance and snapshot-specific behavior are expected.
A production version should use historical snapshots, future bug-fix labels, temporal validation,
and cross-repository testing.
