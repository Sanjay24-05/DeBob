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

Training uses shuffled, stratified five-fold cross-validation with seed `42`. Risky-class F1
and recall are primary metrics. The experiment is reported both with and without `churn_score`.
The churn-inclusive experiment is expected to be optimistic because churn helps define the
label. The no-churn experiment reduces direct feature leakage but still uses the same weak label.

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
0.985 src/engine/index.ts
  - high commit-touch churn (12)
  - high import fan-in (8)
```

The current repository is small, so high variance and snapshot-specific behavior are expected.
A production version should use historical snapshots, future bug-fix labels, temporal validation,
and cross-repository testing.
