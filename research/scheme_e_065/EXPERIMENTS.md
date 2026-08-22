# Scheme E Strict Fold0 Experiments

Fold0 scores in this file are offline validation results. They are not official online scores.

| ID | Hypothesis | Core change | Probe result | Fold0 Score | PAS | PDP | NMSE | Delta | Decision | Cost |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| L0-001 | The authoritative V4 baseline and its saved NPY can be reproduced by one canonical evaluator. | Audit only; no model change. | RUNNING | - | - | - | - | - | PENDING | 0 GPU h |

## L0-001

### Hypothesis

The selected V4 artifacts reproduce `0.62705` before and after saving the Fold0 prediction.

### Evidence

The previous strict Fold0 report recorded PAS `0.56696`, PDP `0.75838`, NMSE `1.06373`, and Score `0.62705`, while a newer adaptive Teacher improved an intermediate PAS but reduced the final Score to `0.62473`.

### Minimal Experiment

Run the fixed V4 checkpoint through every output stage, save the final Fold0 prediction, reload it, and evaluate all stages with both the canonical row-level implementation and the existing streaming evaluator. Also compute per-sample errors and diagnostic-only scale/expert oracles.

### Expected Signal

All baseline measurements remain within `0.0005` of `0.62705`; any Teacher improvement that is lost later can be localized to a specific stage and sample slice.

### Abort Rule

If any in-memory or saved-NPY baseline Score differs from `0.62705` by more than `0.0005`, stop before adaptive diagnostics or new training and repair the evaluator or artifact selection.
