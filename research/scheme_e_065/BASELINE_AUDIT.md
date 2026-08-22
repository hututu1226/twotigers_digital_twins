# Baseline Audit

## 直接结论

权威 Scheme E V4 严格 Fold0 基线已被重新生成和复评，统一 evaluator 得到 `0.627089`，与历史报告 `0.62705` 只差约 `0.000039`。保存为 `.npy` 再读取后分数不变，因此当前约 `0.023` 的差距不是文件保存或 evaluator 漂移造成的。

## 权威产物

- Git branch: `codex/0821_schemeE_v3`
- Audit code commit: `9102bfe`
- Config: `schemeE_spectral_gaussian_hybrid/configs/v4_fold_best.json`
- Checkpoint: `schemeE_spectral_gaussian_hybrid/artifacts/v4/fold0_attempt1/hybrid/best.pt`
- Checkpoint size: about 42.4 MB
- Policy: `schemeE_spectral_gaussian_hybrid/reports/generated/v4_attempt1_policy.json`
- Output projection: `schemeE_spectral_gaussian_hybrid/reports/generated/v4_attempt1_output_projection.json`
- Rebuilt prediction: `research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy`
- Per-sample diagnostics: `research/scheme_e_065/PER_SAMPLE_METRICS.npz`

## Split

- Fold: `0`
- Fold0 train: `3435`
- Fold0 validation: `565`
- Validation selection: `metadata["validation_masks"][0]`
- Base-station identity is preserved; no cross-BS neighbor support is used.

## Evaluator

- Metric primitives and official formula: `scheme_e/metrics.py`
- Canonical row-level aggregation: `scheme_e/diagnostics.py`
- Audit/replay entry point: `scripts/audit_scheme_e_065.py`

The confirmed formula is:

`Score = 0.4 * PAS + 0.4 * PDP + 0.2 / (1 + NMSE)`

PAS/PDP exclude true outage rows. NMSE is one global error-energy sum divided by one global target-energy sum; it is not the arithmetic mean of sample NMSE.

## Reproduced Result

| Representation | PAS | PDP | NMSE | Score | Samples |
|---|---:|---:|---:|---:|---:|
| In-memory final | 0.567081 | 0.758360 | 1.063711 | 0.627089 | 565 |
| Saved/reloaded NPY | 0.567081 | 0.758360 | 1.063711 | 0.627089 | 565 |

## Reproduction

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
bash scripts/run_scheme_e_065_level0.sh
```

This is an offline Fold0 result. The official online score remains `0.59`.
