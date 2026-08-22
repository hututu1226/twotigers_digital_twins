# Low-Rank Residual Oracle

## 直接结论

低秩频谱残差有足够容量突破 `0.65`，因此值得训练一个可部署的系数预测器。这里不是把 30720 维 latent 压成 16 维再重建，而是完整保留原始 latent，只用 16 个数描述“还需要改多少”。

类比：原图完整保留，16 个数只控制一层小修图滤镜；不是拿 16 个数重新画整张图。

## Leakage Boundary

- PCA/SVD basis is fitted from Fold0-train non-outage residuals only.
- Fold0-train seed predictions are OOF and use self-excluded observed references.
- Fold0 target is used only to calculate diagnostic oracle coefficients and metrics.
- All oracle scores below are **DIAGNOSTIC ONLY - NOT DEPLOYABLE**.

## Explained Variance

| Cell / branch | rank8 | rank16 | rank32 | rank64 | rank128 |
|---|---:|---:|---:|---:|---:|
| BS0 train spectrum | 0.3078 | 0.4454 | 0.5930 | 0.7275 | 0.8355 |
| BS1 train spectrum | 0.3569 | 0.4946 | 0.6331 | 0.7587 | 0.8550 |
| Validation teacher spectrum | 0.3111 | 0.4265 | 0.5460 | - | - |
| Validation baseline spectrum | 0.2868 | 0.4003 | 0.5218 | - | - |

## Oracle Scores

| Starting output | Correction | Score |
|---|---|---:|
| Teacher seed AE roundtrip | none | 0.616381 |
| Teacher seed | rank8 spectrum+detail | 0.653821 |
| Teacher seed | rank16 spectrum | 0.671566 |
| Teacher seed | rank32 spectrum | 0.687886 |
| Teacher seed | rank64 spectrum | 0.697410 |
| Teacher seed | rank128 spectrum+detail | 0.704282 |
| V4 baseline AE roundtrip | none | 0.624347 |
| V4 baseline | rank8 spectrum | 0.660820 |
| V4 baseline | rank16 spectrum | 0.679030 |
| V4 baseline | rank32 spectrum | 0.695510 |
| V4 baseline | rank64 spectrum | 0.705733 |
| V4 baseline | rank128 spectrum+detail | 0.717948 |

## Decision

PROMOTE to Level 1 with rank16 spectrum-only correction. Detail correction is excluded from the first probe because its useful low-rank metric gain is much weaker. The inner probe must gain at least `0.004` before strict Fold0 is evaluated.
