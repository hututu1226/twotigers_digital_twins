# Error Mining

## 直接结论

主要问题不是所有样本都差一点，而是一批“更难猜”的样本贡献了大部分误差。它们更常出现在 BS1、离训练支撑点较远或局部样本稀疏的位置。

例如，最差 5% 只有约 28 个验证样本，却贡献了 `67.27%` 的总误差能量。只平均看 565 个样本会掩盖这个现象。

## Spatial Slices

| Slice | Score |
|---|---:|
| BS0 | 0.67759 |
| BS1 | 0.57600 |
| Near support | 0.66749 |
| Far from support | 0.58647 |
| High local density | 0.66959 |
| Low local density | 0.59344 |

- Nearest-support distance vs sample Score, Spearman: `-0.1874`
- Local density vs sample Score, Spearman: `+0.2030`
- Target power vs sample NMSE, Spearman: `-0.5506`
- Worst 1% contribution to global error energy: `34.38%`
- Worst 5% contribution to global error energy: `67.27%`

## Metric Interpretation

- PAS is the weakest headline component and needs better angular structure.
- PDP is the strongest component and should be preserved rather than rebuilt from scratch.
- NMSE is disproportionately affected by a small number of high-energy misses.
- There are 35 true outage samples and only 2 hard predicted outages, but true-outage hard-zero oracle improves Score by only `0.000048`; existing predictions at most missed outages are already very small.

## Consequence

The next model should retain the complete stable seed latent and predict a small correction conditioned on local support. A model that recreates the entire channel from a few hundred dimensions would discard information that is already correct.
