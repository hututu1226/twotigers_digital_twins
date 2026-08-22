# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 初赛的水平/垂直边缘频谱交替投影可形成第三个结构不同的专家。 | L0-014 新基线=`0.631581`；L0-013 旧基线二专家 oracle=`0.648161`；当前 Scheme E 使用联合 2D PAS，而初赛使用独立 H/V marginals。 | 新候选使二专家 oracle `>0.66`。 | 必须先超过 `0.66` 才训练 gate。 | 固定初赛 k24/p2/8轮设置，使用双基站 quality gate 和现有 AI power。 | <=0.2 h | 二专家 oracle `<=0.65`。 | 仅过线后训练严格 OOF gate。 |
| 2 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量，但当前候选 Router OOF 增益为 0。 | `+0.003` 到 `+0.010`。 | 必须先有新的二专家 oracle `>=+0.010`。 | 新候选产生后先算二专家 oracle。 | <=0.2 h | 无新候选或 oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

## 已 DROP

- 当前四专家 routing：完美 router 也仅 `0.645999`。
- outage 阈值优化：真值 outage oracle 仅增加 `0.000048`。
- 单样本复数标量作为主路线：oracle 仅 `0.642992`。
- rank16 local-set coefficient predictor：inner 平均增益 `0.000000`，所有非零 alpha 均更差。
- rank16 residual candidate Router：inner spatial OOF 增益 `0.000000`，严格 Fold0 回退到 baseline。
- rank8 magnitude coefficient GP：inner gain=`-0.019753`，系数 skill=`-1.133`。
- query-only full-resolution magnitude CNN：一次 metric-aligned 修改后 inner 最佳仍为 epoch0。
- 未对齐的 local full-resolution residual 直接迁移：strict `0.615896`；仅作为互补专家保留。
- adaptive-prior Hybrid fine-tune：best epoch1，canonical strict `0.621198`。
- Teacher-profile aligned local residual：strict `0.617594`；V4 二专家 oracle `0.648161`，不足以训练 Router。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
