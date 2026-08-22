# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 剩余误差主要集中在 angle-delay 幅度或逐路径相位中的一个分量。 | L0-019 最强复数标量 oracle 仅 `0.644092`，说明单一全局相位/幅度不够，必须检查逐 bin 结构。 | 识别一个 oracle `>0.67` 的可建模分支。 | 分别替换 target magnitude 和 target phase。 | 对保存的新基线做两种 component swap oracle，不训练。 | <=0.05 h | 两个 oracle 均 `<0.65`。 | 选择上限更高且空间可预测性更强的分支做最小 neural probe。 |
| 2 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量；L0-017 三专家 oracle 为 `0.651713`。 | `+0.003` 到 `+0.010`。 | 新候选需把 oracle 推到 `>0.66`。 | 新候选产生后先算联合 oracle。 | <=0.2 h | oracle `<0.66`。 | 只有过线后才允许严格 OOF gate。 |

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
- Round1 H/V marginal projection：strict `0.613227`；与 quality-gated V4 的二专家 oracle 仅 `0.637756`。
- 现有三专家 Router：联合 oracle `0.651713`，仍低于预先固定的 `0.66` Router 门槛。
- quality-gated transport count：BS1 count 8→1 仅提升 `0.001869`，低于 `0.003` 门槛。
- quality-gated scale calibration：complex-scale oracle 仅 `0.644092`，不足以达到目标。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
