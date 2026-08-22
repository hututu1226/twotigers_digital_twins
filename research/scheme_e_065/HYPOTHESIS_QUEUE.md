# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 同基站局部观测集合能预测 rank16 spectrum residual coefficients。 | rank16 oracle `0.679030`；误差与距离/密度相关。 | 主要提升 PAS、PDP 和 NMSE，目标 `+0.004` 到 `+0.020`。 | `0.679030`，仅诊断。 | inner-train-only PCA + 小型 set encoder，保留完整 seed latent。 | <=0.5 h | inner Score 增益 `<0.004`。 | 通过后只做一次严格 Fold0；按 PROMOTE/MODIFY_ONCE/DROP 处理。 |
| 2 | 当前 seed 的复相位残差存在另一种可部署的低维表示。 | 仅当 L1-001 证明 spectrum 系数不可预测，但误差仍主要来自 NMSE 时再补诊断。 | `+0.004` 到 `+0.015`。 | 待独立 oracle。 | 先做 phase/complex residual oracle，不直接训练。 | <=0.2 h | oracle `<0.65` 或破坏 PAS/PDP。 | 只执行一个最小 probe。 |
| 3 | 极端高误差样本可由可靠性模型做保守回退。 | 最差 5% 占 67.27% 误差能量。 | `+0.003` 到 `+0.010`。 | 需使用新候选与 baseline 的专家 oracle。 | 新候选产生后先算二专家 oracle，再决定是否训练 gate。 | <=0.2 h | oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

## 已 DROP

- 当前四专家 routing：完美 router 也仅 `0.645999`。
- outage 阈值优化：真值 outage oracle 仅增加 `0.000048`。
- 单样本复数标量作为主路线：oracle 仅 `0.642992`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。
