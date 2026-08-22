# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | rank16 候选与权威 V4 基线在严格 Fold0 上具有足够互补性。 | inner candidate oracle 增益 `+0.011291`。 | 只提供 router 是否值得训练的决策信息。 | 待 L0-009。 | 用已选 1/5 epoch 训练 full 模型并算严格 candidate oracle。 | <=0.03 h | strict oracle `<0.65` 或增益 `<0.010`。 | 只有跨过 0.65 才允许 OOF gate。 |
| 2 | 当前 seed 的复相位残差存在另一种可部署的低维表示。 | 仅当 L1-001 证明 spectrum 系数不可预测，但误差仍主要来自 NMSE 时再补诊断。 | `+0.004` 到 `+0.015`。 | 待独立 oracle。 | 先做 phase/complex residual oracle，不直接训练。 | <=0.2 h | oracle `<0.65` 或破坏 PAS/PDP。 | 只执行一个最小 probe。 |
| 3 | 极端高误差样本可由可靠性模型做保守回退。 | 最差 5% 占 67.27% 误差能量。 | `+0.003` 到 `+0.010`。 | 需使用新候选与 baseline 的专家 oracle。 | 新候选产生后先算二专家 oracle，再决定是否训练 gate。 | <=0.2 h | oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

## 已 DROP

- 当前四专家 routing：完美 router 也仅 `0.645999`。
- outage 阈值优化：真值 outage oracle 仅增加 `0.000048`。
- 单样本复数标量作为主路线：oracle 仅 `0.642992`。
- rank16 local-set coefficient predictor：inner 平均增益 `0.000000`，所有非零 alpha 均更差。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。
