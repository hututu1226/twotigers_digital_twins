# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 邻近观测点的完整 Teacher 幅度残差在同一 BS 内具有可迁移性。 | L1-003 证明全局 PCA 坐标不稳定；L1-004/005 证明只看 query Teacher 与静态几何不足，但误差挖掘显示远支撑/低密度区域明显更差。 | inner 至少 `+0.003`，或与 V4 的二专家 oracle `>=+0.010`。 | 完整幅度 residual oracle `0.762587`。 | 不训练网络；迁移最近 1/4/8 个观测点完整 log-power residual，固定 3 个强度只在 inner split 选择。 | <=0.1 h | inner 无正增益且严格二专家 oracle `<+0.010`。 | 有信号才训练 query-conditioned local residual encoder。 |
| 2 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量，但当前候选 Router OOF 增益为 0。 | `+0.003` 到 `+0.010`。 | 必须先有新的二专家 oracle `>=+0.010`。 | 新候选产生后先算二专家 oracle。 | <=0.2 h | 无新候选或 oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

## 已 DROP

- 当前四专家 routing：完美 router 也仅 `0.645999`。
- outage 阈值优化：真值 outage oracle 仅增加 `0.000048`。
- 单样本复数标量作为主路线：oracle 仅 `0.642992`。
- rank16 local-set coefficient predictor：inner 平均增益 `0.000000`，所有非零 alpha 均更差。
- rank16 residual candidate Router：inner spatial OOF 增益 `0.000000`，严格 Fold0 回退到 baseline。
- rank8 magnitude coefficient GP：inner gain=`-0.019753`，系数 skill=`-1.133`。
- query-only full-resolution magnitude CNN：一次 metric-aligned 修改后 inner 最佳仍为 epoch0。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。
