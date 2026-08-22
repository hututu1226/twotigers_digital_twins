# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 完整 OOF Teacher 能量图包含足够线索，可由局部 3D 卷积修正其角度-时延纹理。 | L0-011 rank8 oracle 高，但 L1-003 PCA 系数相关为 `-0.248`；说明表示坐标不稳定，而非能量纹理没有上限。 | inner 至少 `+0.004`；严格 Fold0 争取 `+0.008` 到 `+0.025`。 | Teacher rank8 oracle `0.661765`，rank128 `0.755722`。 | 保留完整 16x8x8x192 图，零初始化小型 depthwise 3D CNN，71维几何仅作 FiLM。 | <=0.5 h | inner 最佳增益 `<0.004` 或训练升、验证持续降。 | 过 inner 后按最佳 epoch 全 Fold0-train 重训一次。 |
| 2 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量，但当前候选 Router OOF 增益为 0。 | `+0.003` 到 `+0.010`。 | 必须先有新的二专家 oracle `>=+0.010`。 | 新候选产生后先算二专家 oracle。 | <=0.2 h | 无新候选或 oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

## 已 DROP

- 当前四专家 routing：完美 router 也仅 `0.645999`。
- outage 阈值优化：真值 outage oracle 仅增加 `0.000048`。
- 单样本复数标量作为主路线：oracle 仅 `0.642992`。
- rank16 local-set coefficient predictor：inner 平均增益 `0.000000`，所有非零 alpha 均更差。
- rank16 residual candidate Router：inner spatial OOF 增益 `0.000000`，严格 Fold0 回退到 baseline。
- rank8 magnitude coefficient GP：inner gain=`-0.019753`，系数 skill=`-1.133`。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。
