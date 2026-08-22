# Scheme E Strict Fold0 Experiments

本文中的 Fold0 均为离线验证分数，不是官方线上分数。官方线上 Scheme E 当前为 `0.59`。

| ID | Hypothesis | Core change | Probe result | Fold0 Score | PAS | PDP | NMSE | Delta | Decision | Cost |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| L0-001 | 权威 V4 基线可被统一 evaluator 复现。 | 仅审计。 | 保存前后完全一致。 | 0.627089 | 0.567081 | 0.758360 | 1.063711 | +0.000039 vs report | PASS | ~0.01 h |
| L0-002 | Teacher 增益可能在后续链路中丢失。 | 逐阶段统一复评。 | Adaptive Teacher 的 coarse 特征更好，但最终输出更差。 | 0.624773 | 0.568953 | 0.753717 | 1.089747 | -0.002316 | DROP adaptive path | ~0.01 h |
| L0-003 | 误差集中在稀疏区、远邻点和 BS1。 | 逐样本切片。 | BS1/远邻/低密度显著更差；最差 5% 占 67.27% 误差能量。 | - | - | - | - | - | EVIDENCE | CPU |
| L0-004 | 全局尺度校准足以跨过 0.65。 | 实数、复数、功率 oracle。 | 最优复数标量仅到 0.642992。 | 0.642992 | unchanged | unchanged | 0.772802 | +0.015903 | DROP primary route | ~0.01 h |
| L0-005 | 现有专家可通过 router 跨过 0.65。 | 四专家逐样本 oracle。 | 完美选择也只有 0.645999。 | 0.645999 | 0.593940 | 0.777602 | 1.053769 | +0.018910 | DROP routing | ~0.01 h |
| L0-006 | outage 误判是主要瓶颈。 | 真值 outage 硬置零 oracle。 | 只增加 0.000048。 | 0.627137 | ~same | ~same | 1.062697 | +0.000048 | DROP | ~0.01 h |
| L0-007 | 频谱 latent 残差存在可预测的低秩结构。 | Fold0-train-only PCA；Fold0 target 仅提供 oracle 系数。 | Baseline rank16 spectrum oracle 达 0.679030。 | 0.679030 | - | - | - | +0.051940 | PROMOTE TO L1 | ~0.02 h |
| L1-001 | 局部观测集合能够预测 rank16 频谱残差系数。 | 保留完整 seed latent，只预测 16 维修正。 | READY | - | - | - | - | - | RUNNING | <=0.5 h |

## L0 结论

1. 评估链路没有发现能解释 `0.023` 差距的实现错误。
2. 当前专家、outage 和单标量校准的 oracle 都不足以独立达到 `0.65`，因此不再投入复杂 router 或阈值扫描。
3. 低秩频谱残差是唯一已由 oracle 证明具有足够上限的方向。rank16 的严格诊断上限为 `0.679030`，但该值使用了 Fold0 target 求 oracle 系数，标记为 **DIAGNOSTIC ONLY - NOT DEPLOYABLE**。

## L1-001

### Hypothesis

使用目标位置、71 维几何特征、完整 6144 维 seed spectrum latent，以及同基站 16 个观测点的真实残差系数，可以在未见空间块上预测 rank16 修正系数。

### Evidence

rank16 spectrum residual oracle 为 `0.679030`；样本分数与最近训练点距离的 Spearman 相关为 `-0.1874`，与局部密度为 `+0.2030`。

### Minimal Experiment

每个基站独立训练一个小型 set encoder。inner PCA 仅由 inner-train 拟合；先在 inner spatial holdout 扫描固定的 5 个 residual 强度。inner Score 至少增加 `0.004` 才允许进行一次严格 Fold0 验证。

### Expected Signal

inner holdout 上 residual 系数 loss 稳定下降、有效邻居数不塌缩，并且统一 evaluator 的 Score 增加至少 `0.004`。

### Abort Rule

inner 增益低于 `0.004` 时立即 DROP，不以增加 epoch、宽度或扫描参数为理由继续。
