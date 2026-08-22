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
| L1-001 | 局部观测集合能够预测 rank16 频谱残差系数。 | 保留完整 seed latent，只预测 16 维修正。 | 非零 alpha 全部更差。 | inner 0.599283 | 0.586466 | 0.718066 | 1.581632 | +0.000000 | DROP | 0.009 h |
| L0-008 | L1-001 平均失败但可能存在可识别的互补子集。 | 计算修正候选 oracle、系数 skill 和空间连续性。 | inner oracle 增益 +0.011291；邻点系数 skill 为负。 | oracle 0.610575 | - | - | - | +0.011291 | KEEP FOR STRICT ORACLE | 0.010 h |
| L0-009 | 新候选与权威 V4 基线的严格 Fold0 oracle 能否跨过 0.65。 | 只训练已选 epoch，统一评估固定候选。 | 完美选择达到 0.656555；所有单候选低于基线。 | oracle 0.656555 | - | - | - | +0.029465 | PROMOTE ROUTER PROBE | 0.014 h |
| L1-002 | inner OOF gain router 能否学到候选互补性。 | ExtraTrees 预测每个候选相对基线的逐样本收益，保守阈值回退。 | inner OOF 增益 0；router 未过门槛。 | 0.627089 | 0.567081 | 0.758360 | 1.063711 | +0.000000 | DROP | 0.013 h |
| L0-010 | 归一化复数角时延残差是否存在不同于 AE Detail 的低秩结构。 | Fold0-train OOF teacher 残差拟合 PCA；分别测 magnitude/phase/complex oracle。 | READY | - | - | - | - | - | RUNNING | <=0.25 h |

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

### Result

模型在训练集上迅速拟合，但 inner validation loss 从 cell0 的 `0.370353` 上升到约 `0.45`，cell1 也未稳定改善。alpha `0.25/0.5/0.75/1.0` 的 Score 均低于不修正的 `0.599283`，因此没有进入严格 Fold0。

### Decision

`DROP`。只允许一次无训练的反事实诊断，确认候选是否具有至少 `+0.010` 的逐样本 oracle 互补性；否则关闭该模型族。

## L1-002

### Hypothesis

仅使用 Fold0-train 内层留出标签训练的保守 Router，可以识别何时用 rank16 残差候选替换 V4 基线。

### Evidence

L0-009 的逐样本真值选择上限为 `0.656555`，相对基线具有 `+0.029465` 的互补空间。

### Minimal Experiment

使用空间 tile 五折 OOF 的 ExtraTrees 回归每个候选相对基线的样本收益；Router 只能读取位置、71 维几何、邻距、seed 统计、候选系数和候选自身功率，不能读取 Fold0 target。

### Expected Signal

内层 OOF 至少增加 `0.004`，才允许把同一个 Router 应用到严格 Fold0。

### Abort Rule

内层 OOF 增益低于 `0.004` 时停止，不增加树深、神经 Router 或参数扫描。

### Result

内层 OOF 最佳阈值为 `0.02`，但 Score 增益为 `0.000000`，`passed=false`。因此没有训练最终 Router；严格 Fold0 仍选择 baseline，PAS=`0.567081`、PDP=`0.758360`、NMSE=`1.063711`、Score=`0.627089`。运行耗时 `46.23` 秒。`0.656555` 仍是使用 Fold0 target 逐样本选候选的不可部署 oracle。

### Interpretation

残差候选在少数样本上有帮助，但这种帮助无法由当前可部署特征在新空间块上预测。继续增加 Router 深度只会重新拟合内层标签，不符合止损规则。

### Decision

`DROP` deployable residual routing。候选只保留作诊断，不再训练更复杂 Router。

### Repository Update

报告：`research/scheme_e_065/L1_002_OOF_ROUTER_FIX.json`；产物：`artifacts/scheme_e_065/l1_002_oof_router_fix/`；修复提交：`3998454`。

### Next Action

执行 L0-010，比较 AE Detail latent 与直接复数角时延残差的 train-only oracle 上限。AE Detail rank128 已知只有 `0.631194`，因此不会再训练 Detail latent predictor。
