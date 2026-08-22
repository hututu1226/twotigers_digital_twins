# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | BS1 的低质量载波拟合使多邻居复数 transport seed 相消。 | L0-014 将 BS1 斜率回退到先验后提升 `+0.007338`；其拟合质量仅 `0.124`，当前 transport 仍融合多个复数邻居。 | BS1 至少提升 `+0.003`，随后统一 Fold0 有望再增 `0.001~0.004`。 | 先用 BS1 count=1 最小 probe 验证，不使用 target 选 count。 | BS0 完全不变；BS1 只把 transport count 从当前固定值改为 1。 | <=0.1 h | BS1 提升 `<0.003` 或任一主指标明显恶化。 | 仅信号成立后实现按 cell 的 production gate。 |
| 2 | 路径选择性的 angle-delay 相关可提高 BS1 carrier fit 相干度。 | 原始全信道相关质量只有 `0.124`，提示不相关多径抵消了公共载波相位。 | 改善 BS1 PAS/NMSE，总 Fold0 `+0.002~0.006`。 | 仅在 L0-018 支持“transport 相干度是瓶颈”后启用。 | 用 Fold0-train 共享高能 angle-delay bins 拟合一次斜率，不扫描 mask 比例。 | <=0.2 h | train-only fit 质量不升或 strict Score 不升。 | 固化 estimator，再做 final fit。 |
| 3 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量；L0-017 三专家 oracle 为 `0.651713`。 | `+0.003` 到 `+0.010`。 | 新候选需把 oracle 推到 `>0.66`。 | 新候选产生后先算联合 oracle。 | <=0.2 h | oracle `<0.66`。 | 只有过线后才允许严格 OOF gate。 |

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
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
