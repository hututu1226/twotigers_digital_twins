# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | query-conditioned local set 能学习“哪个邻点的哪些 full-resolution magnitude 残差可迁移”。 | target-magnitude oracle=`0.822616`；query-only CNN、低秩 GP、固定邻点搬运都失败，但三者都没有联合使用局部观测集合与逐位置门控。 | inner spatial Score 至少 `+0.004`，重点改善 PAS 且不损害 PDP。 | 完整幅度上限充足；模型保留 `[Mp*N,Mv,Mh,S]` 网格，不经过低维瓶颈。 | K 个邻点残差经过共享 3D encoder，query-relative attention 融合，再输出 bounded full-resolution correction。 | <=0.8 h | inner 最佳仍是 epoch0，或增益 `<0.004`。 | 只有通过 inner 门槛才跑一次 strict Fold0。 |
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
- observed neighbor phase transport：最近邻/相干融合均下降，十专家 oracle 仅 `0.646909`。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
