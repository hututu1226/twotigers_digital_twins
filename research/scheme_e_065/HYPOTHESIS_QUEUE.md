# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Scheme1 `0.3925` 离线但官方 `0.62`，Scheme E 严格 Fold0 `0.6316` 但旧线上仅 `0.59`，可能存在验证空洞与真实测试几何难度不匹配。 | 测试位置和 Fold0 都可在不使用测试信道标签的前提下计算同基站支撑距离、密度和 71D 几何；当前只看了总体距离摘要。 | 解释线上/离线反转，并确定严格 Fold0 是否应继续作为唯一决策门槛。 | 不适用；这是分布诊断，不产生可部署分数。 | 固定比较旧 Scheme1 验证、strict Fold0 和 test 的距离/密度/几何分布；用域分类 AUC 量化可分性，并按 test 几何重加权 Fold0 指标。 | CPU, <=0.05 h | test 与 strict Fold0 高度一致且重加权结论不变。 | 若明显偏移，重建 test-matched 多 Fold；否则关闭“验证集不匹配”解释。 |
| 2 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量；L0-022 二专家 oracle 仅 `0.640268`。 | `+0.003` 到 `+0.010`。 | 新候选需把 oracle 推到 `>0.66`。 | 新候选产生后先算联合 oracle。 | <=0.2 h | oracle `<0.66`。 | 只有过线后才允许严格 OOF gate。 |

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
- query-conditioned local-set full-resolution magnitude：inner best epoch0、增益 `0.000000`；所有非零修正显著下降。
- quality-gated aligned magnitude composition：strict `0.633628`，虽有 `+0.002047` 真增益，但低于固定 `+0.003` 门槛；二专家 oracle 仅 `0.640268`。候选文件保留用于诊断，不作为主路线。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
