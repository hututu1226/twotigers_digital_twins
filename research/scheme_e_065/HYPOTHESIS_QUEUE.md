# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 将每个 test 点一对一匹配到同基站最近 train 点，可直接复刻测试点所在空间邻域，比平移周期洞更可能匹配 RF 环境。 | L0-024 的 support AUC 已降至 `0.5512`，但 link/environment AUC 仍为 `0.9939`；主要偏移来自局部墙面法向、BS 距离/方位和走廊密度。 | 建立 500 条、cell 分布与 test 一致的可解释验证集。 | 不适用；匹配阶段不读训练或测试信道。 | 每 cell 用空间距离的最小成本一对一 assignment；移除被匹配 train 点后重新计算 support 和 link/environment AUC。沿用样本/支撑门槛，并要求 link AUC 至少下降 0.10。 | CPU, <=0.05 h | link AUC 仍高于 `0.8963`，或最近支撑中位数差超过 `2 m`。 | 通过才允许一次 matched-split V4 复训；失败则先审计可达的训练几何覆盖，不训练模型。 |
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
- 旧 Scheme1 空洞可代表真实测试难度：旧验证最近支撑中位数 `21.594 m`，test 仅 `6.199 m`；该解释已被 L0-023 否定。
- translated periodic phase matching：L0-024 最佳 support AUC=`0.551188`，但 link/environment AUC 仍为 `0.993904`，只下降 `0.002390`，且样本数仅 437。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
