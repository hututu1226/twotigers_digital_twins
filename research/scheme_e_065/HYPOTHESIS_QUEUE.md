# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 当前 periodic Fold0 与 test 的邻点距离匹配，但 link/environment 域 AUC=`0.9963`；为两个基站独立选择 periodic phase，可能得到更像测试场景的验证洞。 | L0-023 已确认 support-only AUC=`0.5959`、最近距离中位数仅差 `0.380 m`，但五个现有 Fold 的 link/environment AUC 均 `>0.993`。 | 建立更可信的离线决策集，避免继续优化只对 Fold0 有效的方向。 | 不适用；搜索阶段不读任何信道标签。 | 固定 `72 m` tile、`26 m` hole，先按无标签分布签名筛 phase，再对少量组合计算域 AUC。仅当样本 450-650、每 cell>=200、最近距离差<=2m、support AUC<=0.70 且 link AUC 至少下降 0.10 时晋级。 | CPU, <=0.10 h | 最佳组合仍未通过任一固定门槛。 | 通过才允许一次 test-matched V4 复训；否则 DROP periodic-phase matching。 |
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
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。

## 已 KEEP

- L0-014 carrier quality gate：strict `0.631581`，相对旧基线 `+0.004492`；BS1 `+0.007338`。
