# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 低可信度 BS1 carrier fit 应回退到初赛全局先验。 | BS0 fit quality=`0.784`，BS1 仅 `0.124`；初赛官方 `0.62` 使用稳定斜率约 `-140.33`，当前 BS1 却偏到 `-146.067`。 | strict `+0.0005` 以上，主要改善 BS1。 | 不需要 target oracle。 | 固定 quality `<0.5` 回退，不训练、不扫斜率。 | <=0.1 h | 总分不增或 BS1 不增。 | 有效则写入 final 规则；无效则 DROP。 |
| 2 | 初赛的水平/垂直边缘频谱交替投影可形成第三个结构不同的专家。 | L0-013 二专家 oracle=`0.648161`，只差 `0.001839`；当前 Scheme E 使用联合 2D PAS，而初赛使用独立 H/V marginals。 | 新候选使三专家 oracle `>0.66`。 | 必须先超过 `0.66` 才训练 gate。 | 固定初赛 k24/p2/8轮设置，只替换为双基站和现有 AI power。 | <=0.2 h | 三专家 oracle `<=0.66`。 | 仅过线后训练严格 OOF gate。 |
| 3 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量，但当前候选 Router OOF 增益为 0。 | `+0.003` 到 `+0.010`。 | 必须先有新的二专家 oracle `>=+0.010`。 | 新候选产生后先算二专家 oracle。 | <=0.2 h | 无新候选或 oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

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
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。
