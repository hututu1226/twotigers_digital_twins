# Hypothesis Queue

Fold0 target 只允许用于最终评估和明确标注的 oracle，不得拟合任何可部署参数。

| Priority | Hypothesis | Evidence | Expected gain | Oracle ceiling | Minimal probe | GPU cost | Failure signal | Follow-up |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Adaptive Teacher 的粗谱收益需要针对新先验重新适配 Hybrid，旧 Hybrid 才不会覆盖正确信息。 | adaptive coarse PAS `0.66953→0.68143`、PDP `0.84482→0.85934`，但直接送入旧 Hybrid 后最终分 `0.627089→0.624773`。 | 最终 Fold0 至少 `+0.004`，目标进入 M1 `0.635`。 | coarse 指标同时明确改善；最终上限待本实验验证。 | V4 best 初始化；架构、AE、split 不变，只替换 adaptive OOF priors，以 `5e-5` 单次微调。 | <=1.25 h | 最佳 Fold0 不超过 baseline `+0.004`，或训练只恢复旧分且 coarse 优势仍消失。 | 过 M1 后保存完整里程碑并研究与 local-transfer 专家的新 oracle。 |
| 2 | 极端高误差样本可由新的可靠回退候选改善。 | 最差 5% 占 67.27% 误差能量，但当前候选 Router OOF 增益为 0。 | `+0.003` 到 `+0.010`。 | 必须先有新的二专家 oracle `>=+0.010`。 | 新候选产生后先算二专家 oracle。 | <=0.2 h | 无新候选或 oracle 增益 `<0.010`。 | 只允许 OOF gate。 |

## 已 DROP

- 当前四专家 routing：完美 router 也仅 `0.645999`。
- outage 阈值优化：真值 outage oracle 仅增加 `0.000048`。
- 单样本复数标量作为主路线：oracle 仅 `0.642992`。
- rank16 local-set coefficient predictor：inner 平均增益 `0.000000`，所有非零 alpha 均更差。
- rank16 residual candidate Router：inner spatial OOF 增益 `0.000000`，严格 Fold0 回退到 baseline。
- rank8 magnitude coefficient GP：inner gain=`-0.019753`，系数 skill=`-1.133`。
- query-only full-resolution magnitude CNN：一次 metric-aligned 修改后 inner 最佳仍为 epoch0。
- 未对齐的 local full-resolution residual 直接迁移：strict `0.615896`；仅作为互补专家保留。
- AE Detail latent residual：即使 rank128 target-informed oracle 也只有 `0.631194`。
- 邻居权重、投影轮数、少量 loss 权重和无证据扩容的连续扫描。
