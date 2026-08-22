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
| L0-010 | 归一化复数角时延残差是否存在不同于 AE Detail 的低秩结构。 | Fold0-train OOF teacher 残差拟合 PCA；分别测 magnitude/phase/complex oracle。 | rank64 magnitude 首次越过 0.65。 | oracle 0.663754 | 0.649174 | 0.766221 | 1.049267 | +0.036665 | PROMOTE | 0.004 h |
| L0-011 | 幅度专用 log-power 表示能否用更干净的低秩系数保留 L0-010 上限。 | 全分辨率 log(1+4P) 残差 PCA，只做 magnitude oracle。 | rank8 已越过 0.65，rank128 达 0.762587。 | oracle 0.762587 | 0.795743 | 0.869032 | 1.068750 | +0.135498 | PROMOTE TO L1 | 0.003 h |
| L1-003 | 三核共享多输出 GP 能否跨空间洞预测 rank8 幅度残差系数。 | 每个 BS 独立拟合 RQ10/RQ20/Matern20，固定等权；inner 仅选 0.5/1.0 修正强度。 | 系数 skill=-1.133，修正显著退化。 | inner 0.578980 | 0.559592 | 0.695186 | 1.595091 | -0.019753 | DROP | 0.004 h |
| L1-004 | 完整能量图上的局部卷积修复能否避开不稳定的全局 PCA 坐标。 | OOF Teacher log-power 图输入；3D depthwise residual CNN；71维几何 FiLM；零初始化残差。 | PAS/NMSE 改善，PDP 小幅退化，净增 +0.001211。 | inner 0.599955 | 0.590019 | 0.716790 | 1.589615 | +0.001211 | MODIFY_ONCE | 0.101 h |
| L1-005 | 直接约束角度/时延能量边缘能否保住 L1-004 收益并修复 PDP。 | 模型、数据、采样均不变；仅增加固定权重的 PAS/PDP proxy cosine loss。 | best epoch0，所有修正均低于基线。 | inner 0.598744 | 0.586458 | 0.718097 | 1.600037 | +0.000000 | DROP | 0.064 h |
| L0-012 | 邻近观测点的完整 Teacher 幅度误差是否可直接迁移到查询点。 | 同 BS 最近 1/4/8 个点的 target-minus-OOF-teacher log-power 残差；只在 inner split 选固定强度。 | 直接候选退化，但与 V4 的二专家 oracle 提升 0.019597。 | 0.615896 | 0.555455 | 0.764117 | 1.270997 | -0.011193 | KEEP_AS_EXPERT | 0.028 h |
| L2-001 | 已提升的 adaptive Teacher 需要重新适配 Hybrid 才能把粗谱收益传到最终信道。 | V4 架构和 AE 不变；以 V4 best 初始化，只替换 leakage-safe adaptive OOF priors 并低学习率微调。 | best epoch1，之后验证持续下降。 | 0.621198 | 0.560096 | 0.754747 | 1.099510 | -0.005891 | DROP | 0.209 h |
| L0-013 | Teacher profile 对齐能否修复 L0-012 未对齐迁移造成的 PAS 损失。 | 仅增加由 query/neighbor Teacher 估计的离散角度/时延 circular shift；固定 k8、strength0.25。 | READY | - | - | - | - | - | RUNNING | <=0.1 h |
| L1-006 | query-conditioned local set 能否逐位置选择可迁移的 full-resolution magnitude residual。 | K=4 同 BS residual；共享 3D encoder；逐 voxel attention；零初始化 bounded correction。 | best epoch0；所有非零修正明显更差。 | inner 0.598744 | 0.586458 | 0.718097 | 1.600037 | +0.000000 | DROP | 0.141 h |

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

## L0-010

### Hypothesis

Fold0-train-only 的归一化复数角时延残差基底具有超过 `0.65` 的严格 Fold0 oracle 上限。

### Evidence

AE Detail latent 的 rank128 oracle 只有 `0.631194`，但基线 NMSE 仍高；因此需要检查 AE latent 之外的高分辨率表示。

### Minimal Experiment

每个基站独立使用 Fold0-train OOF teacher seed 残差拟合 PCA。Fold0 target 仅用于计算不可部署系数；对 rank0/8/16/32/64 分别评估 complex、magnitude-only 和 phase-only。

### Expected Signal

至少一种 rank64 以内的修正达到 `0.65`，且不会只依赖功率标量校准。

### Abort Rule

所有候选均低于 `0.65` 时关闭该表示，不训练系数预测器。

### Result

rank8/16/32/64 magnitude oracle 分别为 `0.631395 / 0.638181 / 0.648372 / 0.663754`。rank64 指标为 PAS=`0.649174`、PDP=`0.766221`、NMSE=`1.049267`，相对基线增加 `+0.036665`。两个基站 rank64 对训练残差解释率约为 `45.29% / 45.42%`。全流程耗时 `14.15` 秒。

### Interpretation

增益主要来自 PAS，而不是相位或功率尺度。可学习目标应改成高分辨率角时延幅度纹理，而不是 AE Detail 或复数相位。

### Decision

`PROMOTE`。先做一次 magnitude-specific 表示诊断，删除复数系数中的无用相位变化；随后只允许一个严格 inner spatial 系数 Probe。

### Repository Update

代码提交：`fca722c`；报告：`research/scheme_e_065/L0_010_COMPLEX_ORACLE.json`；基底：`artifacts/scheme_e_065/l0_010_complex_residual/train_only_complex_basis.pt`。

### Next Action

执行 L0-011 full-resolution log-power residual oracle，并据最低过线 rank 设计共享多输出 GP Probe。

## L0-011

### Hypothesis

去掉相位、直接在全分辨率 `log(1+4P)` 能量图上建立残差基底，可以用比 L0-010 更少的系数跨过 `0.65`。

### Evidence

L0-010 的最佳分支是 magnitude-only，rank64 为 `0.663754`；phase-only 和 complex 修正都不是最佳选择。

### Minimal Experiment

每个基站仅用 Fold0-train 的 OOF Teacher 残差拟合 PCA。Fold0 target 只计算不可部署的 oracle 系数，测试 rank0/8/16/32/64/128。

### Result

V4 baseline 的 rank8/16/32/64/128 magnitude oracle 分别为 `0.671796 / 0.702648 / 0.724032 / 0.744009 / 0.762587`。rank8 已是最低测试过线维度；对应 PAS=`0.650232`、PDP=`0.787438`、NMSE=`1.067647`。OOF Teacher seed 的 rank8 oracle 也达到 `0.661765`。全流程耗时 `11.04` 秒。

### Interpretation

幅度专用表示显著优于复数残差表示，而且只需 8 个理想系数就有足够上限。现在真正需要验证的是这 8 个系数能否在未见空间块上预测，而不是继续提高 PCA rank。

### Decision

`PROMOTE TO L1`。固定 rank8，执行一次每基站三核共享多输出 GP 的严格 inner spatial Probe。

### Repository Update

代码提交：`7485c64`；报告：`research/scheme_e_065/L0_011_MAGNITUDE_ORACLE.json`；基底：`artifacts/scheme_e_065/l0_011_magnitude_residual/train_only_log_power_basis.pt`。

### Next Action

执行 L1-003。inner Score 至少增加 `0.004` 才允许严格 Fold0；不通过则直接 DROP，不扩大 rank 或扫描核参数。

## L1-003

### Hypothesis

每个基站独立的 RQ10/RQ20/Matern20 共享多输出 GP，可以跨空间洞预测 rank8 全分辨率能量残差系数。

### Minimal Experiment

inner PCA 只用 inner-training 拟合；三个 GP 固定等权；只比较固定修正强度 `0.5/1.0`。inner 至少增加 `0.004` 才允许严格 Fold0。

### Result

inner Teacher 基线 PAS=`0.586466`、PDP=`0.718066`、NMSE=`1.600123`、Score=`0.598732`。alpha0.5 降至 `0.578980`，alpha1.0 降至 `0.557679`。GP 系数预测 skill=`-1.1334`、Pearson=`-0.2479`，两个基站 rank8 的训练残差解释率为 `32.22%/38.20%`。耗时 `13.76` 秒，没有进入严格 Fold0。

### Interpretation

PCA 能用真值系数重建误差，但这些全局系数在新空间块上不连续。继续改核长度、扩大 rank 或增加 GP 复杂度只会拟合训练点，不能解决表示不稳定的问题。

### Decision

`DROP` rank8 magnitude coefficient GP regression。下一实验不再预测全局 PCA 坐标，而是在完整能量图上学习局部、平移等变的卷积修正。

### Repository Update

代码提交：`56d559e`；云端报告：`research/scheme_e_065/L1_003_MAGNITUDE_GP_PROBE.json`；日志：`schemeE_spectral_gaussian_hybrid/logs/scheme_e_065_l1_003_magnitude_gp.log`。

### Next Action

执行 L1-004 full-resolution magnitude refiner。仅当 inner gain 至少 `+0.004` 时训练一次严格 Fold0 模型。

## L1-004

### Result

inner 基线 PAS=`0.586458`、PDP=`0.718097`、NMSE=`1.600037`、Score=`0.598744`。最佳 epoch10 为 PAS=`0.590019`、PDP=`0.716790`、NMSE=`1.589615`、Score=`0.599955`，净增 `+0.001211`。平均绝对 log-power 修正为 `0.13024`，总耗时 `364.02` 秒。

### Interpretation

完整网格卷积确实学到了可泛化的局部信号：PAS 和 NMSE 同时改善；但逐 bin 加权 Huber 没有直接约束能量边缘，PDP 的小幅下降抵消了大部分收益。这不是继续加宽网络的理由，而是一次明确的目标函数错位。

### Decision

`MODIFY_ONCE`。保持模型、数据、采样和训练预算不变，只加入一个固定权重的角度/时延能量边缘余弦项。若仍低于 `+0.004`，该模型族直接 DROP。

### Repository Update

代码提交：`9525b67`；云端报告：`research/scheme_e_065/L1_004_FULLRES_REFINER.json`；日志：`schemeE_spectral_gaussian_hybrid/logs/scheme_e_065_l1_004_fullres_refiner.log`。

## L1-005

### Result

加入固定权重 `0.5` 的角度/时延能量边缘损失后，inner 最佳仍是 epoch0：PAS=`0.586458`、PDP=`0.718097`、NMSE=`1.600037`、Score=`0.598744`，增益 `0.000000`。实验在 228.55 秒早停。

### Interpretation

新的边缘损失虽然继续降低训练目标，但从第一个 epoch 起验证分就低于不修正 Teacher，说明该代理损失与最终复数信道指标仍不对齐。L1-004 的微小收益无法通过一次针对性修改稳定复现。

### Decision

`DROP`。完整网格、仅依赖 query Teacher 与静态几何的 refiner 模型族不再调权重、宽度或 epoch。

### Next Action

执行 L0-012，直接检查真实邻点的完整幅度残差在空间上是否可迁移，为新的局部上下文模型提供或否定证据。

## L0-012

### Result

inner 选择 `k8, strength=0.25`，但严格 Fold0 仅为 PAS=`0.555455`、PDP=`0.764117`、NMSE=`1.270997`、Score=`0.615896`，比 V4 低 `0.011193`。与 V4 的逐样本诊断 oracle 为 `0.646686`，提升 `0.019597`，其中 oracle 选择 V4 324 个样本、local transfer 241 个样本。总耗时 102.26 秒。

### Interpretation

局部误差不是整体可平移的：直接搬运会明显破坏 PAS，但候选确实在一部分样本上与 V4 互补。由于完美 Router 的上限仍低于 0.65，当前证据不支持投入复杂 Router；该候选只保留为专家资产。

### Decision

`KEEP_AS_EXPERT`，但暂不训练 Router。

### Next Action

执行 L2-001。adaptive Teacher 的 coarse PAS/PDP 已有独立严格提升，验证针对新先验微调 Hybrid 能否修复此前的 metric bridge 损失。

## L2-001

### Result

训练期最佳为 epoch1，原始 Score=`0.615161`；固定使用 V4 outage policy 后统一复评为 PAS=`0.560096`、PDP=`0.754747`、NMSE=`1.099510`、Score=`0.621198`，比 V4 低 `0.005891`。BS0=`0.676158`，BS1=`0.566855`。训练与最终投影复评共 752.90 秒，产物已备份到 `/root/autodl-fs/scheme_e_065_l2_001`。

### Interpretation

旧 Hybrid 对 adaptive priors 的分布偏移确实敏感，但低学习率微调没有恢复 coarse Teacher 的收益，反而从首轮起继续下降。说明损失不是简单的“网络没适配”，而是 coarse PAS/PDP 改善与最终 phase/latent 目标不一致。

### Decision

`DROP`。不再对 adaptive Hybrid 重训、改学习率或增加 epoch。

### Next Action

执行 L0-013：只对 L0-012 的邻点残差增加可观测 Teacher profile 对齐，验证 PAS 损失是否来自角度/时延错位。

## L0-013

### Result

Teacher profile 对齐后的局部残差候选为 PAS=`0.559358`、PDP=`0.764518`、NMSE=`1.271590`、Score=`0.617594`。相对未对齐候选提高 `0.001698`，但仍比 V4 低 `0.009495`。V4 与该候选的逐样本诊断 oracle 为 `0.648161`，选择 V4 317 个样本、局部候选 248 个样本。实验耗时 `54.02` 秒。

### Interpretation

可观测频谱对齐确实修复了一部分角度错位，但局部幅度残差整体仍不可直接迁移。二专家 oracle 也没有越过 `0.65`，更未达到训练 Router 所要求的 `0.66`，因此继续微调邻居数、shift 范围或修正强度没有足够上限依据。

### Decision

`KEEP_AS_EXPERT`，停止该模型族的继续调参，不训练 Router。

### Next Action

执行 L0-014。初赛稳定使用全局载波斜率约 `-140.33 rad/m`；当前 BS1 拟合为 `-146.067` 且质量仅 `0.124`。固定规则为拟合质量低于 `0.5` 时回退到初赛先验，不使用 Fold0 target 选择阈值。

## L0-014

### Hypothesis

低相干度的单基站载波拟合落入了周期别名；对低质量拟合回退到初赛验证过的全局载波先验，可以改善 BS1 transport seed 和最终信道。

### Minimal Experiment

不训练模型、不扫描阈值。保留 BS0 的高质量拟合；BS1 因质量低于固定门槛 `0.5`，将 `-146.067` 回退为 `-140.33`。先复现严格 Fold0 基线，再使用相同 checkpoint、Teacher、outage policy 和投影设置只替换 carrier fit。

### Expected Signal

严格 Fold0 Score 至少增加 `0.0005`，且增益主要来自 BS1；否则立即 DROP，不扫描更多斜率。

### Leakage Control

载波拟合只来自 Fold0-train；回退先验来自初赛程序；规则在查看本实验 Fold0 结果前固定。Fold0 target 只计算最终指标。

### Result

严格 Fold0 从 PAS=`0.567081`、PDP=`0.758360`、NMSE=`1.063711`、Score=`0.627089` 提升为 PAS=`0.570384`、PDP=`0.759130`、NMSE=`1.004499`、Score=`0.631581`，净增 `+0.004492`。BS0 保持 `0.677591`；BS1 从 `0.575997` 提升到 `0.583336`。耗时 `37.28` 秒。

### Interpretation

BS1 的低质量拟合确实把 transport seed 搬向了错误相位。回退规则同时改善 PAS 和 NMSE，且改善集中在 BS1，符合预先提出的因果方向，不是单纯功率缩放造成的偶然收益。

### Decision

`KEEP`，将 `0.631581` 设为新的权威严格 Fold0 基线。后续 final inference 必须使用同一 quality gate。

## L0-015

### Hypothesis

初赛的独立水平/垂直 PAS 边缘交替投影与 V4 的联合 2-D PAS 生成误差不同，可以成为新基线之外的第三个互补专家。

### Minimal Experiment

固定使用初赛提交参数 `k=24`、距离幂 `2.0`、投影 `8` 轮；transport 使用 L0-014 已冻结的 quality gate；频谱和 UE 功率目标来自严格 Fold0 OOF AI Teacher。生成完整候选 NPY，并与新基线计算严格二专家诊断 oracle。

### Expected Signal

候选本身若超过 `0.631581` 则直接升级；否则只有二专家 oracle 超过 `0.66` 才允许进入 OOF Router。`0.65~0.66` 仅保留专家，不训练 Router。

### Abort Rule

二专家 oracle 不超过 `0.65` 时立即 DROP，不扫描邻居数、投影轮数或幅度常数。

### Result

Round1 marginal 候选 PAS=`0.547398`、PDP=`0.741646`、NMSE=`1.048985`、Score=`0.613227`，比 quality-gated V4 低 `0.018354`。二专家诊断 oracle 为 `0.637756`，只增加 `0.006175`。耗时 `30.42` 秒。

### Interpretation

独立 H/V 边缘投影没有保住联合二维角度结构；它既不是更好的直接输出，也没有形成足够强的互补性。

### Decision

`DROP`。不扫描邻居数、投影轮数或幅度常数。

## L0-016

### Result

使用 L0-014 新基线重新计算 aligned local residual 的二专家上限。候选仍为 `0.617594`；诊断 oracle 为 PAS=`0.596790`、PDP=`0.771787`、NMSE=`0.980819`、Score=`0.648399`，相对新基线增加 `0.016818`。耗时 `149.53` 秒。

### Interpretation

新基线与局部残差确有互补，但理想选择也没有越过 `0.65`，因此单独围绕这两个专家训练 Router 没有足够上限。

### Decision

`KEEP_AS_EXPERT`，不训练二专家 Router。

## L0-017

### Result

quality-gated V4、aligned local residual 与 Round1 marginal 的三专家诊断 oracle 为 PAS=`0.603995`、PDP=`0.772775`、NMSE=`0.980102`、Score=`0.651713`，相对可部署基线增加 `0.020132`。理想选择计数为 V4=`263`、local=`204`、marginal=`98`。耗时 `152.19` 秒。

### Interpretation

三个专家合起来的理想上限刚超过目标，但仍低于事先固定的 `0.66` Router 晋级线。现实 Router 只能追回 oracle 增益的一部分，现在训练复杂 Router 很可能仍达不到 `0.65`。

### Decision

`KEEP_AS_EXPERT`。不训练 Router，转向能直接提高 BS1 基线的 transport 相干度问题。

## L0-018

### Hypothesis

BS1 的 carrier fit 质量只有 `0.124`，即使斜率回退后，多邻居复数平均仍可能发生相消；低质量 cell 使用单一 transport 邻居应改善 BS1。

### Minimal Experiment

保持 checkpoint、Teacher、outage policy、output projection、BS0 和载波斜率不变。仅在 BS1 将 transport count 固定为 `1`，与当前 count 做严格同样本比较。

### Expected Signal

BS1 Score 至少增加 `0.003`；否则不实现按 cell gate，也不扫描其他 count。

### Result

新基线严格复现为 Score=`0.631581`。BS1 从 PAS=`0.504569`、PDP=`0.711720`、NMSE=`1.065703`、Score=`0.583335` 变为 PAS=`0.507764`、PDP=`0.711550`、NMSE=`1.051741`、Score=`0.585204`，净增 `+0.001869`。耗时 `16.09` 秒。

### Interpretation

多邻居复数平均确有少量相消，但只解释了 BS1 很小一部分损失，远不足以成为主要突破方向。

### Decision

`DROP`。不扫描 count，也不继续实现路径选择性 carrier fit。

## L0-019

### Hypothesis

L0-014 已改变相位对齐和 NMSE，因此必须重新计算新基线的单样本 real scale、complex scale 与 power scale 上限，才能判断尺度校准是否仍有足够空间。

### Minimal Experiment

直接读取保存的 quality-gated Fold0 prediction；不训练、不拟合可部署参数。Fold0 target 只用于明确标注的 oracle 缩放与统一评估。

### Expected Signal

至少一个尺度 oracle 达到 `0.65`，否则尺度校准继续保持 DROP。

### Result

新基线复现 Score=`0.631581`。real-scale oracle=`0.444200`，power-scale oracle=`0.607012`，最强的 complex-scale oracle 为 PAS=`0.570384`、PDP=`0.759130`、NMSE=`0.781157`、Score=`0.644092`，净增 `+0.012511`。耗时 `2.42` 秒。

### Interpretation

逐样本统一复数标量能改善一部分 NMSE，但即使使用 Fold0 target 求最优标量也达不到 `0.65`。剩余误差不是一个统一幅度或统一相位能解释的。

### Decision

`DROP`。不训练尺度校准网络。

## L0-020

### Hypothesis

剩余误差来自 angle-delay 幅度和逐路径相位中的一个主分量；分别替换真值幅度或真值相位可以确定下一套神经模块应预测什么。

### Minimal Experiment

保留 quality-gated prediction。分别构造“预测幅度+真值逐 bin 相位”和“真值幅度+预测逐 bin 相位”，逆变换回完整复数信道并统一评估。

### Expected Signal

至少一个分支的诊断上限明显超过 `0.67`，否则不值得继续做单分量模型。

### Result

预测幅度加真值逐 bin 相位得到 PAS=`0.577754`、PDP=`0.782157`、NMSE=`0.633792`、Score=`0.666379`。真值幅度加预测逐 bin 相位得到 PAS=`0.957186`、PDP=`0.912516`、NMSE=`1.676107`、Score=`0.822616`。耗时 `2.45` 秒。

### Interpretation

幅度误差主导 PAS/PDP，但此前 rank GP、full-resolution refiner 和 local transfer 已连续证明这部分很难空间预测。逐路径相位分支的上限虽较低，却已足够超过目标，而且可以从真实观测邻点获得，不必仅凭坐标生成。

### Decision

`PROMOTE_COMPONENT_PROBE`。先验证载波对齐邻点相位的可用性，不立即训练大型网络。

## L0-021

### Hypothesis

载波对齐后的同基站观测邻点，在 angle-delay 每个 bin 上保留了可迁移相位；用它替换新基线相位可以验证 query-conditioned phase attention 的可行性。

### Minimal Experiment

固定使用当前 transport 的 8 个候选和 L0-014 carrier quality gate。幅度始终取 quality-gated V4；分别使用最近邻相位、8 邻居相干融合相位，并计算 8 个单邻居相位专家的 target-informed oracle。

### Expected Signal

任一直接候选提升至少 `0.003`，或邻点相位专家 oracle 达到 `0.655`；否则 DROP，不训练 phase attention。

### Result

最近邻相位得到 PAS=`0.563862`、PDP=`0.755625`、NMSE=`1.018001`、Score=`0.626903`，下降 `0.004679`。8 邻居相干融合相位得到 Score=`0.628798`，下降 `0.002784`。baseline、相干融合和 8 个单邻点相位的 target-informed oracle 仅为 PAS=`0.574851`、PDP=`0.774229`、NMSE=`0.864322`、Score=`0.646909`。耗时 `9.03` 秒。

### Interpretation

邻点逐路径相位即使做全局载波对齐，仍会随位置快速变化。一个理想 Router 在这些候选中挑选也达不到目标，因此 learned phase attention 没有足够可观测上限。

### Decision

`DROP`。不训练 phase attention。

## L1-006

### Hypothesis

此前 magnitude 路线失败，不代表完整幅度不可预测；它们分别缺少局部集合、完整分辨率或可学习门控。query-conditioned local set full-resolution operator 可以同时补齐这三个缺口。

### Minimal Experiment

完整保留 `[Mp*N,Mv,Mh,S]` log-power 网格。共享 3D encoder 处理 K 个邻点 OOF residual，relative geometry 产生 attention，融合后只输出 bounded residual correction。先跑 Fold0-train 内部 spatial holdout，最低晋级增益固定为 `+0.004`。

### Abort Rule

inner 最佳为 epoch0、增益低于 `0.004`，或 PDP 明显下降时立即 DROP，不进入 strict Fold0。

### Result

内部基线为 PAS=`0.586458`、PDP=`0.718097`、NMSE=`1.600037`、Score=`0.598744`。epoch1/2 已降至 `0.571678/0.571936`，之后始终未恢复；epoch14 早停，best epoch=`0`、净增=`0.000000`。模型在 epoch0 的有效邻点数约为 `3.99`，因此失败不是 attention 先塌成单邻点。总耗时 `505.90` 秒，未进入 strict Fold0。

### Interpretation

网络能够降低 full-resolution log-power 训练损失，但学到的残差无法跨空间洞迁移，且与最终 PAS/PDP/NMSE 方向相反。将局部集合、完整分辨率和可学习门控同时加入后仍失败，说明继续扩大该模型或扫描 K、宽度、学习率没有证据支持。

### Decision

`DROP`。关闭 query-conditioned local-set full-resolution magnitude predictor；只保留一次固定组合诊断，检查此前 local magnitude 候选是否只是被较差的 Teacher 相位拖累。

## L0-022

### Hypothesis

L0-012/L0-013 的局部幅度残差可能有部分可用信息，但候选使用 Teacher seed 相位，整体分数被相位与 NMSE 拖累。把固定的 aligned `k8, strength=0.25` 幅度修正叠加到 L0-014 quality-gated V4 的幅度上，并完整保留 V4 相位，可能得到真正互补的候选。

### Minimal Experiment

不训练、不扫参数。邻点、残差和对齐全部只用 Fold0-train 与 OOF Teacher；Fold0 quality-gated V4 提供目标位置的部署基线相位和幅度。固定叠加 inner 已选出的 `k=8, strength=0.25` 幅度残差，再统一评估直接候选及其与 V4 的 target-informed 二专家 oracle。

### Promotion Rule

直接候选至少增加 `0.003`，或二专家诊断 oracle 达到 `0.66`；否则立即 DROP，不训练 gate，也不扫描修正强度。
