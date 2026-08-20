# Scheme F 算法设计说明书

## 1. 一句话结论

Scheme F 不再让模型在“只抄一个邻居”和“平均四十多个邻居”之间二选一，而是先预测目标点属于哪一种无线传播局部，再选 4--8 个合适锚点；每个角度/时延 token 只融合其中最可靠的 1--2 个，并在融合前学习多径位移和复相位变化。

它保留 Scheme C 的高质量 AE，吸收 Scheme E 有效的 PAS/PDP 先验，但禁止频谱先验直接控制绝对功率。目标是同时改善 PAS、PDP 和 NMSE，而不是靠某一个指标单独拉分。

## 2. 为什么不是简单扩大 Scheme D/E

### 2.1 Scheme D 证明“更多邻居”不等于“更多信息”

Scheme D 的有效邻居从约 1--2 个增加到 `41.98`，但 Score 从 Scheme C 的 `0.6027` 降到 `0.5960`。原因是不同邻居可能位于不同墙面、街角或遮挡区域，其多径峰值出现在不同角度和时延。没有对齐就平均，相当于把多个清晰峰值摊成宽峰，PAS 下降。

### 2.2 Scheme E 证明“频谱先验有用，但功率必须独立”

Scheme E 的 OOF 频谱教师达到 PAS/PDP `0.6009/0.7695`，说明点云和坐标可以预测一部分粗信道形状。但 BS1 NMSE 达到 `966.82`，说明把 GP 功率、UE 能量和多轮投影直接串联会放大误差。

### 2.3 一个全局 Warp 不足以搬运多径

同一个邻居信道中可能有直达、墙面反射和地面反射。目标位置变化时，这些簇并不会以完全相同的角度和时延偏移。Scheme D 每个邻居只预测近似全局三轴位移，实测 Detail 仅移动 `0.146` bin，模型基本放弃使用 Warp。

Scheme F 改成低秩位移场和复相位场：不同 angle-delay 区域可产生不同的平滑位移，但参数量仍受控。

## 3. 评分目标倒推

评分公式：

```text
Score = 0.4 * PAS + 0.4 * PDP + 0.2 / (1 + NMSE)
```

达到 `0.70` 的一组可行组合是：

```text
PAS  = 0.65
PDP  = 0.80
NMSE = 0.60
Score = 0.705
```

因此 Scheme F 的 Fold0 目标为：

- PAS：`>= 0.65`；
- PDP：`>= 0.80`；
- NMSE：`<= 0.60`；
- Score：`>= 0.70`。

这是设计目标，不是结果承诺。工程上的继续投入门槛是先稳定超过 `0.63`，再以多折验证确认是否接近 `0.70`。

## 4. 总体数据流

```text
同基站训练观测集合                         目标位置
  | latent / PAS / PDP / power                | 坐标 / 点云环境
  v                                           v
无线图谱编码器                         OOF 频谱教师 + 查询编码器
  | anchor chart embedding                    | predicted chart embedding
  +---------------- 候选检索与粗路由 ----------+
                         |
                    Top 4--8 anchors
                         |
       +-----------------+-----------------+
       |                                   |
Spectrum latent                        Detail latent
[64,2,4,12]                           [32,4,8,24]
       |                                   |
低秩位移场                            低秩位移 + 单位复相位场
       |                                   |
逐 token Top2 锚点融合                  逐 token Top2 锚点融合
       +-----------------+-----------------+
                         |
              局部卷积 + Fourier 神经算子
                         |
                 有界 residual latent
                         |
                   固定 AE decoder
                         |
单位能量复数信道形状 <---- 不确定度门控 PAS/PDP 软约束
                         |
                 独立 PowerCNP 乘回总功率
                         |
                 高精度 outage 最终门控
                         |
                  [256,4,192] complex64
```

## 5. 双基站处理

基站识别沿用已经验证的坐标分区，但所有以下模块均按 BS0/BS1 分开：

- 无线图谱原型和检索索引；
- 频谱先验标准化与不确定度校准；
- PowerCNP 输出头；
- outage 概率校准；
- 验证报告和阈值选择。

主干神经算子可以共享参数，并使用基站 embedding；容易出现尺度偏差的输出头不共享。任何锚点都不得跨基站进入候选集。

## 6. 模块 A：无线图谱检索

### 6.1 为什么需要无线图谱

几何上最近不代表传播结构最像。墙的另一侧可能只差 2 m，却属于完全不同的角度簇；同一走廊内相距 8 m 的点反而更像。

无线图谱（radio chart）把每条训练信道编码成 64 维单位向量，使传播结构相似的样本靠近。它不是把 30,720 维 latent 压缩后再复原，而只用于“找合适锚点”。最终信道仍由完整 latent 网格生成。

### 6.2 AnchorChartEncoder

锚点端输入：

- 真实 PAS/PDP 压缩表示；
- AE Spectrum latent 的统计 token；
- 71 维几何特征；
- BS id、坐标和 log-power。

输出 64 维归一化 chart embedding。训练使用同一基站内的信道相似度构造软标签：PAS/PDP 相似且空间相邻的样本为正对，传播结构差异明显的近距离样本作为 hard negative。

### 6.3 QueryChartPredictor

测试点没有真实信道，因此查询端只使用：

- 坐标和 BS id；
- 71 维 RF Gaussian 几何特征；
- 学习型点云局部/走廊 token；
- Scheme E OOF PAS/PDP 均值及不确定度。

输出预测 chart embedding。训练时只能使用 OOF 频谱先验，禁止给训练样本输入由包含自身真值拟合出的 GP 结果。

### 6.4 候选与锚点数

1. 同基站先取最多 64 个空间可达候选；
2. chart 相似度和可学习几何门控共同粗排；
3. 保留 Top8；
4. token 级融合时每个 token 只保留 Top2。

Top8 是锚点池，不代表每个输出位置平均 8 个。预期每个 token 的有效锚点约 1.5--3，避免 Scheme D 的 42 邻居过平滑。

## 7. 模块 B：学习型环境编码

保留 Scheme E 已实现的 10,070 个 RF Gaussian 和 71 维统计，同时增加轻量图算子：

- UE 周围按尺度采样 128 个 Gaussian token；
- BS--UE 走廊采样 64 个 Gaussian token；
- token 包含中心相对坐标、法向、切向尺度、厚度、面积和 corridor 位置；
- 2 层 PointTransformer/GNO 生成 query-local 环境表示。

它只从点云和数据中学习遮挡/表面关系，不发射射线、不枚举反射路径，符合“不使用传统射线追踪”的约束。

71 维特征作为稳定旁路保留，避免 4000 条样本不足以从零训练大型点云网络。

## 8. 模块 C：分区域 latent 搬运

### 8.1 低秩位移场

对每个锚点，模型根据 `target-anchor` 相对坐标、两端环境差异和频谱先验，预测低分辨率控制网格，再上采样为 latent 位移场。

- Spectrum 使用较平滑的位移场；
- Detail 使用更高分辨率但有 TV 平滑约束的位移场；
- 位移按 angle-height、angle-width、delay 三轴分别有界；
- 不惩罚合理的非零位移，只惩罚饱和和不连续。

与 Scheme D 的区别是：一个锚点内不同 angle-delay 区域可以向不同方向移动。

### 8.2 复相位场

Detail latent 不只需要移动，还需要相位旋转。模型预测两个通道并归一化为单位复数：

```text
phasor = (a + j*b) / sqrt(a^2 + b^2 + eps)
aligned_detail = grid_sample(detail) * phasor
```

相位场采用角度轴、时延轴和低秩交互项的可分结构，避免直接生成 24,576 个互不相关的相位值。它是从训练数据学习的复数变换，不是手工射线公式。

### 8.3 逐锚点监督

Scheme D 只重点监督融合后的 base，Warp 可以被其他邻居和 residual 掩盖。Scheme F 增加：

- 每个锚点搬运后的 latent 重建损失；
- Top2 中最佳锚点的 best-of-K 损失；
- 位移场平滑和饱和损失；
- 相位单位模约束；
- 关闭搬运的消融对照。

只有当搬运后的单锚点结果显著优于未搬运锚点，才认为 Transport 真正有效。

## 9. 模块 D：逐 token 稀疏融合

对每个 Spectrum/Detail token，单独计算 8 个锚点的可靠度：

- 对齐后的 token 内容；
- anchor 与 query 的 chart 相似度；
- 相对几何和距离；
- anchor outage/power；
- 锚点之间的局部方差；
- GP 频谱不确定度。

只保留该 token 的 Top2 logits 再 softmax。这样某个邻居可以贡献主径区域，另一个邻居贡献反射径区域，不会把所有邻居的整个 latent 一起平均。

Router 诊断不再追求 entropy 越高越好，而要求：

- 全局候选 Top8；
- token 有效锚点中位数在 `1.5--3.0`；
- Top1 mass 中位数在 `0.55--0.90`；
- 不同 token 的首选锚点具有足够差异；
- BS0/BS1 分开报告。

## 10. 模块 E：全分辨率混合神经算子

融合后的两个 latent 保持原形状：

- Spectrum `[64,2,4,12]`，6144 个值；
- Detail `[32,4,8,24]`，24576 个值。

每个分支使用 4--6 个混合块：

```text
Depthwise 3D Conv（局部修正）
+ truncated Fourier mixing（跨角度/时延全局关联）
+ query/environment FiLM
+ residual gate
```

Fourier mixing 只在 latent 网格轴上工作，不压平 30,720 维，也不存在大尺寸全连接层。Spectrum/Detail 在中间通过低带宽 cross-gate 交换不确定度和包络信息，但 Detail 不被压成 Spectrum。

最终 residual 输出零初始化并有界。训练开始时答案等于对齐锚点融合，算子只学习修正误差。

## 11. 模块 F：不确定度门控频谱先验

Scheme E 的 PAS/PDP GP 保留，但用途改为先验：

1. 频谱教师输出 PAS/PDP 均值和 OOF 不确定度；
2. 神经算子自己输出一份 PAS/PDP；
3. 门控网络按 BS、距离、环境和不确定度预测融合比例；
4. 只在单位能量信道形状上施加软 envelope correction；
5. 每次 correction 后重新归一化为单位能量；
6. 总功率只在最后由 PowerCNP 乘回。

不再执行会改变绝对能量的 8 轮交替投影。候选为 `no prior / soft prior`，由 OOF 官方分数选择。

## 12. 模块 G：独立 PowerCNP

### 12.1 解耦原则

模型先生成单位能量的复数形状，再单独预测 `log10(power)`。PAS/PDP 调整不得改变总功率，功率分支也不改变角度/时延分布。

### 12.2 输入和输出

PowerCNP 输入：

- query 坐标和环境编码；
- Top8 锚点的标准化 log-power 与 pair features；
- GP power 均值和不确定度，仅作为特征；
- BS 专属 embedding。

通过 query-to-context cross-attention 输出：

- 中位数 `q50`；
- 下/上分位数 `q10/q90`；
- 一个校准方差。

BS0/BS1 使用独立输出头和独立标准化统计。

### 12.3 防爆机制

- Huber + pinball quantile + Gaussian NLL 联合训练；
- 按真实信道能量分桶采样，避免低能量样本数量占优；
- 输出限制在该基站 OOF 真值的稳健范围；
- 预测若落在 `q10/q90` 外，由可微 barrier 惩罚；
- 报告 P50/P90/P99 绝对误差和最大能量倍率；
- 任一基站 NMSE 大于 `3` 立即判定实验失败，禁止生成 final。

这里的范围约束是防止神经网络产生未见过的数量级，不是旧方案的人工邻居距离加权或幅度校准。

## 13. 模块 H：保守 outage

outage 分类沿用 SVC/XGBoost/LightGBM 或等价神经集成，但采用三层门槛：

1. 5 折 OOF 官方 Score 选择阈值；
2. OOF precision 的保守下界必须达到 `0.85`；
3. 至少 4/5 折模型一致判为 outage 才输出精确零。

若门槛不满足，final 默认不输出精确零信道。验证报告同时给出“零输出关闭”和“零输出开启”两套分数，防止高 accuracy 掩盖误杀。

## 14. 损失函数

总损失由以下部分构成，权重以一次小规模梯度量级校准后固定：

- 官方分数可微代理：PAS、PDP 和 `NMSE/(1+NMSE)`；
- Spectrum latent Smooth-L1；
- Detail complex Smooth-L1、相关性和单位复相位损失；
- 单锚点 transport best-of-K；
- token 融合后的 base latent 监督；
- chart contrastive/distillation；
- PowerCNP Huber、quantile、NLL；
- outage focal/BCE；
- 位移平滑、位移饱和、相位单位模；
- residual 小幅正则和 token gate 稀疏正则。

采样与 loss 都按 BS 和最近支持距离分层，防止 BS0 或近距离样本主导训练。

## 15. 训练策略

这是一套完整模型，但一次 AutoDL 任务会自动执行内部训练阶段，不需要用户中途选择：

1. 复用固定 AE `0.9491` checkpoint，编码全部 latent；
2. 生成 5 折 OOF 频谱先验、功率先验和 chart 目标；
3. 预训练 chart retrieval 与 PowerCNP；
4. 训练 Transport + token fusion；
5. 解冻神经算子进行官方分数端到端训练；
6. 自动做关键消融并选择配置；
7. 三折通过门槛后，用 4000 条全量训练 final；
8. 生成 500 条测试 NPY、报告、权重归档、SHA256；
9. 全部校验通过后自动关机。

最大 epoch 设为较大值，例如 `1500--2000`，真正停止由验证分数早停和最大时长共同决定，不再把经验 epoch 当作充分收敛的证明。

## 16. 与研究工作的关系

本方案不是直接照搬论文，而是把可验证的结构原则用于当前赛题：

- [Attentive Neural Processes](https://arxiv.org/abs/1901.05761)：每个查询位置应选择与自身相关的上下文点，支持 query-specific attention，而不是对上下文做固定平均。
- [GNOT](https://arxiv.org/abs/2302.14376)：对不规则输入使用异构注意力，并用几何门控进行软区域划分；对应 Scheme F 的同基站候选和传播区域专家。
- [GINO](https://arxiv.org/abs/2309.00583)：用图算子处理不规则几何，再映射到规则 latent 网格；对应点云 Gaussian token 到 Spectrum/Detail 网格的环境条件。
- [Channel Charting-Based Channel Prediction](https://arxiv.org/abs/2410.11486)：利用 CSI 的空间关系建立无线 latent chart；对应 Scheme F 的 chart 检索，但这里不把 chart 当最终信道瓶颈。
- [Spatial CSI Prediction with Generative AI](https://arxiv.org/abs/2401.08023)：空间 CSI 应关注路径的角度、时延、衰减和相位；对应分区域位移、复相位场和独立功率建模。

## 17. 参数规模与显存预算

初始设计范围：

- 固定 AE：8.53M；
- chart + point/GNO encoder：5--8M；
- transport + token router：12--18M；
- Spectrum/Detail neural operator：18--28M；
- Power/outage heads：2--4M；
- 总 Context：约 40--58M。

5090 采用 AMP、batch size 1--2、Top8 anchors、gradient checkpointing 和 latent GPU cache。目标峰值显存 `<= 28 GiB`，超过时先减小 operator width，不减少 30,720 latent 分辨率。

## 18. 风险与成功判断

### 已测事实

- AE ceiling 足够高；
- 粗 PAS/PDP 可以被部分预测；
- 全局多邻居平均无效；
- BS1 功率会产生灾难性爆点。

### 合理推断

- 无线图谱检索有机会排除几何近但传播类型不同的邻居；
- per-token Top2 比全局 42 邻居平均更适合保留多径峰；
- 单位能量频谱约束和独立功率头应消除 Scheme E 的 NMSE 爆炸。

### 尚未验证

- 低秩相位场能否把 NMSE 从约 1.07 降到 0.60；
- 4000 条数据是否足以稳定训练 40--58M Context；
- Fold0 的提升能否迁移到榜单测试区域。

因此 `0.70` 属于有针对性的高目标，而不是高概率保证。只有在三个空间折都超过 `0.65`、均值接近 `0.68` 且最差基站无爆点后，才值得把 final 结果作为主提交候选。
