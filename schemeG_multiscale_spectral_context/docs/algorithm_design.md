# Scheme G 算法设计说明书

## 1. 任务定义

输入为两个基站区域的 4,000 个训练位置、训练复信道、500 个测试位置、场景 PLY 和赛题设置。输出必须是：

```text
shape = [500, 256, 4, 192]
dtype = complex64
```

离线分数：

```text
Score = 0.4 * PAS + 0.4 * PDP + 0.2 / (1 + NMSE)
```

PAS/PDP 越接近 1 越好，NMSE 越接近 0 越好。Scheme G 的首要目标是严格 Fold0 超过 0.65，再决定是否值得做 4,000 条全量训练。

## 2. 不改 AE 的理由

固定使用 Scheme C `factorized_residual_v4`：

```text
Spectrum latent = [64, 2, 4, 12] = 6,144
Detail latent   = [32, 4, 8, 24] = 24,576
Total           = 30,720
```

该 AE 用真实 latent 解码的 Fold0 分数约 0.9491，当前瓶颈是从空间上下文预测 latent。Context 训练时 `train_decoder=false`，避免噪声 Context 反向破坏已经验证的 decoder。

## 3. 双基站处理

预处理根据两个互不重叠的坐标区域识别基站，每个区域恰有 2,000 条训练和 250 条测试。所有观测候选先按 `cell_id` 分开，BS0 查询永远不会使用 BS1 信道，反之亦然。

共享参数用于学习通用规律；基站 embedding、latent 标准化统计和 PowerCNP 输出头按基站区分。验证报告必须同时给出 BS0、BS1，避免总分掩盖 BS1 退化。

## 4. 输入

### 4.1 几何和环境

- UE/BS 坐标、二维/三维距离、相对方位；
- PLY 面片法向和尺度体素化得到的 10,070 个 RF Gaussian；
- 2/4/8/16 m 局部表面密度、高度、法向统计；
- BS--UE corridor 密度、净空、Fresnel 和朝向，共 71 维；
- 1 m 环境 BEV FPN 和沿 BS--UE 连线采样的 corridor Transformer。

这些特征是数据驱动神经网络输入，不进行射线发射、反射路径枚举或传统射线追踪。

### 4.2 观测频谱网格

每个观测 Spectrum latent 按连续区段池化出 16 个均值和 16 个 RMS，共 32 维。它们与功率、栅格偏移、outage 一起散射到 3 m 网格，再经 U 形 FPN 形成连续空间上下文。

32 维只是地图描述符。最终 Spectrum 和 Detail 始终在完整 5D latent 张量上运算，代码级检查拒绝任何输入或输出宽度达到 30,720 的全 latent `Linear`。

### 4.3 Scheme E 结构化先验

Scheme E 输出：

- 24 个频率代理加全频均值 PAS；
- 4 个 UE 的 192 taps PDP；
- 功率、outage probability 和不确定度。

普通 query MLP 接收压缩统计；额外的 `StructuredSpectralPriorEncoder` 保留角度/时延结构：

1. PAS reshape 为 `[25, Mv, Mh]`，形成代理均值、代理标准差和全频均值；
2. PDP reshape 为 `[N, S]`，形成 UE 均值和 UE 标准差；
3. 自适应池化到 Spectrum latent 的 `[2,4,12]`；
4. 3D 卷积生成 160 通道条件场；
5. 由 prior uncertainty 控制置信门。

## 5. 三尺度 Router

### 5.1 Spectrum

```text
全部同基站非 outage 观测
-> 按欧氏距离硬选最近 96
-> 几何、网格上下文、预测 Spectrum chart 排序
-> Top32
```

Top32 设 10% 均匀混合下限，并以 12 个有效邻居为训练正则目标。它负责稳定的 PAS/PDP 主轮廓，不负责微相位。

### 5.2 Detail

```text
全部同基站非 outage 观测
-> 最近 32
-> 仅几何/上下文排序
-> Top8
```

Detail 不使用预测 chart 越过局部距离。每个 angle-delay token 最多再选其中 4 个。`detail_phase_rotation=false`：AE 没有定义复数通道配对，不能擅自旋转。

### 5.3 Power

```text
全部同基站非 outage 观测
-> 最近 32
-> Top16
-> BoundedPowerCNP
```

PowerCNP 输出标准化 log-power 的中位数、Q10、Q90；按基站使用独立 head，最大残差和值域均有界。Scheme E GP power 只作为 query feature，不能直接接管最终幅度。

## 6. Full-Resolution Transport

每个分支从选中观测直接取完整 latent：

1. 根据 query-observation 几何预测 angle-v、angle-h、delay 小位移；
2. `grid_sample` 在完整 latent 网格搬运；
3. 每个 token 用稀疏多头注意力融合局部锚点；
4. 3D depthwise residual、低频 Fourier operator 和 axial attention 修正；
5. 零初始化输出残差，训练起点就是可解释的局部搬运基线。

不存在 `30720 -> 512 -> 30720` 的瓶颈。

## 7. 损失

总损失为配置中各项加权和：

- 官方指标对齐的 `score` loss；
- Spectrum latent MSE；
- Detail Smooth-L1 和 cosine correlation；
- Power Smooth-L1 与 Q10/Q90 pinball loss；
- query Spectrum chart 蒸馏；
- outage BCE；
- angle-delay joint power；
- transport base 辅助监督；
- Spectrum/Detail Router 有效邻居下限；
- token diversity、warp saturation 和 residual 正则。

验证仍以真实 `Score` 保存 `best.pt`，训练 loss 只用于优化，不能替代比赛指标。

## 8. 严格验证

Scheme E 原 8 折先验与 Context Fold0 空洞不一致，可能让验证点间接使用同一验证洞的信息。`build_strict_fold_prior.py` 对每个基站：

1. 删除 Fold0 validation 后，在剩余约 3177 个可见点内部按原 8 折重新生成 OOF prior；
2. 核集成权重、outage 阈值也只由这批训练区 OOF 结果选择；
3. 再用全部可见训练点拟合每个基站的三个 GP kernel，只预测 Fold0 validation；
4. 训练特征和验证特征都不接触 Fold0 标签，结果写入 `artifacts/fold0/spectral_teacher/strict_priors.npz`。

全量训练时改回标准 OOF prior 作为训练 query feature；500 个测试点使用由全部 4,000 条训练拟合的 final prior。

## 9. 后处理选择

`scan_outage.py` 只在严格 Fold0 上选择：

- exact-zero threshold；
- soft outage strength；
- PAS/PDP 投影 alpha：0 到 1 共 7 档。

软投影只改变 PAS/PDP 形状，每次后都恢复模型原总功率。先扫描统一策略，再分别扫描 BS0/BS1，并用 PAS/PDP 样本数和 NMSE 能量分母重组真实总分。两基站策略组合包含原统一策略，因此离线最优不会低于统一扫描。选择结果写入 `outage_scan.json` 并自动复制到 final inference 配置。

## 10. 自动两次实验

Attempt 1 使用正式配置。若严格分数小于 0.65，`prepare_second_attempt.py` 根据真实指标自动做一次有边界的修正：

- NMSE 高：收紧 PowerCNP 并加强功率/分位数损失；
- PAS 明显低于 PDP：加强网格 Spectrum 和 chart 蒸馏，但降低 chart 对距离路由的支配；
- Spectrum 平均参考距离大于 20 m：候选池从 96 收紧到 64；
- Detail 平均参考距离大于 12 m：候选池从 32 收紧到 16；
- warp 几乎未使用：缩小范围，不重新启用伪相位旋转。

两次都从头训练，按严格 Fold0 分数选优。全量样本从约 3177 增加到 4000，固定 steps/epoch 时每个点被抽到的频率会降低，因此最佳 epoch 默认乘 `1.35` 后成为全量训练 epoch；墙钟上限仍负责止损。

## 11. 风险与验收

主要风险：

- 4,000 条数据对 30,720 维 Detail 仍然偏少，Detail 上限可能无法由空间插值完全达到；
- Scheme E BS1 PAS 先验明显弱于 BS0；
- 更强频谱投影可能提高 PAS/PDP，但破坏复信道相关性；
- 严格 prior 可能让离线分数低于旧 OOF，这是更真实的评估，不应回避。

必须通过：

- 32 个单元测试和 CPU/CUDA 冒烟；
- `full_resolution_check=true`；
- 三个 Router 平均距离以米报告；
- 输出 shape/dtype/finite 检查；
- 权重、配置、日志、报告、NPY 和 SHA-256 进入持久盘。

分数门槛：

```text
< 0.596  说明没有超过 Scheme D，停止把架构复杂度当收益
0.596--0.63  有局部改进，但不支持 0.65 目标
0.63--0.65   接近门槛，可检查 BS1 和先验 alpha
>= 0.65      达到本轮最低目标，执行并保留全量提交
>= 0.70      达到研究目标
```

更完整的 Scheme F 证据和论文依据见 `scheme_f_failure_and_scheme_g_design.md`。
