# Scheme F 算法设计与实现说明书

## 1. 结论先说

Scheme C 的 AE 已经能把真实信道压缩后还原到 `0.9491`，所以 Scheme F 不再改 AE。真正的问题是：测试位置没有真实信道，Context 必须从坐标、环境和同基站观测中预测 30,720 维 latent。

Scheme F 的做法不是把 30,720 维再次压成几百维，也不是平均几十个邻居，而是：

1. 用坐标、71 维 RF Gaussian 几何和 Scheme E 的 OOF PAS/PDP 先验描述目标点；
2. 在同一基站的观测里检索 Top8 锚点；
3. 对不同 angle-delay token 分别选 Top2 锚点；
4. 在融合前学习区域位移和 Detail 成对通道旋转；
5. 在原尺寸 latent 网格上用局部 3D 卷积、低频 Fourier mixing 和轴向注意力修正；
6. 用独立、按基站分头、有界的 PowerCNP 预测功率；
7. 用验证集自动选择 PAS/PDP 软约束、soft outage 和 exact-zero threshold。

这仍是一个需要上卡验证的新模型。`0.70` 是目标，不是承诺。

## 2. 已知事实与设计依据

### 2.1 AE 不是当前第一瓶颈

Scheme C AE Fold0：

```text
PAS   0.961609
PDP   0.954555
NMSE  0.095132
Score 0.949092
```

因此 Scheme F 固定该 AE decoder，Context 训练时 `train_decoder=false`。否则，噪声 latent 可能反向破坏已验证的 decoder ceiling。

### 2.2 Scheme D 的问题是过度平均

Scheme D Fold0 约为 `0.5960`，有效邻居约 `41.98`。路由不再 Top1 塌缩，但多个没有充分对齐的多径峰被平均后，PAS 反而模糊。

另外还发现一个确定性实现问题：Scheme D 的 router temperature 是普通 Python 属性，没有写入 `state_dict`。训练时最佳温度与重新加载后的初始温度不同。Scheme F 使用持久 buffer，并在 checkpoint 顶层再次记录温度。

### 2.3 Scheme E 的长处和致命短板

Scheme E OOF teacher 的 PAS/PDP 为 `0.6009/0.7695`，说明 71 维点云几何与 GP 粗频谱先验有价值。但 BS1 NMSE 达到 `966.82`，说明 GP 功率、outage 漏判与多轮投影串联后存在灾难性尾部。

因此 Scheme F：

- 接收 PAS/PDP、uncertainty、GP power 和 outage probability 作为输入特征；
- PAS/PDP 只能轻度修改单位能量形状；
- 修改前后强制恢复每个样本的原总功率；
- GP power 不能直接成为最终幅度；
- 最终功率只由有界 PowerCNP 输出。

## 3. 双基站数据边界

基站归属由预处理阶段根据两组坐标区域推断，并与 `Round2_Setup.json` 中基站位置对应。此后所有候选由 `ContextRepository.indices_by_cell` 分组，目标点只会看到同基站观测。

共享部分：

- BEV、corridor、latent operator 主干；
- Router 和 chart predictor。

按基站独立部分：

- latent 与功率标准化统计；
- station embedding；
- PowerCNP 最终输出头；
- Fold0 BS0/BS1 指标报告。

## 4. 输入特征

### 4.1 71 维 RF Gaussian 几何

从 PLY 三角面片重建法向、面积、切向尺度和法向厚度，聚合为最多 10,070 个各向异性 Gaussian。对 UE 与 BS-UE corridor 提取：

- 坐标、距离、方位和基站 one-hot；
- 2/4/8/16 m 局部表面密度、高度、法向和厚度；
- corridor density、clearance、Fresnel 多尺度统计和 facing。

这些是静态几何描述，不发射射线、不枚举反射路径，也不执行传统射线追踪。正式配置可直接复用 Scheme E 兼容的 metadata，避免重复计算。

### 4.2 Scheme E OOF 频谱先验

训练样本只读取八折 OOF 预测，测试样本读取 full-data teacher 预测，防止训练点通过自身信道泄漏标签。

原始 PAS/PDP 不直接塞进大 MLP，而压成 168 维统计：

```text
PAS: 24 个代理的 mean/std + 全频 mean = 96
PDP: 每 UE 分成 16 个 delay bin        = 64
UE energy                              = 4
power/uncertainty/outage/available     = 4
总计                                  = 168
```

每个基站用训练 OOF 统计独立标准化。

### 4.3 学习型环境特征

原有 1 m BEV 经过 ConvNeXt 风格 feature pyramid；BS 到 UE 的走廊序列经过 Transformer。目标 query 和观测 anchor 均采样局部环境特征。

## 5. 无线图谱与 Top8 检索

Anchor chart 不是最终信道瓶颈。它只用于找锚点：

- 从真实 Spectrum/Detail latent 分别计算分段 mean 和 RMS；
- 拼成 64 维单位向量；
- query 端根据坐标、几何、OOF 频谱先验预测同维向量；
- 用 cosine chart similarity、query-key 相似度和相对几何共同排序；
- 只保留 Top8。

训练增加 query chart 与目标真实 chart 的 cosine distillation loss。Top8 是候选池，不表示把 8 个完整信道平均。

这借鉴了 channel charting 利用 CSI 相似关系组织无线空间的思想，但最终生成仍保留完整 latent 网格。[Channel Charting-Based Channel Prediction](https://arxiv.org/abs/2410.11486)

## 6. 区域 Transport

### 6.1 低秩位移场

旧模型每个 anchor 只有一个三轴位移。Scheme F 对 angle-v、angle-h、delay 三条轴分别预测可分控制量，相加形成每个 token 的三轴位移场：

```text
offset(v,h,d) = global + f_v(v) + f_h(h) + f_d(d)
```

位移经过 `tanh` 和每轴上限约束，再由 `grid_sample` 搬运完整 latent。控制层零初始化，模型开始时不会随机扭曲信道。

### 6.2 Detail 成对通道旋转

Detail 32 个 latent channels 前后各 16 个组成成对表示。网络预测同样可分的 angle-delay rotation field，并执行二维旋转。rotation 初始为零，因此只有数据证明有用时才逐渐启用。

这不是根据手工路径公式计算相位，而是完全由训练数据学习的 latent 变换。

## 7. 每 token Top2 融合

对每个 latent token、每个 attention head，模型对 Top8 anchor 计算 logits，只保留最高两个再 softmax。于是：

- 主径 token 可以参考 anchor A；
- 另一段反射 token 可以参考 anchor B；
- 不会把 8 个 anchor 的所有峰统一平均；
- token base 直接由搬运后的完整 latent 计算，不经过全 latent 线性层。

报告记录 Spectrum/Detail token effective neighbors 和 Top1 mass。Top2 的有效锚点理论上位于 `[1,2]`。

Query-specific context attention 与 Attentive Neural Processes 的“每个 query 选择相关 context”原则一致。[Attentive Neural Processes](https://arxiv.org/abs/1901.05761)

## 8. 全分辨率神经算子

latent 形状始终为：

```text
Spectrum [64,2,4,12] = 6,144
Detail   [32,4,8,24] = 24,576
总计                    30,720
```

每个分支包含：

- depthwise 3D residual blocks，修正局部峰；
- truncated Fourier operator blocks，混合低频全局结构；
- axial attention，建模 delay 与 angle 方向关系；
- station FiLM；
- 零初始化、有界 residual head。

代码级 architecture check 会扫描所有 `Linear`，若任一层输入或输出达到 30,720 就失败。当前实现 Context 参数约 `19.19M`，固定 AE 参数 `8.53M`。

Fourier mixing 采用神经算子在频域参数化全局核的原则，但仅作为 latent 网格修正模块，不声称在求解物理 PDE。[Fourier Neural Operator](https://arxiv.org/abs/2010.08895)

## 9. 独立 PowerCNP

PowerCNP 只接收 Top8 pair features、观测标准化 log-power、outage 和 query feature。它有自己的 attention，不复用 latent Router 权重。

输出：

- `q50`：最终标准化 log-power；
- `q10/q90`：有序不确定区间；
- `power_base` 与 effective neighbors。

保护措施：

- BS0/BS1 独立输出头；
- q50 只允许在 attention base 上做有界 residual；
- 最终标准化值 clamp 到配置范围，默认 `[-4.5,4.5]`；
- Smooth-L1 + q10/q90 pinball loss；
- 验证报告输出 log10 power MAE、P90、P99；
- 双基站 breakdown 若任一 NMSE 非有限或大于 10，流水线失败。

该约束用于防止未见过的数量级，不是旧方案的人工距离权重或幅度校准。

## 10. PAS/PDP 与 outage 推理策略

### 10.1 单位能量 soft spectral prior

解码后可执行一次温和 PAS/PDP projection，scale 限制默认 `[0.75,1.35]`。projection 与原信道按 alpha 混合，最后严格恢复 projection 前逐样本总功率。

Fold0 自动扫描 `alpha = 0/0.15/0.30`，只有官方 Score 更好才用于 final。

### 10.2 soft + hard outage

outage probability 先按：

```text
amplitude_scale = sqrt(1 - soft_strength * p_outage)
```

做软衰减，再对超过 threshold 的样本输出精确零。Fold0 自动扫描：

- soft strength：`0/0.5/0.75/1.0`；
- hard threshold：`0.2 ... 0.999`。

最终策略完全由 Fold0 选择，不查看测试输出后人工挑选。

## 11. 训练损失

非 outage 样本：

- 官方 Score 可微代理；
- Spectrum MSE；
- Detail Smooth-L1 与 cosine correlation；
- Spectrum/Detail transport base loss；
- joint angle-delay power loss；
- Power Smooth-L1 与 quantile pinball；
- chart cosine distillation；
- residual 与 warp saturation 正则。

所有样本：

- outage BCE；
- Router effective-neighbor 下限；
- token Top2 防过早退化为 Top1 的轻量正则。

## 12. 自动实验和模型选择

1. 只读扫描 Scheme D：加载温度、Top64/Top16/Top8、no-warp 反事实；
2. 只读扫描 Scheme E：baseline、oracle outage、oracle power、两者、oracle-free bounded GP power；
3. 把明确诊断信号转换成有限配置改动；
4. GPU 冒烟；
5. Fold0 attempt 1；
6. 若 Score `<0.66`，允许唯一一次 attempt 2；
7. 自动选 Score 更高者；
8. 分别计算 BS0、BS1 指标；
9. 即使 Score `<0.63`，只要数值有限且单基站 NMSE `<=10`，仍按用户边界完成 full-data final，但报告标记不建议提交。

## 13. 风险与判断标准

已通过的是代码可运行性，不是竞赛精度。主要未知风险：

- 4000 条样本是否足够让 query chart 准确检索传播类型；
- Detail latent 在空间上的可预测性是否足以把 NMSE 降到 `0.6`；
- Top2 与 regional warp 是否能明显超过 D 的全局平均；
- Fold0 是否代表 leaderboard 测试洞。

目标组合示例：

```text
PAS=0.65, PDP=0.80, NMSE=0.60
Score=0.4*0.65 + 0.4*0.80 + 0.2/(1+0.60) = 0.705
```

`0.70` 只有在正式 Fold0 和后续多折结果支持时才成立。任何输出 shape 检查通过都不能当作准确率通过。
