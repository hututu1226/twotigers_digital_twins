# Scheme D 算法设计说明书

## 1. 一句话结论

旧 Context 像是“看 64 个邻居，最后只抄 1 个，而且几乎不搬动多径”；Scheme D 改为“至少综合约 8 个邻居，把它们的完整 latent 搬到目标位置后作为主答案，再让网络做小修正”。

目标是把 Fold0 从约 0.60 推到 0.65 以上，并向 0.70 尝试。它不是未经上卡验证的分数保证。

## 2. 已测问题与对应改动

### 问题 A：Router 塌缩

旧模型名义 Top64，实测有效邻居约 1–2。旧流程先 softmax，再把 `log(weight)` 塞进 attention；一旦某个邻居初期权重小，后续几乎永远翻不了身。

改动：

- 温度从 2.0 逐步降到 0.9，前期先广泛学习、后期再精确选择；
- 10% 均匀混合，保证弱邻居仍有梯度；
- 10% 路由 dropout，防止单点依赖；
- 有效邻居低于 8 时产生 diversity loss；
- attention 中只使用截断后的弱 route bias，Router 做粗筛，attention 自己精排。

举例：旧权重可能是 `[0.99,0.005,...]`；新模型前期更接近 `[0.20,0.16,0.14,...]`。这不是人工距离权重，而是给可学习 Router 设置防塌缩训练约束。

### 问题 B：Warp 被压成 0

旧 loss 直接惩罚平均位移，所以“完全不搬”天然最省损失。消融中关闭 Warp 几乎不掉分，证明它没有承担任务。

改动：

- 删除位移大小惩罚；
- 仅在位移达到允许范围 90% 以后惩罚饱和；
- 对搬运融合得到的 base latent 单独监督，让 Warp 必须对主答案负责。

### 问题 C：从特征凭空生成 30,720 维

旧模型虽然内部维度大，但最后仍主要由 token features 生成目标 latent。AE 最擅长保存的 Detail 又恰好最难凭位置生成。

改动：

```text
完整邻居 latent
  -> 几何 Warp
  -> 学习 Router 多邻居融合
  -> base latent（主答案）
  -> 全分辨率残差网络
  -> base + gated residual（最终答案）
```

Spectrum `[64,2,4,12]` 和 Detail `[32,4,8,24]` 始终保持网格形状。没有 30,720 维全连接瓶颈。

### 问题 D：功率从零预测

功率改为邻居标准化功率的学习融合值，再加最大 0.75 的有界残差。这样初始模型已有合理尺度，网络只修正遮挡、距离和环境造成的差异。

## 3. 输入

- 同基站所有可见训练样本的 Spectrum/Detail latent、功率和 outage；
- 3 m 动态上下文栅格；
- 1 m 点云 BEV；
- BS 到 UE 的走廊序列；
- 相对坐标、距离、方位、栅格内偏移；
- 基站 embedding 和 Fourier 坐标特征。

两个基站完全分开建立上下文和候选集，绝不跨基站搬 latent。

## 4. 训练洞

每一步随机隐藏一个完整空间区域，而不是只遮一个独立点。洞形状包含矩形、椭圆、走廊、复合区域和测试点连通模板，并加入 2–5.5 m observation guard，避免训练时偷看离目标过近的点。

这对应比赛场景：测试点位于训练区域内的空洞，而不是均匀随机缺失。

## 5. Router V3

Router 输入目标/观察点的环境上下文和 10 维成对几何关系。它从全部可见点学习选 Top64。输出两个概念：

- `route weights`：用于 attention 和诊断；
- `latent weights`：排除零信道邻居后重新归一化，用于真实 latent 主干融合。

报告必须检查：

- `router_effective_neighbors` 不再接近 1；
- `router_top1_mass` 不应接近 1；
- 关闭 Router 先验是否会明显影响结果。

## 6. 几何搬运

每个候选点根据目标-邻居关系预测三轴连续偏移：角度高、角度宽、时延。偏移由 `grid_sample` 对完整 latent 做三线性采样，最大范围分别由 Spectrum/Detail 配置限制。

搬运是可微的，base latent 的监督会同时更新 Router 和 Warp。`warp_saturation` 只防止长期撞边，不鼓励回到 0。

## 7. Base + Residual

对每个 latent bin：

```text
base = sum(latent_weight_i * Warp(neighbor_i))
prediction = base + sigmoid(gate) * bounded_residual
```

残差输出层严格零初始化，所以训练开始时 prediction 等于 base。Spectrum/Detail 各自有 3D depthwise residual blocks 和轴向 attention，但它们只修正，不取代真实 latent 主干。

## 8. 损失

- 官方对齐 PAS/PDP/NMSE score loss；
- Spectrum latent MSE；
- Detail Smooth-L1 和余弦相关性；
- 功率和 outage；
- joint power；
- AE decoder teacher loss；
- base Spectrum/Detail/Power 辅助监督；
- Router diversity；
- Warp saturation；
- 小残差正则。

旧 `warp_regularization` 已删除。

## 9. AE 策略

固定复用：

```text
schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

该 AE 的已测 Fold0 重建分数约 0.9491。正式 Scheme D 不重新训练 encoder；Context 阶段只允许 decoder 用极低学习率微调，以适配预测 latent 的分布。

## 10. Fold0 与 Final

Fold0：

- 使用固定空间洞验证；
- 最大 1200 epoch；
- 每 5 epoch 验证；
- 50 epoch 无提升早停；
- 最大训练时间 6.75 小时；
- outage 阈值在 9 个候选值中自动扫描。

Final：

- 使用 4000 条全部编码 latent；
- 从头训练 Context，避免 Fold0 validation 身份造成偏差；
- epoch 为 Fold0 最佳 epoch ×1.1；
- 沿用 Fold0 outage 阈值；
- 输出 500 条测试信道。

## 11. 成功判据

第一门槛是 Fold0 真实验证分数超过旧约 0.603，而不是训练 loss 下降。建议解释：

- `<0.62`：路线尚未证明有效；
- `0.62–0.65`：有改进但不足以支撑目标；
- `0.65–0.70`：值得全量和集成；
- `>=0.70`：达到阶段目标，但仍需榜单验证。

同时必须看到有效邻居提升、Warp 非零且未饱和、残差 RMS 不爆炸。若分数提高但 Router 仍 Top1，说明提升来源不是设计目标，泛化风险仍高。

## 12. 风险

- 多径精细相位可能在数米内快速变化，任何空间搬运都有不可预测上限。
- 强制多邻居过多可能在遮挡边界把不同传播区域混合；温度退火和 attention 用于后期收缩。
- 18.6M Context 加 AE 解码的验证耗时明显，早停以真实 score 为准。
- `>0.65` 需上卡实测，不应从参数量或训练时长直接推断。
