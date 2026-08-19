# AE v4 失败分析与重构依据

## 1. 先给结论

这次不通过“多加 epoch”修补失败模型。失败包证明 v3 的主要问题是网络结构和训练职责，而不是训练时间不够：24,576 维 Detail latent 中确实有样本差异，但这些信息经过共享融合层后只剩约 2% 的输出影响。

可以把它理解成两个人共同写答案：Detail 已经写了很多细节，但交卷前把两份答案塞进同一个摘要器，摘要器几乎只保留了 Spectrum 的内容。继续训练相当于让 Detail 再写几百遍，并不能保证摘要器开始采用它。

v4 的处理方式是让 Spectrum 和 Detail 分别写“功率轮廓”和“复数残差”，直到最终答案前都不混合；最后按明确公式合成。这样 Detail 的信息不会再经过一个可以选择忽略它的共享融合层。

## 2. 已确认的测量事实

下面的数据来自 `schemeC_ae_failure_analysis.tar.gz` 中的 Epoch 38 最佳权重和固定样本诊断。

| 项目 | 结果 |
| --- | ---: |
| Fold0 PAS | 0.1014 |
| Fold0 PDP | 0.3736 |
| Fold0 NMSE | 1.9985 |
| Fold0 Score | 0.2567 |
| Spectrum-only Score | 0.2556 |
| 完整相对 Detail 置零增益 | 0.0011 |
| 32 样本 Detail 打乱 Score | 0.2597，几乎未变 |
| 训练集 Epoch 38 Score | 约 0.2562 |

训练集和验证集都低，说明模型连见过的样本都没有学好，不是“只在陌生区域泛化差”。

## 3. 已排除的错误方向

以下可能性已经由实验排除，不再作为本轮主要修改方向：

- 数据或权重出现 NaN/Inf：没有发现。
- 信道与 angle-delay 变换写错：恒等往返 Score 为 1.0。
- Detail latent 全部塌缩成常数：跨样本标准差正常。
- Detail gate 完全关闭：门值约 0.523，不是简单的全零门。
- 只因为验证集特别难：训练集同样只有约 0.25。
- 只需要继续跑更多 epoch：学习率下降后长期没有改善，结构消融也证明 Detail 未进入输出。

## 4. 根因证据

### 4.1 Detail 在哪里消失

把真实 Detail 与全零 Detail 的激活差异逐层测量：

| 位置 | 相对差异 |
| --- | ---: |
| Detail refine 后 | 106.31% |
| Fusion 后 | 2.96% |
| Angle-delay upsample 后 | 2.36% |
| 最终输出 | 2.34% |

因此信息不是没有进入 Detail encoder，也不是 gate 直接关掉，而是在共享 Fusion 处被 Spectrum 激活淹没。

### 4.2 旧课程学习为什么无效

旧代码在进入 decoder 前执行 `detail_latent * detail_scale`，随后立刻经过 GroupNorm。GroupNorm 会按当前样本重新归一化幅度，所以乘 0.1 后又被拉回相近尺度。

实测 `detail_scale=0.1` 与 `1.0` 的最终输出相对差只有约 `3.8e-6`。日志里的 Detail 比例虽然逐步增大，网络实际看到的基本只是“关闭或开启”两种状态。

### 4.3 Spectrum-only 辅助任务职责错误

Spectrum 输入只有 angle-delay 功率，没有复数相位。旧损失却要求它同时优化复数 NMSE 和相干性，相当于“只给黑白亮度，要求还原每个像素的颜色方向”。模型只能学习平均模板，而且 Spectrum-only 路径还会反向更新共享 Detail/Fusion 模块。

### 4.4 decoder 自身容量或条件不足

固定 decoder，只优化单样本 latent 50 步：

- 只优化 Detail：约 `0.241 -> 0.242`。
- 同时优化两个 latent：约 `0.241 -> 0.264`。

这说明问题不只是 encoder 没有产生好 latent；旧 decoder 即使得到可自由调整的 latent，也很难快速表示正确答案。

## 5. v4 修改与证据的对应关系

| 失败证据 | v4 修改 |
| --- | --- |
| Detail 在 Fusion 被淹没 | 删除共享 Fusion，两个 decoder 隔离到最后一步 |
| Detail 置零仍产生固定模板 | Detail decoder 使用无 bias、无 affine 归一化，零输入严格输出零 |
| latent 缩放被 GroupNorm 抵消 | 比例在分支归一化之后作用于最终残差 |
| Spectrum 被要求猜复数相位 | Spectrum 辅助损失只重建功率及角度/时延边缘 |
| 复数方向学不准，NMSE 接近 2 | 增加逐 bin 能量加权复数方向损失 |
| Detail 网络容量明显偏小 | Detail encoder/decoder 分别提高到约 215 万/180 万参数 |
| 低学习率导致严重欠拟合 | 正式初始学习率提高到 `8e-4`，容量测试使用 `1e-3` |
| 阶段之间互相干扰 | Coarse、Detail、Joint 三阶段切换可训练模块 |

这些修改解决的是已观察到的故障机制，但不能在运行前保证 Fold0 达到 0.8。是否有效必须由下面的门槛判定。

## 6. 四层验收

### 6.1 代码结构测试

- 输出形状保持 `[B,4,8,16,192]`。
- latent 保持 6,144 + 24,576。
- 零 Detail latent 的残差严格为零。
- `detail_scale=0.1` 与 `1.0` 输出明显不同。
- 正反向梯度 finite。

### 6.2 训练集容量测试

```bash
bash scripts/run_ae_capacity_gates.sh
```

- 1 样本 200 step：Score 至少 0.90。
- 32 样本 1200 step：Score 至少 0.85。

若失败，说明模型连记住少量答案都困难，不能用“泛化难”解释。

### 6.3 Fold0 AE 门槛

- 完整 AE Score 至少 0.75。
- 去掉 Detail 后至少下降 0.10。
- 打乱 Detail 后至少下降 0.05。

若失败，`run_fold0.sh` 在 Context 前退出。

### 6.4 最终研究目标

- AE 目标：固定 Fold0 Score 0.80。
- 最终 Joint 目标：固定 Fold0 Score 0.70。

0.75 只是继续 Context 的下限，不等于宣称 AE 已达到目标。

## 7. 新结果应该怎样解读

1. 容量测试失败：先修 AE 结构或损失，不跑 Fold0。
2. 容量测试通过、Fold0 AE 低：表示模型能记忆但泛化不足，优先调整正则、增强和训练样本划分。
3. AE 过 0.75、Detail 消融不过：仍说明细节分支没有可靠参与，不能训练 Context。
4. AE 达到 0.8、Context 明显低：AE 已不再是主瓶颈，集中优化空间 Context。
5. AE 和 Context 都高、Joint 不升：减少 Joint 预算，避免 decoder 破坏已学表示。

所有结论都以固定 Fold0、相同随机种子和相同 Score 实现为准，不能用全量训练 loss 或测试集输出格式代替验证分数。
