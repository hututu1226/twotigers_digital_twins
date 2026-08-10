# 方案二算法说明：稀疏角度-时延 Token Transformer

## 1. 核心思想

方案二不先训练信道自编码器，而是让神经网络直接预测一组稀疏、连续、可微的角度-时延 Token，再将这些 Token 合成为完整复信道。

模型学习：

```text
(地图 M, 用户位置 x, 基站 i)
    -> K 个传播 Token
    -> 归一化角度-时延场
    -> 完整 MIMO-OFDM 信道 H
```

Token 是网络潜变量，不是由几何程序搜索得到的物理射线。代码没有反射点枚举、相交检测、材质系数、路径损耗公式或邻居插值。

## 2. 采用稀疏 Token 的依据

对真实非零信道进行角度-时延变换后观察到：

- 90% 时延能量的中位数只需要约 3/192 个时延单元；
- 90% 角度能量集中在少量波束；
- 90% 联合角度-时延能量的中位数约为 73/24576 个单元。

因此直接为全部 `16 x 8 x 16 x 192` 实数输出分配独立自由度浪费参数。方案二用 `K` 个连续 Token 描述主要能量结构，再由可微核展开为稠密张量。

## 3. 与方案一共享的部分

以下模块与方案一完全相同：

- 双基站空间伪标签；
- 多高度 PLY-BEV 表示；
- 链路上下文地图 Token；
- Fourier 位置编码；
- 学习式 BS 门控；
- outage 分类；
- 每基站标准化对数功率预测；
- 角度-时延正逆变换；
- 空间块验证；
- PAS/PDP/NMSE 评估。

共享实现可以保证两方案的比较只反映信道生成器差异，而不是数据划分或指标实现差异。

## 4. 输入编码

`LinkContextEncoder` 对每个候选基站输出：

```text
context_i in R^[hidden_dim], i in {0,1}
```

训练时使用真实基站标签选择 `context_i`；推理时使用门控网络的 `argmax` 选择。两个基站各有一套 Token Expert，地图和坐标编码器共享。

## 5. Token Transformer

### 5.1 条件查询

每个专家包含 `K` 个可学习查询：

```text
Q in R^[K, d_model]
```

链路 context 经过线性层后加到每个查询：

```text
Q_conditioned = Q + Linear(context)
```

随后通过多层 Transformer Encoder，使 Token 之间可以协调：

- 避免所有 Token 收敛到同一角度/时延；
- 允许一个 Token 根据其他 Token 调整宽度和增益；
- 表达多个传播簇之间的相关性。

RTX 4070 默认配置：

```text
K = 32
d_model = 128
attention_heads = 4
transformer_layers = 4
feedforward_dim = 256
```

CPU 冒烟配置缩小为 `K=4, d_model=32, layers=1`。

## 6. Token 参数化

每个 Token 输出：

```text
mu_v, mu_h, mu_d       三个连续中心
sigma_v, sigma_h, sigma_d  三个连续宽度
c[2*Mp*N]              各复数实虚/极化/用户天线通道系数
```

中心通过 `tanh` 限制在 `[-1,1]`；宽度通过 sigmoid 限制为正，且设置最小宽度防止数值尖峰。

对第 `k` 个 Token 构建三个可微高斯基：

```text
Bv_k(v) = exp(-0.5 * ((v - mu_v,k) / sigma_v,k)^2)
Bh_k(h) = exp(-0.5 * ((h - mu_h,k) / sigma_h,k)^2)
Bd_k(d) = exp(-0.5 * ((d - mu_d,k) / sigma_d,k)^2)
```

完整实数角度-时延场为：

```text
G[c,v,h,d] = sum_k c[k,c] * Bv_k(v) * Bh_k(h) * Bd_k(d)
```

代码使用一个可微 `einsum` 完成合成，所有中心、宽度、系数和 Transformer 权重都接收梯度。

## 7. 复数与功率处理

输出通道维度：

```text
c = 2 * Mp * N = 16
```

其中实部和虚部分开输出。Token 合成后的场先被归一化到单位复功率，再乘以功率头预测的幅度：

```text
G_pred = normalize(G_token) * 10^(p_pred/2)
```

最后通过角度-时延逆变换恢复 `complex64 H[256,4,192]`。

功率头和 Token 形状端到端联合训练，不存在训练后人工幅度修正。

## 8. 基站门控与硬路由

训练时：

```text
route = true_bs_label
```

这样 BS0 Expert 只学习 BS0，BS1 Expert 只学习 BS1。同时门控头用交叉熵独立学习标签。

推理时：

```text
route = argmax(gate_logits)
```

实现只执行被选择的专家，不同时生成两份巨大信道。硬路由既减少计算，也避免复相位软混合。

## 9. Outage

严格全零信道不适合用 Token 逼近。模型为每个候选基站输出 outage 概率：

```text
P_outage >= threshold -> H_pred = 0
```

零信道只参与 BS 和 outage 分类损失，不参与 Token 形状、功率、PAS、PDP 和 NMSE 训练。正式模型应在空间验证集上扫描阈值，而不是固定沿用 0.5。

## 10. 损失函数

方案二默认损失：

```text
L = 0.4 * (1 - C_pas)
  + 0.4 * (1 - C_pdp)
  + 0.2 * log(1 + NMSE)
  + 0.2 * MSE(G_shape_pred, G_shape_true)
  + 0.1 * SmoothL1(power_z_pred, power_z_true)
  + 0.1 * CrossEntropy(bs)
  + 0.2 * BCE(outage)
```

PAS/PDP 直接从最终复信道计算，确保 Token 不只是拟合 AD 张量逐元素误差，而是对齐竞赛排名指标。验证阶段仍报告官方原始 NMSE；训练使用单调的 `log(1+NMSE)`，避免极弱信道 batch 让 NMSE 梯度压制其他任务。

## 11. 为什么方案二只设一个训练阶段

Token 中心、宽度、系数和位置编码器互相依赖。不存在独立的真实 Token 标签，因此无法像方案一那样先监督训练 Token 再训练位置预测器。

全部模块从头联合训练：

```text
joint: context encoder + gate + outage + power + token experts
```

虽然参数量通常小于方案一，但连续 Token 容易出现以下优化现象：

- 多个 Token 重合；
- 宽度过大导致过度平滑；
- 宽度过小导致局部梯度不稳定；
- 功率头先于形状收敛，造成早期 NMSE 较大；
- 少数强路径占据全部 Token。

因此方案二通常需要更多 epoch，参数少不等于收敛更快。

## 12. 空间验证与模型选择

使用和方案一相同的 3369/631 空间块划分。模型选择依次观察：

1. gate accuracy；
2. outage accuracy 及后续 precision/recall；
3. PAS；
4. PDP；
5. NMSE；
6. 综合 score。

如果 PAS/PDP 高而 NMSE 很差，说明 Token 能量位置正确但复相位/功率不足；可以增加通道系数容量或引入小型残差解码器。如果所有指标均低，首先检查 Token 是否塌缩。

## 13. 方案二相对方案一的优缺点

优点：

- 输出结构显式稀疏；
- 参数更少；
- 每个 Token 可解释为一个学习到的角度-时延传播簇；
- 与 PAS/PDP 指标天然匹配；
- 不需要预训练自编码器。

缺点：

- 高斯核对复杂非高斯谱形可能表达不足；
- 精细复相位比功率谱更难；
- Token 无监督，优化容易塌缩；
- 增大 K 会线性增加稠密合成开销；
- 白噪声和弱散射无法用少量 Token 精确还原。

## 14. 推荐实验

优先尝试：

```text
K: 16, 32, 48
d_model: 96, 128
layers: 2, 4
```

不建议在 12GB 4070 上直接从 `K=64,d_model=256` 开始，因为一次实验成本明显增加，且未证明 Token 数是瓶颈。

应记录：

- 每个 Token 中心/宽度分布；
- 有效 Token 数量；
- 两基站分别的 PAS/PDP/NMSE；
- outage 与非 outage 分组分数；
- 相同数据划分下与方案一的对比。

## 15. 代码对应关系

```text
src/channel_ai/models/scheme2.py   Token Transformer 与可微合成
src/channel_ai/models/common.py    地图/坐标编码、门控、outage、功率
src/channel_ai/transforms.py       角度-时延变换
src/channel_ai/training.py         联合损失和训练
configs/scheme2_smoke.json         CPU 冒烟
configs/scheme2_4070.json          RTX 4070 完整训练
```
