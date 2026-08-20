# Scheme F: Query-Adaptive Operator Fusion

Scheme F 是基于 Scheme C/D/E 实测结果设计的下一代 Context 方案。当前目录处于“设计冻结、尚未实现”状态，不能直接运行训练。

核心思路：

```text
目标坐标 + 环境 + OOF 频谱先验
  -> 预测目标所在的无线传播类型
  -> 选择 4--8 个同基站、同传播类型的锚点
  -> 对每个锚点做分路径 latent 对齐和复相位旋转
  -> 每个 latent token 独立选择 1--2 个可靠锚点
  -> 全分辨率神经算子修正
  -> 独立且有界的功率/outage 分支
  -> 固定 AE decoder
```

设计文档：

- `docs/algorithm_design.md`：架构、损失、依据、风险与目标指标；
- `docs/experiment_and_implementation_plan.md`：实现边界、自动实验顺序、验收门槛、5090 时间预算和最终产物。

Scheme F 的目标是 Fold0 `Score >= 0.70`，但这不是未训练代码可以保证的结果。第一道继续投入门槛为 `Score >= 0.63` 且两个基站均无 NMSE 爆点。
