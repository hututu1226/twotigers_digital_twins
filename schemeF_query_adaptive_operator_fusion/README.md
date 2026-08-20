# Scheme F: Query-Adaptive Operator Fusion

Scheme F 是基于 Scheme C/D/E 实测故障重新实现的全分辨率 Context 方案，当前代码已可运行。它复用 Scheme C Fold0 `0.9491` 的固定 AE，使用 Scheme E 的 71 维 RF Gaussian 几何特征和 OOF PAS/PDP 先验，但最终功率由独立、按基站分头且有界的 PowerCNP 预测。

核心思路：

```text
目标坐标 + 环境 + OOF 频谱先验
  -> 预测目标所在的无线传播类型
  -> 选择 Top8 个同基站、传播类型相近的锚点
  -> 对每个锚点做低秩区域位移和 Detail 成对通道旋转
  -> 每个 latent token 独立选择 Top2 锚点
  -> 全分辨率神经算子修正
  -> 独立且有界的功率/outage 分支
  -> 固定 AE decoder
```

已完成的本地验证：

- `26` 个单元测试通过；
- CPU 小样本训练、验证和推理链路通过；
- 输出 shape/dtype 为 `[B,256,4,192] complex64`；
- 30,720 维 latent 全程保持网格结构，不存在全 latent 线性瓶颈；
- PAS/PDP 软投影前后逐样本总功率保持不变。

文档：

- `docs/algorithm_design.md`：架构、损失、依据、风险与目标指标；
- `docs/experiment_and_implementation_plan.md`：实现边界、自动实验顺序、验收门槛、5090 时间预算和最终产物。
- `docs/autodl_end_to_end.md`：Git LFS、AutoDL、实时日志、自动关机和结果下载的逐命令教程。

本机检查：

```bash
cd schemeF_query_adaptive_operator_fusion
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cpu
```

AutoDL 一条命令无人值守运行：

```bash
cd /root/autodl-fs/twotigers_digital_twins/schemeF_query_adaptive_operator_fusion
BACKUP_ROOT=/root/autodl-fs/schemeF_0820 \
SHUTDOWN_ON_SUCCESS=1 SHUTDOWN_ON_FAILURE=1 \
bash scripts/run_unattended.sh
```

Scheme F 的研究目标是 Fold0 `Score >= 0.70`，不是代码可以预先保证的结果。若最优 Fold0 小于 `0.63`，流水线仍按已确认边界生成 final NPY，但报告会明确标记为“不建议提交”。
