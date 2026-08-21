# Scheme E: Spectral Gaussian Hybrid

## Scheme E-v3

初赛程序官方分数 `0.62` 后，复赛严格 Fold0 诊断确认“载波相位对齐的 8 邻居复信道种子”相对单一最近点具有正增益。E-v3 将两种种子分别投影、分别送入 0.95 AE，再在完整 30,720 维 latent 网格上按基站和邻居相干度学习门控融合。

- 算法说明：[docs/v3_algorithm_design.md](docs/v3_algorithm_design.md)
- AutoDL 教程：[docs/v3_autodl_end_to_end.md](docs/v3_autodl_end_to_end.md)
- 正式配置：`configs/v3_5090.json`
- 无人值守入口：`scripts/run_v3_unattended.sh`
- 独立输出：`outputs/v3/Round2_Test_Channel.npy`

E-v1、E-v2 的配置和输出均保留，E-v3 不覆盖它们。

## Scheme E-v2

E-v1 的官方反馈分数为 `0.59`。当前推荐运行 E-v2：保留频谱 GP 和真实相位初始化，增加严格 Fold0、分基站 OOF 功率标定与安全边界、参考点相对上下文、六核教师、学习率退火以及自动三次实验选择。

- 算法说明：[docs/v2_algorithm_design.md](docs/v2_algorithm_design.md)
- AutoDL 教程：[docs/v2_autodl_end_to_end.md](docs/v2_autodl_end_to_end.md)
- 正式配置：`configs/v2_5090.json`
- 无人值守入口：`scripts/run_v2_unattended.sh`

原 E-v1 配置、权重路径和 `outputs/final/Round2_Test_Channel.npy` 均保留，不被 E-v2 覆盖。

Scheme E 把官方 0.63 思路改造成无数据泄漏、可端到端验证的实现：先用点云条件的频谱教师模型预测 PAS/PDP/功率/零信道，再以同基站非零训练信道提供相位初值，最后用保留 30,720 个 AE latent 的卷积适配器修正结果。

它不使用确定性射线追踪，也不使用旧方案的“多个邻居距离加权 + 幅度校准”。最近训练信道只提供相位初值，预测目标仍由八折 OOF 教师、频谱投影和可训练的全分辨率网络完成。

## Quick Start

```bash
cd schemeE_spectral_gaussian_hybrid
pip install -r requirements.txt
bash scripts/run_smoke.sh
bash scripts/run_all_5090.sh
```

无人值守并在成功或失败后关机：

```bash
SHUTDOWN_ON_SUCCESS=1 SHUTDOWN_ON_FAILURE=1 bash scripts/run_unattended.sh
```

详细原理见 `docs/algorithm_design.md`，AutoDL 全流程见 `docs/autodl_end_to_end.md`。
