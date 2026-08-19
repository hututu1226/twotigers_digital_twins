# Scheme E: Spectral Gaussian Hybrid

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
