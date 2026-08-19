# Scheme D: Multi-neighbor Transport + Residual Context

Scheme D 针对 Scheme C Context 的实测失败重新设计：Router 不再塌成 Top1，Warp 不再被“移动越大惩罚越大”的损失压回 0；目标 latent 直接由多个邻居的完整 30,720 维 latent 搬运和融合得到，网络只学习有界小残差。

正式训练固定复用 Scheme C 中验证分数约 0.9491 的 AE，不重新试错 AE。

```bash
cd schemeD_transport_residual_context
bash scripts/run_smoke.sh
bash scripts/run_all_5090.sh
```

无人值守：

```bash
SHUTDOWN_ON_SUCCESS=1 SHUTDOWN_ON_FAILURE=1 bash scripts/run_unattended.sh
```

算法说明见 `docs/algorithm_design.md`，AutoDL 全流程见 `docs/autodl_end_to_end.md`。
