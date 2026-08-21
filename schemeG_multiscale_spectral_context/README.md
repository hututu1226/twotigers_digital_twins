# Scheme G: Multi-Scale Spectral Context

Scheme G 是针对 Scheme F 实测退化重新实现的全分辨率 Context。它复用 Scheme C Fold0 `0.9491` 的 AE，不重新压缩 30,720 维 latent，也不使用传统射线追踪。

主要修正：

```text
同基站观测
  +-> 频谱摘要散射到 3 m 网格 -> FPN 形成连续频谱上下文
  +-> 最近 96 点 -> Spectrum Top32 -> 完整 6,144 维频谱 latent
  +-> 最近 32 点 -> Detail Top8   -> 完整 24,576 维细节 latent
  +-> 最近 32 点 -> Power Top16  -> 有界功率与不确定区间

Scheme E 完整 PAS/PDP -> (2,4,12) 结构化条件场
环境/坐标/71 维 RF Gaussian 特征 -> 查询条件
完整 latent -> 固定 AE decoder -> 复信道 -> 验证选出的软频谱投影/outage 策略
```

关键边界：

- 三个分支使用独立 Router 和硬距离候选池，避免再次选到约 51 m 外的锚点；
- Detail latent 是普通卷积特征，`detail_phase_rotation=false`；
- 32 维频谱摘要只用于地图与检索，不负责重建 latent；
- Fold0 训练区内部交叉拟合 Scheme E 先验，验证区完全只由可见训练点预测；
- BS0/BS1 分别选择 outage 阈值、软抑制强度和 PAS/PDP 投影强度；
- 训练按最大 epoch 加早停，最佳 checkpoint 决定全量训练轮数；
- 目标为严格 Fold0 `Score >= 0.65`，研究目标为 `0.70`，均不是预先保证值。

本地验证：

```bash
cd schemeG_multiscale_spectral_context
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cpu --force-preprocess
python scripts/inspect_architecture.py --config configs/fold0_5090.json
```

当前结果：32 个单元测试通过；含双基站独立策略的 CPU 端到端冒烟通过；架构检查确认 30,720 维完整 latent、无全 latent 线性瓶颈，Context 约 20.61M 参数。

AutoDL 无人值守入口：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeG_multiscale_spectral_context
nohup env \
  BACKUP_ROOT=/root/autodl-fs/schemeG_0821 \
  LEGACY_RUN_ROOT=/root/autodl-fs/schemeF_0820_20260820 \
  SHUTDOWN_ON_SUCCESS=1 SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_unattended.sh > logs/launcher.log 2>&1 &
echo $! | tee logs/launcher.pid
```

详细资料：

- `docs/scheme_f_failure_and_scheme_g_design.md`：日志、场景、论文和根因证据；
- `docs/algorithm_design.md`：模型、损失、验证约束和风险；
- `docs/autodl_end_to_end.md`：从 Git LFS 到自动关机、下载结果的完整操作；
- `docs/experiment_and_implementation_plan.md`：自动实验顺序与验收门槛。
