# Scheme B: Structured Context Field

Huawei Round 2 双基站信道预测的独立新方案。它不依赖旧的 KNN 权重、幅度校准或射线追踪，核心是：

- Structured Angle-Delay AE v2；
- spectrum 与 phase/detail 双潜变量；
- 每个基站独立的 3 m latent context map；
- 1 m 点云 BEV 环境分支；
- learned cell pooling + full-map Gated FPN；
- exact-coordinate Query Head；
- shared trunk + per-BS adapters/heads；
- PAS/PDP 优先的端到端损失和 joint decoder fine-tuning。

## 文档

- [算法设计说明](docs/algorithm_design.md)
- [完整运行说明](docs/runbook.md)
- [AutoDL 4090 教程](docs/autodl_4090_guide.md)
- [4090 时长与资源估算](docs/runtime_4090_estimate.md)

## 最快验证

```bash
cd schemeB_structured_context_field
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cpu
```

本机已实际完成 CPU 端到端冒烟：预处理、AE、latent 编码、Context、Joint 和推理全部通过；输出为合法 `complex64 [2,256,4,192]`。冒烟只验证链路，不代表模型分数。

## 4090 训练

```bash
mkdir -p logs
set -o pipefail
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
python scripts/prepare_final_config.py
bash scripts/run_final.sh 2>&1 | tee logs/final.log
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
bash scripts/package_results.sh
```

预计单卡 4090/4090D：Fold0 约 2.0-4.5 小时，Final 约 1.5-3.0 小时。训练 5 个 epoch 后用真实日志修正估算：

```bash
python scripts/estimate_runtime.py --config configs/fold0_4090.json --recent 5
```

## 关键输出

```text
artifacts/fold0/autoencoder/evaluation.json
artifacts/fold0/context/evaluation.json
artifacts/fold0/joint/evaluation.json
artifacts/fold0/stage_gap.json
artifacts/fold0/joint/outage_scan.json
artifacts/final/autoencoder/final.pt
artifacts/final/context/final.pt
artifacts/final/joint/final.pt
outputs/final/Round2_Test_Channel.npy
```

`artifacts/`、`outputs/` 和原始比赛数据均被 `.gitignore` 排除，使用 `scripts/package_results.sh` 归档下载。

