# Scheme A: Angle-Delay Spatial Inpainting

本目录是复赛方案 A 的独立实现。它不会导入或覆盖仓库原有的 Scheme1/Scheme2，也不使用 KNN、邻居加权、幅度校准或射线追踪。

核心流程：

```text
完整复信道 H
  -> 独立拆分 log10 power 与单位功率 shape
  -> Angle-Delay 3D AutoEncoder，压缩为 256 维 latent
  -> 按两个基站分别铺成 1 m 空间网格
  -> 动态挖 12~48 m 连续空洞
  -> 共享 2D U-Net 补全 latent / power / outage
  -> AE Decoder + 功率恢复 + inverse Angle-Delay
  -> Round2_Test_Channel.npy
```

## 从这里开始

- [算法设计说明](docs/algorithm_design.md)：数学表示、双基站路由、网络、损失、验证协议和风险。
- [代码运行说明](docs/runbook.md)：本地测试、正式训练、断点续训、评估、推理和打包。
- [AutoDL 4090 教程](docs/autodl_4090_guide.md)：从 GitHub 和数据上传到训练、下载与关机的完整操作。

## 三套配置

| 配置 | 用途 | 数据 | 输出 |
|---|---|---|---|
| `configs/smoke.json` | CPU/CUDA 冒烟，只判定代码链路 | 极小分层样本 | `artifacts/smoke` |
| `configs/fold0_4090.json` | 严格空间验证、选 epoch 和 outage 阈值 | 4/5 训练，1/5 连续空间验证 | `artifacts/fold0` |
| `configs/final_4090.json` | 4000 条全量训练模板 | 全部训练样本 | `artifacts/final` |

推荐顺序是先跑 `fold0_4090.json`，再用 `prepare_final_config.py` 从最佳 checkpoint 自动生成 `configs/final_selected.json`，最后全量训练。不要直接依据训练损失挑模型。

## 最短自检

从本目录执行：

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cpu
```

本机已完成 7 项单元测试，以及真实数据 CPU 冒烟：预处理、AE 一轮训练、latent 编码、U-Net 一轮训练、固定空间验证、两个基站各一条测试样本推理。输出为 `complex64`，shape 为 `(2, 256, 4, 192)`。

## 正式训练

```bash
bash scripts/run_fold0.sh

python scripts/prepare_final_config.py
CONFIG=configs/final_selected.json bash scripts/run_final.sh

bash scripts/package_results.sh
```

正式提交文件必须通过：

```bash
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
```

预期 shape 为 `(500, 256, 4, 192)`，dtype 为 `complex64`，文件约 `0.732 GiB`。

## 训练产物与 Git

`artifacts/`、`outputs/`、`logs/` 和 `configs/final_selected.json` 已在本目录 `.gitignore` 中排除。GitHub 只管理代码、固定配置和文档；比赛数据、模型权重、日志和生成信道应通过 AutoDL 文件存储或本地下载保存。
