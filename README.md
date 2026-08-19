# Huawei Round 2 Physical AI Channel Generation

## 0819 新方案

当前新增两套基于已验证 0.9491 AE 的方案：

- `schemeE_spectral_gaussian_hybrid`：八折点云条件频谱 GP + 零信道集成 + 相位初值投影 + 30,720 维全分辨率修正。
- `schemeD_transport_residual_context`：防塌缩多邻居 Router + 几何 latent 搬运 + direct base + gated residual。

AutoDL 按 Scheme E 后 Scheme D 无人值守运行的唯一总教程：

- [0819 Scheme E → Scheme D AutoDL 全流程](docs/0819_schemeE_then_D_autodl.md)

以下内容保留为早期基线说明。

本项目针对华为算法大赛复赛任务，实现两套不依赖传统射线追踪、也不使用邻居加权或人工幅度校准的 AI 信道生成方案：

1. `scheme1`：基站门控 Mixture-of-Experts + 角度-时延自编码器。
2. `scheme2`：基站门控 + 稀疏角度-时延 Token Transformer。

两套方案均包含：

- 双基站学习式门控；
- 空间相关的全零信道 `outage` 分类；
- 端到端学习的对数功率预测；
- PLY 地图多高度 BEV 上下文；
- PAS、PDP、NMSE 联合损失与验证；
- CPU 小样本冒烟配置和 RTX 4070 完整训练配置；
- 断点续训、验证、推理和 `.npy` 提交文件生成。

## 当前验证状态

- 单元测试：角度-时延可逆变换、PAS/PDP/NMSE、自适应双小区标签均通过。
- 方案一：真实数据上依次跑通 `autoencoder -> predictor -> joint`，并成功推理。
- 方案二：真实数据上跑通稀疏 Token 合成、反向传播和推理。
- 两个推理样例均输出 `(2, 256, 4, 192)`、`complex64`，且不存在 NaN/Inf。

冒烟测试只验证工程链路，不代表模型已经收敛，也不能使用冒烟 checkpoint 提交比赛。

## 目录结构

```text
configs/                    训练配置
docs/                       算法与运行文档
scripts/preprocess.py       数据和地图预处理
scripts/train.py            通用训练入口
scripts/evaluate.py         空间验证集评估
scripts/infer.py            测试集信道生成
scripts/smoke_test.py       两方案一键冒烟
src/channel_ai/             核心 Python 包
tests/                      单元测试
Round2_Map/                 原始数据，不进入 Git
artifacts/                  缓存、日志和 checkpoint，不进入 Git
outputs/                    最终输出，不进入 Git
```

## 最短运行路径

在项目根目录执行：

```bash
python -m pip install -e .
python scripts/preprocess.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cpu
```

RTX 4070 上训练：

```bash
python scripts/train.py --config configs/scheme1_4070.json --device cuda
python scripts/train.py --config configs/scheme2_4070.json --device cuda
```

生成最终提交文件，以方案一为例：

```bash
python scripts/infer.py \
  --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --output outputs/Round2_Test_Channel.npy \
  --device cuda
```

Windows PowerShell 可将续行符替换为反引号，或将命令写成一行。

## 详细文档

- [方案一算法说明](docs/scheme1_algorithm.md)
- [方案一运行说明](docs/scheme1_runbook.md)
- [方案二算法说明](docs/scheme2_algorithm.md)
- [方案二运行说明](docs/scheme2_runbook.md)
- [Git 与 GitHub 协作指南](docs/git_github_guide.md)
- [AutoDL RTX 4090 完整训练与结果回传教程](docs/autodl_4090_full_guide.md)
