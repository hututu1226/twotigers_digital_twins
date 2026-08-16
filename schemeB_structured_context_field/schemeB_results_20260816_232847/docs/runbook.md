# 方案 B 代码运行说明书

本文说明如何从零跑通 Scheme B，包括本地 CPU 冒烟、4090 Fold0 验证、全量重训、测试集生成、断点续训、结果检查和打包。

所有命令默认在以下目录执行：

```text
twotigers_digital_twins/schemeB_structured_context_field
```

## 1. 目录要求

仓库和数据应形成以下结构：

```text
twotigers_digital_twins/
├── Round2_Map/
│   ├── Round2_Setup.json
│   ├── Round2_Map.ply
│   ├── Round2_Train_Pos.npy
│   ├── Round2_Train_Channel.npy
│   └── Round2_Test_Pos.npy
└── schemeB_structured_context_field/
    ├── configs/
    ├── docs/
    ├── scripts/
    ├── structured_context_field/
    └── tests/
```

`Round2_Map` 被根目录 `.gitignore` 排除，不会上传 GitHub。云端 clone 代码后必须单独上传/解压数据。

检查数据：

```bash
ls -lh ../Round2_Map
python -c "import numpy as np; print(np.load('../Round2_Map/Round2_Train_Pos.npy').shape); print(np.load('../Round2_Map/Round2_Train_Channel.npy', mmap_mode='r').shape); print(np.load('../Round2_Map/Round2_Test_Pos.npy').shape)"
```

正确输出应包含：

```text
(4000, 3)
(4000, 256, 4, 192)
(500, 3)
```

## 2. Python 环境

最低要求：

- Python 3.10+；
- NumPy 1.26+；
- PyTorch 2.2+；
- 正式训练建议 CUDA 版 PyTorch 和 24 GB 显存。

### 2.1 Windows 本地 CPU 环境

```powershell
cd D:\华为算法大赛复赛\schemeB_structured_context_field
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

若 PowerShell 禁止激活脚本，可只在当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 2.2 AutoDL 环境

AutoDL 镜像通常已安装 CUDA 版 PyTorch。先检查，再安装本项目且不替换 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m pip install -e . --no-deps
python -c "import numpy, torch, structured_context_field; print(numpy.__version__, torch.__version__, structured_context_field.__version__)"
```

`torch.cuda.is_available()` 必须为 `True`，设备名应为租用的 4090/4090D。

## 3. 先运行单元测试

```bash
python -m compileall -q structured_context_field scripts tests
python -m unittest discover -s tests -v
```

当前应通过 7 项测试，覆盖：

- 角度-时延变换往返；
- PAS/PDP/NMSE 基本正确性；
- Structured AE 输出尺寸与反向传播；
- 栅格 offset 和连续采样坐标；
- 仿测试集 holdout；
- 双基站 balanced limit；
- cell pooling、FPN、Query Head 反向传播。

## 4. CPU 小样本冒烟

```bash
python scripts/smoke_test.py --config configs/smoke.json --device cpu
```

首次运行包括 6 步：

```text
[1/6] dual-resolution preprocessing
[2/6] structured AE v2
[3/6] latent encoding
[4/6] contextual coordinate field
[5/6] joint decoder fine-tuning
[6/6] test inference
```

最后必须打印：

```json
{"status": "PASS"}
```

当前机器已实际完成该测试：总耗时约 19 秒，输出为 `complex64 [2,256,4,192]`，两个基站各取 1 个测试点。冒烟 Score 没有统计意义，因为只训练 1 个 epoch、4 个训练样本；它只证明数据、训练、checkpoint、联合微调和推理链路都可运行。

单独检查冒烟输出：

```bash
python scripts/inspect_output.py outputs/smoke/Round2_Test_Channel.npy --expected-count 2
```

若预处理参数变更，需要显式重建：

```bash
python scripts/smoke_test.py --config configs/smoke.json --device cpu --force-preprocess
```

不要无理由使用 `--force-preprocess`，它会覆盖共享的双分辨率预处理产物。

## 5. 三套配置的区别

| 配置 | 用途 | 验证 fold | 样本限制 | 输出 |
| --- | --- | --- | --- | --- |
| `configs/smoke.json` | 链路检查 | Fold0 | 极小 | 2 条测试样本 |
| `configs/fold0_4090.json` | 选模型/选 epoch/诊断 | Fold0 | 无 | 500 条诊断输出 |
| `configs/final_4090.json` | 默认全量模板 | 无 | 无 | 500 条正式输出 |
| `configs/final_selected.json` | Fold0 自动选择 epoch 后生成 | 无 | 无 | 500 条正式输出 |

正式提交不能使用 smoke 输出。

## 6. Fold0 完整训练

### 6.1 一键执行

```bash
mkdir -p logs
set -o pipefail
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
```

该脚本按顺序执行：

1. 双分辨率预处理；
2. Structured AE；
3. 编码 4000 条 latent；
4. Context field；
5. Joint fine-tuning；
6. AE/Context/Joint 三阶段评估；
7. stage gap 报告；
8. outage threshold 扫描；
9. 生成 500 条 Fold0 测试预测；
10. 检查输出格式。

### 6.2 分步执行

预处理只需一次：

```bash
python scripts/preprocess.py --config configs/fold0_4090.json
```

训练 AE：

```bash
python scripts/train_autoencoder.py --config configs/fold0_4090.json
```

核心输出：

```text
artifacts/fold0/autoencoder/best.pt
artifacts/fold0/autoencoder/last.pt
artifacts/fold0/autoencoder/final.pt
artifacts/fold0/autoencoder/history.jsonl
artifacts/fold0/autoencoder/summary.json
```

编码 latent：

```bash
python scripts/encode_latents.py --config configs/fold0_4090.json
```

输出：

```text
artifacts/fold0/encoded.npz
artifacts/fold0/encoded.json
```

训练 Context field：

```bash
python scripts/train_context.py --config configs/fold0_4090.json
```

联合微调：

```bash
python scripts/finetune_joint.py --config configs/fold0_4090.json
```

## 7. 断点续训

每个 epoch 都写入 `last.pt`。训练中断后不要删除 artifacts，直接：

```bash
RESUME=1 bash scripts/run_fold0.sh 2>&1 | tee -a logs/fold0.log
```

脚本规则：

- 某阶段存在 `final.pt`：认为已完成，跳过；
- 不存在 `final.pt`，但存在 `last.pt`：添加 `--resume`；
- 两者都不存在：从头开始。

也可单阶段恢复：

```bash
python scripts/train_autoencoder.py --config configs/fold0_4090.json --resume
python scripts/train_context.py --config configs/fold0_4090.json --resume
python scripts/finetune_joint.py --config configs/fold0_4090.json --resume
```

不要用不同架构配置恢复旧 checkpoint。改变 latent channels、FPN channels 或 Query width 后必须新建输出目录并从头训练。

## 8. 如何看验证结果

### 8.1 AE ceiling

```bash
python scripts/evaluate.py \
  --config configs/fold0_4090.json \
  --stage autoencoder \
  --checkpoint artifacts/fold0/autoencoder/best.pt \
  --output artifacts/fold0/autoencoder/evaluation.json
```

重点字段：

```text
pas, pdp, nmse, score
spectrum_only_pas
spectrum_only_pdp
spectrum_only_nmse
spectrum_only_score
```

若 AE ceiling 仍低，不要继续盲目增加 Context epoch。

### 8.2 Context 与 Joint

```bash
python scripts/evaluate.py \
  --config configs/fold0_4090.json \
  --stage context \
  --checkpoint artifacts/fold0/context/best.pt \
  --output artifacts/fold0/context/evaluation.json

python scripts/evaluate.py \
  --config configs/fold0_4090.json \
  --stage joint \
  --checkpoint artifacts/fold0/joint/best.pt \
  --output artifacts/fold0/joint/evaluation.json
```

除官方指标外还会输出：

- spectrum/phase latent MSE；
- power MAE/RMSE；
- outage accuracy/precision/recall/F1；
- 实际预测为 outage 的样本数。

### 8.3 自动拆解阶段损失

```bash
python scripts/report_stage_gap.py
cat artifacts/fold0/stage_gap.json
```

判断原则：

- AE 低：表示层是瓶颈；
- AE 高、Joint 低：空间上下文/Query 是瓶颈；
- Joint 与 Context 几乎相同：联合微调可以缩短；
- Joint 明显下降：joint 学习率过大或 decoder 被破坏。

### 8.4 Outage 阈值

```bash
python scripts/scan_outage.py \
  --config configs/fold0_4090.json \
  --checkpoint artifacts/fold0/joint/best.pt \
  --output artifacts/fold0/joint/outage_scan.json
```

默认扫描：`0.9, 0.95, 0.97, 0.99, 0.999, 0.999999`。阈值越高，越接近禁用主动清零。以验证 Score 最高的阈值为准。

## 9. 用 Fold0 结果生成最终配置

Fold0 完成后执行：

```bash
python scripts/prepare_final_config.py \
  --template configs/final_4090.json \
  --ae-checkpoint artifacts/fold0/autoencoder/best.pt \
  --context-checkpoint artifacts/fold0/context/best.pt \
  --joint-checkpoint artifacts/fold0/joint/best.pt \
  --outage-report artifacts/fold0/joint/outage_scan.json \
  --output configs/final_selected.json
```

它会把三个 best checkpoint 对应的训练 epoch 数和最佳 outage threshold 写入全量配置。`final_selected.json` 是生成文件，默认被本方案 `.gitignore` 排除，避免误把某次实验选择覆盖模板。

需要略微增加最终训练轮数时，例如 10%：

```bash
python scripts/prepare_final_config.py --epoch-multiplier 1.10
```

不建议在没有对比实验时随意放大。

## 10. 全量 4000 条重训

一键运行：

```bash
mkdir -p logs
set -o pipefail
bash scripts/run_final.sh 2>&1 | tee logs/final.log
```

若 `configs/final_selected.json` 存在，脚本优先使用它；否则使用 `configs/final_4090.json`。

指定配置：

```bash
CONFIG=configs/final_4090.json bash scripts/run_final.sh
```

恢复：

```bash
RESUME=1 bash scripts/run_final.sh 2>&1 | tee -a logs/final.log
```

全量配置没有验证集，因此不会生成有意义的 `best.pt`；正式依赖：

```text
artifacts/final/autoencoder/final.pt
artifacts/final/context/final.pt
artifacts/final/joint/final.pt
```

这是设计行为，不是 checkpoint 丢失。

## 11. 单独生成测试集

Fold0 模型：

```bash
python scripts/infer.py --config configs/fold0_4090.json
```

最终模型：

```bash
python scripts/infer.py --config configs/final_selected.json
```

覆盖 checkpoint 或输出路径：

```bash
python scripts/infer.py \
  --config configs/final_selected.json \
  --checkpoint artifacts/final/joint/final.pt \
  --output outputs/final/Round2_Test_Channel.npy \
  --outage-threshold 0.999
```

推理器使用 `.npy` memmap 流式写入，不会同时在内存中保留完整的 0.73 GiB 输出。

## 12. 检查正式输出

```bash
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
```

必须满足：

```text
shape  = [500, 256, 4, 192]
dtype  = complex64
finite = true
valid  = true
```

同目录 JSON 会记录 checkpoint、epoch、阈值、两个基站样本数和推理时间：

```text
outputs/final/Round2_Test_Channel.json
```

## 13. 估算剩余时间

训练至少完成 3-5 个 epoch 后执行：

```bash
python scripts/estimate_runtime.py --config configs/fold0_4090.json --recent 5
```

脚本读取真实 `history.jsonl`，分别报告 AE、Context、Joint：

- 最近 epoch 平均秒数；
- 已完成/配置 epoch；
- 剩余分钟；
- 全阶段估计分钟。

该结果比训练前的理论估计可靠。验证 epoch 比普通 epoch 慢，短窗口可能有波动。

## 14. 打包模型、结果和复现信息

最终训练和推理都成功后：

```bash
bash scripts/package_results.sh
```

脚本会先检查所有必需文件，再生成：

```text
schemeB_results_YYYYMMDD_HHMMSS.tar.gz
schemeB_results_YYYYMMDD_HHMMSS.tar.gz.sha256
```

包内包括：

- 全量 AE/Context/Joint 权重；
- latent 与预处理元数据；
- 500 条测试输出；
- Fold0 best 权重与评估报告（若存在）；
- 源码、配置、测试和文档；
- `pip freeze`、PyTorch/CUDA/GPU、Git commit/branch 信息。

校验包：

```bash
sha256sum -c schemeB_results_*.tar.gz.sha256
```

## 15. 常见问题

### 15.1 找不到 `../Round2_Map`

数据位置不对。将目录放在仓库根目录，或修改三份配置中的 `data.root`。不要把 5 GB 数据提交 Git。

### 15.2 `CUDA was requested, but ... false`

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

若 PyTorch 显示 `+cpu`，使用 AutoDL 镜像自带环境或按 PyTorch 官方 CUDA 安装方式修复，不要继续正式训练。

### 15.3 4090 OOM

按以下顺序调整 Fold0 和 Final 配置：

1. AE `batch_size: 8 -> 4`；
2. AE `gradient_accumulation: 1 -> 2`；
3. `validation_batch_size: 8 -> 4`；
4. Encoding `batch_size: 16 -> 8`；
5. Context/Joint `maximum_targets: 24 -> 12`；
6. `validation_decode_batch_size: 8 -> 4`；
7. Inference `decode_batch_size: 8 -> 4`。

不要首先缩减 latent channels，因为那会改变算法容量并使已有权重失效。

### 15.4 只有 `final.pt`，没有 `best.pt`

全量配置 `validation_fold=null`，没有验证 Score 可选 best，所以只使用 `final.pt`。Fold0 应同时有 `best.pt`。

### 15.5 修改预处理后出现 shape 不一致

删除或另存旧的 `artifacts/preprocessed_dual`，然后使用新配置 `--force` 重建；确保 Smoke/Fold0/Final 的双分辨率参数一致。

### 15.6 Smoke 输出只有 2 条

这是正常的。`runtime.test_limit=2` 只为验证链路。正式配置的 `test_limit=0`，表示不限制，输出 500 条。

### 15.7 Git 看不到模型和输出

这是 `.gitignore` 的预期行为。权重、数据、输出和日志体积很大，不提交 Git；用打包脚本下载或传到专门的对象/文件存储。

## 16. 最短正确流程

```bash
cd schemeB_structured_context_field
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cuda
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
python scripts/prepare_final_config.py
bash scripts/run_final.sh 2>&1 | tee logs/final.log
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
bash scripts/package_results.sh
```

