# Scheme A 代码运行说明书

## 1. 环境与目录

要求：

```text
Python >= 3.10
NumPy >= 1.26
PyTorch >= 2.2
正式训练：CUDA 可用的 PyTorch，推荐单张 RTX 4090 24GB
```

仓库应保持：

```text
twotigers_digital_twins/
├── Round2_Map/
│   ├── Round2_Map.ply
│   ├── Round2_Setup.json
│   ├── Round2_Test_Pos.npy
│   ├── Round2_Train_Channel.npy
│   └── Round2_Train_Pos.npy
└── schemeA_spatial_inpainting/
```

下面所有命令均从 `schemeA_spatial_inpainting` 目录执行。配置路径会相对此目录解析，因此 Windows 和 Linux 无需改绝对路径。

## 2. 安装与基础检查

```bash
cd schemeA_spatial_inpainting
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

应看到 `Ran 7 tests ... OK`。这些测试覆盖：

- power split/restore；
- Angle-Delay 正逆变换；
- 双基站空区规则；
- 网格几何通道；
- U-Net padding 与输出 shape；
- 全零预测下 PAS/PDP 不产生 NaN。
- 6 通道 BEV 的密度、最高高度和四层 occupancy 语义。

确认数据：

```bash
python - <<'PY'
from pathlib import Path
import json, numpy as np

root = Path('../Round2_Map')
setup = json.loads((root / 'Round2_Setup.json').read_text())
train_pos = np.load(root / 'Round2_Train_Pos.npy', mmap_mode='r')
test_pos = np.load(root / 'Round2_Test_Pos.npy', mmap_mode='r')
channel = np.load(root / 'Round2_Train_Channel.npy', mmap_mode='r')
print(setup)
print('train_pos', train_pos.shape)
print('test_pos', test_pos.shape)
print('channel', channel.shape, channel.dtype)
PY
```

预期：

```text
train_pos (4000, 3)
test_pos  (500, 3)
channel   (4000, 256, 4, 192) complex64
```

## 3. CPU 或 CUDA 冒烟测试

CPU：

```bash
python scripts/smoke_test.py --config configs/smoke.json --device cpu
```

AutoDL CUDA：

```bash
python scripts/smoke_test.py --config configs/smoke.json --device cuda
```

第一次执行约包含 10 到 20 秒真实数据预处理，后续会复用。最后必须出现：

```text
"status": "PASS"
```

冒烟实际执行五步：

```text
preprocess
-> train_autoencoder
-> encode_latents
-> train_spatial
-> infer
```

它只训练极小模型、极少样本和 1 epoch，不能用冒烟 score 判断正式精度。

如确实需要重建 smoke 预处理：

```bash
python scripts/smoke_test.py --config configs/smoke.json --device cpu --force-preprocess
```

## 4. 正式预处理

严格开发配置与最终配置共享同一份 1 m 静态预处理：

```bash
python scripts/preprocess.py --config configs/fold0_4090.json
```

如果目录已经存在，脚本会拒绝覆盖。仅在确认配置或数据变化后使用：

```bash
python scripts/preprocess.py --config configs/fold0_4090.json --force
```

输出：

```text
artifacts/preprocessed_1m/
├── manifest.json
├── metadata.npz
├── static_cell_0.npz
└── static_cell_1.npz
```

`manifest.json` 应满足：

```text
train_cell_counts = [2000, 2000]
test_cell_counts  = [250, 250]
outage_count      = 262
cell_rule.axis    = 1
cell_rule.threshold ~= 14.3701
```

## 5. Stage A：严格 fold0 AE

```bash
python scripts/train_autoencoder.py --config configs/fold0_4090.json
```

输出：

```text
artifacts/fold0/autoencoder/
├── best.pt
├── last.pt
├── final.pt
├── history.jsonl
├── resolved_config.json
└── summary.json
```

单独复评最佳 AE：

```bash
python scripts/evaluate.py \
  --config configs/fold0_4090.json \
  --stage autoencoder \
  --checkpoint artifacts/fold0/autoencoder/best.pt
```

重点检查 PAS、PDP、NMSE、score 和 `angle_delay_mse`。如果 AE ceiling 不够，不应直接进入复杂 U-Net 调参。

## 6. 编码 latent

```bash
python scripts/encode_latents.py --config configs/fold0_4090.json
```

输出：

```text
artifacts/fold0/encoded.npz
artifacts/fold0/encoded.json
```

包含每条训练样本的 raw latent、逐维 latent mean/std、每站 power mean/std。严格开发配置只用非验证折计算统计量。

每次更换 AE checkpoint 后必须重新编码，不能复用旧 `encoded.npz`。

诊断 latent 是否具有空间可补性：

```bash
python scripts/analyze_latents.py \
  --config configs/fold0_4090.json \
  --output artifacts/fold0/latent_diagnostics.json
```

该脚本只比较统计量，不把邻居用于训练或推理。`euclidean_separation_ratio > 1` 且 `cosine_similarity_gap > 0` 表示空间最近点通常比同站随机点更接近；若两者都接近无差异，应优先检查 AE latent，而不是盲目加深 U-Net。

## 7. Stage B：严格 fold0 空间 U-Net

```bash
python scripts/train_spatial.py --config configs/fold0_4090.json
```

默认正式设置：

```text
grid       = 1 m
crop       = 96 x 96
hole       = 12~48 m
batch      = 2
crops      = 96 / epoch
epochs     = 180，上限；patience=30 可提前停止
AMP        = true
```

输出：

```text
artifacts/fold0/spatial/
├── best.pt
├── last.pt
├── final.pt
├── history.jsonl
├── resolved_config.json
└── summary.json
```

独立复评：

```bash
python scripts/evaluate.py \
  --config configs/fold0_4090.json \
  --stage spatial \
  --checkpoint artifacts/fold0/spatial/best.pt
```

## 8. Outage 阈值扫描

```bash
python scripts/scan_outage.py \
  --config configs/fold0_4090.json \
  --checkpoint artifacts/fold0/spatial/best.pt \
  --output artifacts/fold0/spatial/outage_scan.json
```

扫描 0.1 到 0.9，并按固定空间验证 official score 选择阈值。输出中的 `best_threshold` 用于测试推理。阈值扫描复用一次 U-Net/AE 推理，不会重复做 9 次整图生成。

## 9. 开发模型测试集生成

开发模型只用了约 4/5 训练点，但可以先生成测试结果检查链路：

```bash
THRESHOLD=$(python -c "import json; print(json.load(open('artifacts/fold0/spatial/outage_scan.json'))['best_threshold'])")

python scripts/infer.py \
  --config configs/fold0_4090.json \
  --checkpoint artifacts/fold0/spatial/best.pt \
  --outage-threshold "$THRESHOLD"

python scripts/inspect_output.py outputs/fold0/Round2_Test_Channel.npy
```

## 10. 自动生成全量配置

根据开发最佳 epoch 和阈值生成最终配置：

```bash
python scripts/prepare_final_config.py
```

输出 `configs/final_selected.json`，其中：

- `validation_fold=null`；
- AE epochs = fold0 AE 最佳 epoch + 1；
- spatial epochs = fold0 U-Net 最佳 epoch + 1；
- outage threshold = 阈值扫描最佳值。

如全量数据增加约 25%，希望略增训练步数，可使用：

```bash
python scripts/prepare_final_config.py --epoch-multiplier 1.1
```

不要无依据把 multiplier 调得很大。`final_selected.json` 是本次实验派生配置，已被 `.gitignore` 排除，但会被结果打包脚本收集。

## 11. 全量最终训练与推理

逐步执行：

```bash
python scripts/train_autoencoder.py --config configs/final_selected.json
python scripts/encode_latents.py --config configs/final_selected.json
python scripts/train_spatial.py --config configs/final_selected.json
python scripts/infer.py --config configs/final_selected.json
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
```

或：

```bash
CONFIG=configs/final_selected.json bash scripts/run_final.sh
```

全量配置没有独立验证集，因此使用固定轮数的 `final.pt`，不是 `best.pt`。

## 12. 断点续训

训练中断后：

```bash
python scripts/train_autoencoder.py --config configs/fold0_4090.json --resume
python scripts/train_spatial.py --config configs/fold0_4090.json --resume
```

`--resume` 从对应输出目录的 `last.pt` 恢复模型、optimizer、scheduler、GradScaler 和 epoch。

整条脚本恢复可用：

```bash
RESUME=1 bash scripts/run_fold0.sh
RESUME=1 CONFIG=configs/final_selected.json bash scripts/run_final.sh
```

整链会跳过已经存在 `final.pt` 的完成阶段；仅有 `last.pt` 的阶段加 `--resume`；尚未开始的后续阶段正常从头启动。

重要行为：

- 带 `--resume`：继续旧训练。
- 不带 `--resume`：清理该输出目录旧的 `history.jsonl`、`best.pt`、`last.pt`、`final.pt`，重新训练。
- 已到配置总 epoch 后再 `--resume`：不会增加新 epoch。Cosine scheduler 不适合在原断点上随意延长周期；若要明显增加轮数，应新建输出目录，作为一轮有记录的新实验。
- AE 更换或续训后：重新执行 `encode_latents.py`，再训练 U-Net。

## 13. 查看进度和估算时间

```bash
tail -f artifacts/fold0/autoencoder/history.jsonl
tail -f artifacts/fold0/spatial/history.jsonl
```

基于最近 5 个 epoch：

```bash
python scripts/estimate_runtime.py --config configs/fold0_4090.json --recent 5
```

该 ETA 比文档静态估计更可信，因为它反映当前 4090、CPU、磁盘和参数设置。

## 14. 结果检查和打包

```bash
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
bash scripts/package_results.sh
```

打包脚本先检查必需文件，缺少任一最终权重、日志或 NPY 就退出。成功后产生：

```text
schemeA_results_YYYYMMDD_HHMMSS.tar.gz
schemeA_results_YYYYMMDD_HHMMSS.tar.gz.sha256
```

压缩包包含：

- 预处理 manifest、metadata、两个静态地图；
- fold0 最佳 AE/U-Net、日志和阈值扫描（存在时）；
- final AE/U-Net、encoded latent 和日志；
- 最终测试 NPY 及 JSON 信息；
- 固定配置、派生 final 配置（存在时）；
- 完整 Python 源码、脚本和测试；
- Git commit、Python/PyTorch/CUDA 环境和 `pip freeze`；
- README 和文档。

## 15. 常见错误

### `CUDA was requested... false`

PyTorch 不是 CUDA 版，或实例没有 GPU。运行：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### CUDA OOM

先把正式配置 `spatial.batch_size` 从 2 改为 1，再把 `maximum_targets` 从 24 改为 16。不要先缩 latent_dim，否则会改变 AE 表示能力和所有 checkpoint 兼容性。

### `manifest.json already exists`

正常保护行为。数据与网格配置未变则跳过预处理；确需重建时显式 `--force`。

### 找不到 `best.pt`

开发配置只有完成至少一次验证才有 `best.pt`。检查日志是否在验证前中断。全量配置不选 best，应使用 `final.pt`。

### 输出 shape 不是 500

确认不是 smoke 配置，并检查 `runtime.test_limit` 为 0。正式使用 `final_selected.json` 或 `final_4090.json`。

### score 出现 NaN

当前指标已处理全零预测的数值边界。若仍出现 NaN，检查训练 loss 是否先变成 NaN、学习率是否被修改、输入/权重是否含非有限值，并保留 `last.pt` 和日志用于定位。
