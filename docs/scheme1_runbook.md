# 方案一代码运行说明

## 1. 适用范围

本文档覆盖从全新环境到生成比赛提交文件的完整过程，包括：

- 本地 CPU 冒烟；
- RTX 4070 CUDA 训练；
- 中断恢复；
- 空间验证；
- 测试集推理；
- 输出文件检查；
- 常见故障处理。

## 2. 环境要求

最低要求：

```text
Python >= 3.10
NumPy >= 1.26
PyTorch >= 2.2
磁盘剩余空间 >= 20 GB
内存 >= 16 GB，建议 32 GB
```

GPU 完整训练建议：

```text
RTX 4070 Desktop 12 GB: batch_size 8 起步
RTX 4070 Laptop 8 GB: batch_size 4 起步
```

不要直接从 `requirements.txt` 猜 CUDA 版本。远端机器先执行 `nvidia-smi`，再从 PyTorch 官方安装页选择与驱动兼容的 CUDA wheel。

## 3. 创建 Python 环境

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安装 CUDA 版 PyTorch 后，再安装本项目且避免覆盖 PyTorch：

```bash
python -m pip install -e . --no-deps
python -m pip install "numpy>=1.26,<3"
```

验证环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 训练前必须看到：

```text
True
NVIDIA GeForce RTX 4070 ...
```

## 4. 数据目录

项目根目录应为：

```text
project/
  Round2_Map/
    Round2_Setup.json
    Round2_Map.ply
    Round2_Train_Pos.npy
    Round2_Train_Channel.npy
    Round2_Test_Pos.npy
  configs/
  scripts/
  src/
```

快速检查：

```bash
python -c "import numpy as np; print(np.load('Round2_Map/Round2_Train_Pos.npy').shape); print(np.load('Round2_Map/Round2_Train_Channel.npy', mmap_mode='r').shape)"
```

预期：

```text
(4000, 3)
(4000, 256, 4, 192)
```

## 5. 预处理

执行：

```bash
python scripts/preprocess.py \
  --data-root Round2_Map \
  --output-dir artifacts/preprocessed \
  --resolution 4 \
  --link-samples 16 \
  --local-grid 3 \
  --local-radius 16 \
  --validation-fraction 0.15 \
  --blocks-per-cell 12 \
  --chunk-size 16 \
  --seed 2026
```

Windows 可写成单行。成功后应看到近似：

```text
Preprocessing complete: train=3369 validation=631 outage=262
```

生成文件：

```text
artifacts/preprocessed/manifest.json
artifacts/preprocessed/metadata.npz
artifacts/preprocessed/train_map_tokens.npy
artifacts/preprocessed/test_map_tokens.npy
```

`manifest.json` 保存源文件大小和修改时间。远端重新传输数据后建议重新预处理，不要盲目复制来自另一绝对路径的缓存。

## 6. 单元测试

```bash
python -m unittest discover -s tests -v
```

必须全部显示 `ok`。其中角度-时延往返测试失败时严禁开始正式训练。

## 7. CPU 冒烟

只跑方案一：

```bash
python scripts/train.py --config configs/scheme1_smoke.json --device cpu
```

该命令会依次打印：

```text
Starting phase=autoencoder
Starting phase=predictor
Starting phase=joint
Training complete
```

冒烟使用 12 条训练和 6 条验证样本，每阶段只有 1 个 epoch。它只验证：

- 数据能读；
- 复数 FFT/IFFT 能执行；
- 三阶段冻结/解冻正确；
- 损失可反向传播；
- checkpoint 可保存。

它不验证比赛精度。冒烟 `score` 很低属于正常现象。

## 8. RTX 4070 完整训练

桌面 4070 12GB：

```bash
python scripts/train.py --config configs/scheme1_4070.json --device cuda
```

配置默认阶段：

```text
autoencoder: 120 epochs
predictor:   180 epochs
joint:        80 epochs
```

训练输出目录：

```text
artifacts/runs/scheme1_4070/
  resolved_config.json
  history.jsonl
  last.pt
  best_autoencoder.pt
  best_predictor.pt
  best_joint.pt
  best.pt
  final.pt
```

优先使用 `best.pt` 推理。`final.pt` 是最后一个 epoch，并不保证验证分数最佳。

### 8.1 Laptop 4070 8GB

复制配置后修改：

```json
"batch_size": 4,
"num_workers": 2
```

如果仍然显存不足，依次尝试：

1. `batch_size=2`；
2. `base_channels=12`；
3. `latent_dim=192`；
4. 保持 AMP 为 `true`。

不要先降低地图分辨率，因为地图 Token 缓存几乎不占 GPU 显存。

## 9. 中断恢复

训练中断后：

```bash
python scripts/train.py \
  --config configs/scheme1_4070.json \
  --device cuda \
  --resume artifacts/runs/scheme1_4070/last.pt
```

checkpoint 保存当前阶段、epoch、优化器和余弦学习率调度器。恢复时必须使用原配置；更改 latent 或卷积通道会导致权重 shape 不匹配。

## 10. 查看训练日志

`history.jsonl` 每行包含：

```json
{
  "phase": "joint",
  "epoch": 12,
  "seconds": 18.4,
  "learning_rate": 0.00018,
  "train": {"total": 0.71},
  "validation": {
    "pas": 0.82,
    "pdp": 0.86,
    "nmse": 0.49,
    "score": 0.79,
    "gate_accuracy": 1.0,
    "outage_accuracy": 0.94
  }
}
```

观察重点：

- `autoencoder` 阶段 PAS/PDP 是否持续提高；
- `gate_accuracy` 应较快接近 1；
- `outage_accuracy` 不能只看总准确率，还需后续补充 precision/recall；
- `joint` 阶段空间验证 `score` 是否高于 predictor；
- NMSE 突然爆炸通常是功率头学习率过高。

完整配置启用了按空间验证 score 的 early stopping。某阶段连续指定数量 epoch 未提高至少 `0.0001` 时会停止该阶段，并保留 `best_<stage>.pt`。

## 11. 独立验证 checkpoint

```bash
python scripts/evaluate.py \
  --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --device cuda \
  --stage joint
```

## 12. 生成测试信道

正式推理前扫描 outage 阈值：

```bash
python scripts/calibrate_outage.py \
  --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --device cuda \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

将输出的最佳阈值写回配置中的 `training.outage_threshold`，再执行推理。

```bash
python scripts/infer.py \
  --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --output outputs/Round2_Test_Channel.npy \
  --device cuda
```

不要使用 `--limit` 生成正式提交。`--limit` 只用于冒烟。

## 13. 检查提交文件

```bash
python -c "import numpy as np; p='outputs/Round2_Test_Channel.npy'; a=np.load(p,mmap_mode='r'); print(a.shape,a.dtype,np.isfinite(a).all(),a.nbytes)"
```

预期：

```text
(500, 256, 4, 192) complex64 True 786432000
```

还应检查：

```bash
python -c "import numpy as np; a=np.load('outputs/Round2_Test_Channel.npy',mmap_mode='r'); print('zero samples=',np.sum(np.all(a==0,axis=(1,2,3))))"
```

零信道数量不应机械等于训练集比例，应由模型和验证阈值决定。

## 14. 常见错误

### `CUDA was requested...false`

CUDA PyTorch 未正确安装，或远程任务没有分配 GPU。先检查 `nvidia-smi` 和 `torch.cuda.is_available()`。

### `out of memory`

减小 batch size。不要删除 AMP，也不要修改输出信道维度。

### `Missing manifest.json`

未执行预处理，或配置中的 `data.artifacts` 指向错误。

### `shape mismatch` 加载 checkpoint

训练配置与推理配置的 `latent_dim/base_channels/hidden_dim` 不一致。推理必须使用训练时的配置。

### 损失为 NaN

依次检查：

1. 单元测试是否通过；
2. 原始信道是否 `complex64`；
3. 是否启用了不受支持的 CPU float16；
4. 功率统计文件是否来自同一数据集；
5. 学习率是否被手工提高。

### Windows 多进程 DataLoader 卡住

将配置中的 `num_workers` 改为 0。Linux GPU 服务器通常可使用 2～4。

## 15. 正式训练验收条件

只有同时满足以下条件才进入提交推理：

- 单元测试全部通过；
- 三个阶段都完成；
- gate accuracy 接近 100%；
- outage 指标经过阈值检查；
- 空间验证 score 不再提高；
- 输出形状、dtype、finite 检查通过；
- 使用 `best.pt` 而不是 smoke/final checkpoint；
- 训练配置与 Git commit 已记录。
