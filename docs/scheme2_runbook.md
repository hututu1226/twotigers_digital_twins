# 方案二代码运行说明

## 1. 前置准备

方案二与方案一共用 Python 环境、原始数据和预处理缓存。首次使用应先完成：

```bash
python -m pip install -e .
python scripts/preprocess.py
python -m unittest discover -s tests -v
```

环境、数据目录和 CUDA 安装细节见 [方案一运行说明](scheme1_runbook.md)。本文件重点说明方案二特有参数和故障。

## 2. 一键双方案冒烟

```bash
python scripts/smoke_test.py --device cpu --samples 2
```

最终必须显示：

```text
PASS scheme1
PASS scheme2
PASS all smoke tests
```

方案二冒烟只训练 12 条样本、1 个 epoch。初始 NMSE 或总 loss 很大是正常现象，只要满足：

- loss 为有限数；
- backward 成功；
- checkpoint 保存成功；
- 推理数组为 `complex64`；
- 形状为 `(2,256,4,192)`；
- 不包含 NaN/Inf。

## 3. 单独运行方案二冒烟

```bash
python scripts/train.py --config configs/scheme2_smoke.json --device cpu
```

输出 checkpoint：

```text
artifacts/runs/scheme2_smoke/best.pt
artifacts/runs/scheme2_smoke/final.pt
artifacts/runs/scheme2_smoke/last.pt
```

限量推理：

```bash
python scripts/infer.py \
  --config configs/scheme2_smoke.json \
  --checkpoint artifacts/runs/scheme2_smoke/final.pt \
  --output artifacts/smoke/scheme2_test_channel_2.npy \
  --device cpu \
  --limit 2
```

## 4. RTX 4070 完整训练

桌面 RTX 4070 12GB：

```bash
python scripts/train.py --config configs/scheme2_4070.json --device cuda
```

默认：

```text
batch_size = 8
K = 32
d_model = 128
layers = 4
epochs = 350
AMP = true
```

预期输出目录：

```text
artifacts/runs/scheme2_4070/
  resolved_config.json
  history.jsonl
  last.pt
  best_joint.pt
  best.pt
  final.pt
```

## 5. 8GB Laptop 4070 调整

首先只修改：

```json
"batch_size": 4
```

如果 OOM：

```text
batch_size: 4 -> 2
token_count: 32 -> 24
model_dim: 128 -> 96
feedforward_dim: 256 -> 192
```

`model_dim` 必须能被 `attention_heads` 整除。例如 `model_dim=96, heads=4` 合法。

不建议关闭 AMP；关闭后显存占用和训练时间都会增加。

## 6. 关键配置解释

### `token_count`

稀疏传播簇数量。过少会欠拟合，过多会增加合成成本并可能产生重复 Token。首轮使用 32。

### `model_dim`

Transformer 隐藏维度。主要影响 Token 之间关系建模，不直接改变输出尺寸。

### `transformer_layers`

Token 交互层数。建议从 2 或 4 开始。更深网络对 4000 条数据未必更好。

### `feedforward_dim`

Transformer FFN 宽度，通常取 `2 * model_dim`。

### `fourier_bands`

坐标多频编码数量。过高容易记忆训练点，必须根据空间验证选择。

### `outage_threshold`

推理硬置零阈值。0.5 只是初始值，应在验证集扫描例如：

```text
0.2, 0.3, 0.4, 0.5, 0.6, 0.7
```

自动扫描：

```bash
python scripts/calibrate_outage.py \
  --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --device cuda
```

## 7. 训练日志解读

早期常见现象：

- `loss` 很大：功率头尚未校准，NMSE 主导；
- PAS/PDP 先上升，NMSE 后改善：属于预期；
- gate 很快接近 1：两个空间区域容易区分；
- outage accuracy 高但输出全非零：类别不平衡，需要检查召回率；
- loss 数十 epoch 不动：可能 Token 宽度饱和或学习率不合适。

收敛判断不要只看训练 loss。建议至少满足：

- 空间验证 score 连续 30～50 epoch 无明显提高；
- PAS/PDP 不再提高；
- NMSE 没有持续恶化；
- 不同基站分数没有严重失衡。

## 8. 断点续训

```bash
python scripts/train.py \
  --config configs/scheme2_4070.json \
  --device cuda \
  --resume artifacts/runs/scheme2_4070/last.pt
```

不要在恢复时修改 K、model_dim 或层数。

## 9. 独立评估

```bash
python scripts/evaluate.py \
  --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --device cuda \
  --stage joint
```

## 10. 正式推理

```bash
python scripts/infer.py \
  --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --output outputs/Round2_Test_Channel_scheme2.npy \
  --device cuda
```

验证：

```bash
python -c "import numpy as np; a=np.load('outputs/Round2_Test_Channel_scheme2.npy',mmap_mode='r'); print(a.shape,a.dtype,np.isfinite(a).all())"
```

预期：

```text
(500, 256, 4, 192) complex64 True
```

最终提交文件必须按赛题要求重命名为：

```text
Round2_Test_Channel.npy
```

## 11. 方案比较

必须在同一个 `metadata.npz` 和空间验证划分下比较方案一、二。不要分别重新运行不同 seed 的预处理，否则分数不可比。

建议生成对比表：

```text
model | PAS | PDP | NMSE | score | gate_acc | outage_acc | params | train_time
```

若需要融合，先分别提交/验证，确认两个模型误差具有互补性。不能直接平均两个复信道；可先尝试按验证分数选择模型，或在角度-时延功率谱层进行经过验证的融合。

## 12. 常见故障

### Transformer 参数不整除

报错通常为 `embed_dim must be divisible by num_heads`。保证：

```text
model_dim % attention_heads == 0
```

### loss 突然 NaN

可能原因：

- 学习率过高；
- Token 宽度过小；
- 手工删除了最小宽度限制；
- AMP 下添加了不稳定的复数算子；
- 功率统计文件与数据不匹配。

先将学习率降低一半，再检查第一个出现 NaN 的 batch。不要用 `nan_to_num` 掩盖训练错误。

### Token 输出过度平滑

症状是 PDP/PAS 峰值位置大致正确但余弦不高。可尝试：

- K 从 32 增至 48；
- 放宽最小宽度；
- 增加一个小型卷积残差头；
- 提高 AD/PAS/PDP 损失相对功率损失的比例。

### Token 塌缩

多个中心和宽度几乎相同。可加入 Token 中心分散正则，但应作为独立实验，不要未经验证直接加入主配置。

### GPU 利用率低

检查：

- 数据是否位于本地 NVMe 而非网络盘；
- `num_workers` 是否为 2～4；
- AMP 是否开启；
- batch size 是否太小；
- 是否每个 epoch 重复执行预处理。

## 13. 正式训练验收条件

- 一键冒烟通过；
- 350 epoch 或早停标准满足；
- `best.pt` 存在；
- 验证指标为有限值；
- gate/outage 行为合理；
- 完整 500 条推理通过 shape/dtype/finite 检查；
- 使用与 checkpoint 相同的配置；
- 输出文件未使用 `--limit`。
