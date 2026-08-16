# 4090 / 4090D 训练时长与资源估算

版本：2026-08-16

## 1. 结论

在单张 24 GB 4090 或 4090D、CUDA AMP 开启、数据位于本地 SSD 的条件下，当前默认配置建议按以下范围预留时间：

| 工作 | 保守范围 |
| --- | ---: |
| Fold0 全流程 | 2.0-4.5 小时 |
| Final 全量重训与推理 | 1.5-3.0 小时 |
| Fold0 + Final + 打包 | 3.5-7.5 小时 |

若你的实际 epoch 已经稳定在 10-15 秒，则应相信实测：

```text
180 epochs * 10-15 s = 30-45 min
220 epochs * 10-15 s = 37-55 min
60 epochs  * 10-15 s = 10-15 min
```

再加验证、latent 编码、阈值扫描、推理和 I/O，Fold0 大约 1.5-2.5 小时，而不是机械地按更保守上限等待。脚本已提供基于真实日志的动态估算。

## 2. 为什么只能给范围

训练时长不仅由 GPU 型号决定，还受以下因素影响：

- 4090 与 4090D 的具体主机功耗/频率限制；
- AutoDL 主机分配的 CPU 核数和内存；
- 数据盘吞吐与其他任务竞争；
- PyTorch/CUDA/cuDNN 版本；
- AE batch 是否因 OOM 从 8 降为 4；
- Context 动态盲区每步实际 target 数；
- 每隔若干 epoch 的完整 Fold0 解码验证；
- early stopping 的实际触发 epoch。

因此“单 epoch 实测时间 x 剩余 epoch”比纯理论估计可靠。

## 3. 当前模型规模

正式配置的静态规模为：

| 模块 | 参数量 |
| --- | ---: |
| Structured AE v2 | 930,496 |
| Context field + 两个 BS heads | 19,344,933 |
| 合计 | 20,275,429 |
| FP32 纯参数体积 | 77.34 MiB |

潜变量维度：

```text
spectrum = 3072
phase    = 1536
total    = 4608
```

参数文件并不算大，但训练显存主要由以下内容占用：

- 3D AE 中间激活；
- 完整解码后的 `[B,16,8,16,192]` 实张量；
- complex `[B,256,4,192]` 指标张量；
- Context 每步最多 24 个 target 的解码图；
- AdamW 的参数、梯度和两组动量。

这就是“只有约 20M 参数”仍可能使用十几 GB 显存的原因。模型参数量不能直接换算训练时间或显存。

## 4. Fold0 分阶段估算

### 4.1 双分辨率预处理

任务：扫描约 3 GB 训练信道计算功率，读取 19,378 个点云顶点，生成两基站 3 m/1 m BEV。

当前本地 CPU 实测：约 12.45 秒。AutoDL 根据磁盘和首次缓存情况，建议按 0.5-3 分钟预留。该步骤只需执行一次，Smoke/Fold0/Final 可复用。

### 4.2 Structured AE

配置：

```text
训练样本 3208（Fold0 之外 3435 条，再排除 227 条 outage）
batch_size = 8
epochs = 180，early stopping patience = 24
每个 batch 做 full decode + spectrum-only decode
每 2 epoch 做一次验证
```

估计：每 epoch 20-45 秒，总计约 60-135 分钟；若实测 10-15 秒，则约 30-45 分钟。

为什么比旧 AE 更重：

- 双 encoder 分支；
- 每个训练 batch 解码两次；
- 直接计算 PAS/PDP/NMSE；
- latent 从旧扁平 256 提高到结构化 4608；
- 验证同时计算 full 和 spectrum-only ceiling。

### 4.3 编码 4000 条训练信道

只运行 encoder，不反向传播。估计 1-5 分钟，包括压缩写入 `encoded.npz`。

### 4.4 Context field

配置：

```text
96 dynamic holes / epoch
220 epochs
每 5 epoch 完整验证
3 m full-map Gated FPN
1 m environment encoder
最多 24 targets / step
```

估计：每 epoch 10-30 秒，总计 40-110 分钟。完整验证 epoch 会明显更慢。

该阶段不按 `4000 / batch_size` 计时，因为一个 epoch 定义为 96 个随机盲区，而不是遍历 4000 条样本一次。

### 4.5 Joint fine-tuning

配置：64 dynamic holes/epoch，60 epochs。估计 10-35 分钟。

虽只解冻 AE decoder，但仍需进行完整信道解码和指标损失反向传播，所以不会按“只训练一小部分参数”同比例缩短。

### 4.6 评估、阈值扫描和测试推理

建议预留 10-30 分钟：

- 三个阶段各一次完整验证；
- 6 个 outage threshold；
- 500 条测试 latent 预测和信道解码；
- 写入约 0.732 GiB 的 `complex64` NPY。

阈值扫描只运行一次 FPN 和信道解码，然后同时累计所有阈值的指标；因此开销接近一次完整验证，而不是 6 次完整推理。

## 5. Final 为什么通常更短

Final 不再搜索模型：

- 使用 Fold0 best epoch 生成 `final_selected.json`；
- 不做 Fold0 验证；
- 不重复 outage scan；
- 只在全部 4000 条训练样本上重训并推理。

AE 全量阶段有 3738 条非 outage 样本，比 Fold0 的 3208 条多约 16.5%，但没有周期验证；Context 每 epoch 仍是固定 96 个盲区，因此总体通常比 Fold0 短。

## 6. 如何得到你这台 4090 的真实估算

训练至少完成 5 个 epoch 后：

```bash
python scripts/estimate_runtime.py --config configs/fold0_4090.json --recent 5
```

示例输出字段：

```json
{
  "stage": "autoencoder",
  "completed_epochs": 10,
  "configured_epochs": 180,
  "average_epoch_seconds": 14.2,
  "estimated_remaining_minutes": 40.23,
  "estimated_full_stage_minutes": 42.6
}
```

训练时也可看最后几条日志：

```bash
tail -n 5 artifacts/fold0/autoencoder/history.jsonl
tail -n 5 artifacts/fold0/context/history.jsonl
tail -n 5 artifacts/fold0/joint/history.jsonl
```

每条记录都有 `elapsed_seconds`。命令行每 epoch 也会打印 `seconds=...`。

## 7. 显存估计与 OOM 后调整

默认配置针对 24 GB 卡，预计峰值约 14-22 GB，但具体值需要在目标环境实测。执行：

```bash
watch -n 2 nvidia-smi
```

若 OOM，优先这样改：

```text
AE batch_size              8 -> 4
AE gradient_accumulation   1 -> 2
AE validation_batch_size   8 -> 4
Encoding batch_size       16 -> 8
Context maximum_targets   24 -> 12
validation decode batch    8 -> 4
Inference decode batch     8 -> 4
```

这种修改基本保持有效 batch/模型容量。不要第一时间缩减 spectrum/phase latent，因为 AE ceiling 是本方案的关键。

## 8. 磁盘容量

建议至少预留 25-35 GB 可用空间：

| 内容 | 典型量级 |
| --- | ---: |
| 原始 Round2 数据 | 约 5 GB |
| 双分辨率预处理 | 数十 MB |
| encoded latent | 约 40-80 MB |
| 多阶段 checkpoint + Adam 状态 | 数百 MB到数 GB |
| 正式测试 NPY | 0.732 GiB |
| 最终 tar.gz | 约 1-数 GB，取决于权重压缩率 |

Fold0 与 Final checkpoint 同时保留，加上打包过程需要“原文件 + 压缩包”双份空间，所以不要只按最后一个输出文件估计。

## 9. 计费时间判断

GPU 是否正在计算不决定按量计费；实例处于开机状态才是计费依据。训练结束后先完成：

1. 输出格式检查；
2. `package_results.sh`；
3. SHA256 校验；
4. 将压缩包下载到本地或复制到可靠文件存储；
5. 再在控制台关机。

不要仅看到最后一个 epoch 结束就立刻关机，否则评估、测试输出和打包还没完成。
