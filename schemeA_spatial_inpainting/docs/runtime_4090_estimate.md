# RTX 4090 训练时长预估

## 1. 结论

当前正式配置在单张标准 RTX 4090 24GB 上，建议按以下预算租用：

| 工作 | 预计时间 |
|---|---:|
| 首次环境、数据复制解压、CUDA 冒烟 | 10~25 分钟 |
| fold0 严格开发训练、验证、阈值扫描、测试生成 | 1.5~3.5 小时 |
| 4000 条全量 final 重训与测试生成 | 1.5~3 小时 |
| 打包并复制到文件存储 | 5~15 分钟 |
| 推荐完整流程合计 | **3.5~7 小时** |

如果只跑 fold0 开发模型并生成测试集，通常准备 **2~4 小时**。如果做完整 5 折并为每折严格重训 AE，再加全量 final，准备 **9~20 小时**。

4090D 与标准 4090 的具体差距受实例功耗限制、CPU 和磁盘影响。保守上可给上述区间再留 15% 余量。

这不是收敛保证。模型是否“充分收敛”必须由固定空间验证的 official score、PAS、PDP、NMSE 曲线判断，而不是只看 epoch 数。

## 2. 预估依据

### 2.1 已有同项目 4090D 日志

仓库已有上一轮真实 GPU 日志，测得：

| 阶段 | 平均 epoch | 中位数 | 实际完成 epoch |
|---|---:|---:|---:|
| 旧 Scheme1 AE | 6.95 s | 6.84 s | 57 |
| 旧 Scheme1 predictor | 11.30 s | 11.42 s | 47 |
| 旧 Scheme1 joint | 13.69 s | 13.85 s | 31 |
| 旧 Scheme2 joint | 15.88 s | 15.86 s | 58 |

这证明当前数据和 AE 在 4090D 上并不是“每 epoch 数分钟”。但新方案空间阶段的一个 epoch 与旧方案定义不同，不能直接用 `15 s x 180` 得出精确总时长。

### 2.2 新方案计算规模

正式模型参数：

```text
Angle-Delay AE: 1,937,136
Spatial U-Net:  2,012,098
Total:          3,949,234
fp32 weights:   约 15.1 MiB
```

参数量不大，耗时来自数据张量和变换：

1. 每条信道有 `256 x 4 x 192 = 196,608` 个 complex64 值。
2. AE 使用 3D 卷积处理 `[8,8,16,192]` Angle-Delay 张量。
3. 每个训练 batch 还要执行空间 FFT、频率 IFFT、逆变换和复数 NMSE。
4. U-Net 每 epoch 处理 96 个 `269 x 96 x 96` crop。
5. 每个 hole 最多解码 24 条完整复信道；batch=2 时最多同时解码 48 条。
6. 每 5 个空间 epoch 还会在约 823 条固定验证样本上做整图推理和完整指标计算。

因此模型权重很小并不意味着训练只有几分钟。正式计算瓶颈是 3D AE Decoder、完整复信道张量和 FFT 指标。

## 3. 分阶段预算

### 3.1 预处理

本机 CPU 从 6.29 GB NPY 完成 1 m 正式预处理实测约 13 秒。AutoDL 本地 NVMe 一般为 15~60 秒；如果错误地直接从 `/root/autodl-fs` 网络存储读取，可能明显变慢。

### 3.2 Stage A AE

fold0 约有 3000 条训练样本，排除 outage 后 batch 数约 370；全量约 3738 条非零样本，batch 数约 468。

```text
fold0: 120 epoch 上限，预计 15~30 分钟
final: 由最佳 epoch 决定，预计 15~35 分钟
```

开发配置每 2 epoch 验证一次，并有 patience=20。若曲线提前停止，实际时间会更短。

### 3.3 Latent 编码

4000 条信道只前向编码一次，预计 1~3 分钟，包括 NPY mmap I/O 和统计量保存。

### 3.4 Stage B Spatial U-Net

这是主要不确定项：

```text
96 crops / epoch
batch = 2
48 optimizer batches / epoch
12~48 m dynamic hole
最多 24 targets / crop
```

预计普通训练 epoch 约 20~50 秒；带固定空间验证的 epoch 会更长。180 epoch 上限约 1~2.5 小时，early stopping 可能提前结束。若实例 CPU 较弱、功耗受限或 `maximum_targets` 经常达到 24，可能接近区间上沿。

### 3.5 推理与打包

双站 U-Net 整图只各执行一次，随后 500 条 AE 解码。预计推理 1~5 分钟。最终 NPY 固定约 `0.732 GiB`；gzip 压缩和复制文件存储预计 5~15 分钟，取决于 CPU 和存储吞吐。

## 4. 上机后如何得到准确 ETA

静态预估只用于租卡预算。开始正式训练后，等待至少 5 个 epoch，再运行：

```bash
python scripts/estimate_runtime.py \
  --config configs/fold0_4090.json \
  --recent 5
```

输出示例：

```json
[
  {
    "stage": "autoencoder",
    "average_epoch_seconds": 8.2,
    "estimated_remaining_minutes": 13.7
  },
  {
    "stage": "spatial",
    "average_epoch_seconds": 31.5,
    "estimated_remaining_minutes": 78.8
  }
]
```

也可直接看日志：

```bash
tail -n 5 artifacts/fold0/autoencoder/history.jsonl
tail -n 5 artifacts/fold0/spatial/history.jsonl
```

每行的 `elapsed_seconds` 已包含该 epoch 触发的验证时间，因此最近 5 个 epoch 最好覆盖一次验证周期。

## 5. 显存与调参对时间的影响

标准配置 `batch_size=2` 预期可放入 24GB。若 OOM：

1. 先将 spatial batch 从 2 改成 1；
2. 再将 `maximum_targets` 从 24 改成 16；
3. 保持 latent_dim=256 和 crop=96，避免改变问题定义；
4. batch 减半后可用 gradient accumulation=2 保持有效 batch，但墙钟时间会增加。

不要同时在一张 4090 上跑两套正式 Scheme A。两套进程会争抢显存、算力和 NPY 磁盘读取，单任务 epoch 变慢，且更容易 OOM。fold0 与 final 应串行运行。

## 6. 何时算充分收敛

满足以下条件后才考虑停止开发训练：

- 固定 spatial validation official score 连续多个验证周期无改善；
- PAS/PDP 不再上升；
- NMSE 没有在总 score 掩盖下持续恶化；
- latent MSE 与 `log10 power` MAE/RMSE 已稳定；
- outage F1 与预测零信道数量合理；
- `best.pt` 的 epoch 明显早于上限时，early stopping 已正常触发。

如果 train loss 继续下降但固定空间 score 下降，应使用较早的 `best.pt`，而不是继续增加 epoch。

