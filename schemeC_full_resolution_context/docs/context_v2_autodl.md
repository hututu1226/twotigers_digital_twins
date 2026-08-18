# Context V2 在 AutoDL 5090 上的完整运行教程

本文只适用于 `geometry_warped_context_v2`。它默认复用已经通过 `0.9491` Fold0 验证的 AE v4，只重新编码 latent、训练一次端到端 Context V2、验证、生成 500 条测试信道、打包并关机。

Context V2 没有单独 Joint 阶段。旧教程中的 `finetune_joint.py` 不应在本次运行中执行。

## 1. 时间和产物目标

单次 Fold0 租卡目标：

| 工作 | 5090 工程预算 |
|---|---:|
| 代码与环境检查 | 5～15 分钟 |
| 重新编码 4000 条 latent | 5～20 分钟 |
| Context V2 正式训练 | 最多 6.75 小时 |
| 验证、阈值扫描、500 条推理与打包 | 20～50 分钟 |
| 总计 | 目标控制在 8 小时以内 |

这是配置预算，不是硬件速度保证。训练脚本在一个 epoch 结束后检查墙钟时间，因此可能超过限制一个 epoch。

本次 Fold0 成功后应得到：

```text
artifacts/fold0/context/best.pt
artifacts/fold0/context/final.pt
artifacts/fold0/context/evaluation.json
artifacts/fold0/context/outage_scan.json
artifacts/fold0/context_mask_report.json
artifacts/fold0/stage_gap.json
outputs/fold0/Round2_Test_Channel.npy
schemeC_fold0_时间戳.tar.gz
schemeC_fold0_时间戳.tar.gz.sha256
```

## 2. 进入项目目录

你此前的目录为：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
pwd
```

`pwd` 应输出以上目录。后续所有命令都在这里执行。

## 3. 更新代码

回到 Git 仓库根目录：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git status
git switch 0817_schemeC
git pull origin 0817_schemeC
git log -1 --oneline
```

如果 GitHub HTTP/2 再次中断：

```bash
git config --global http.version HTTP/1.1
git pull origin 0817_schemeC
```

然后返回 Scheme C：

```bash
cd schemeC_full_resolution_context
```

不要执行 `git clean -fd`、`git reset --hard` 或删除整个 `artifacts/fold0`，因为其中包含已经训练好的 AE 权重。

## 4. 确认正式 AE 仍在

```bash
ls -lh artifacts/fold0/autoencoder/best.pt
cat artifacts/fold0/autoencoder/evaluation.json
cat artifacts/fold0/autoencoder/quality_gate.json
python scripts/verify_completion.py --stage ae
```

应看到：

```text
score ≈ 0.9491
quality_gate = PASS
context_training_allowed = true
```

如果 `verify_completion.py --stage ae` 失败，不要直接开始 Context。先检查 AE 文件是否上传完整。

## 5. 安装或检查环境

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py \
  --config configs/fold0_5090.json \
  --require-cuda
nvidia-smi
```

再检查 PyTorch：

```bash
python - <<'PY'
import torch
print('torch =', torch.__version__)
print('cuda =', torch.version.cuda)
print('available =', torch.cuda.is_available())
print('gpu =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
PY
```

`available` 必须为 `True`。否则不要开始正式计费训练。

## 6. 运行代码级检查

这些检查不会重新训练正式 AE：

```bash
python -m unittest discover -s tests -v
python scripts/inspect_architecture.py \
  --config configs/fold0_5090.json \
  --output artifacts/context_v2_architecture.json
python scripts/analyze_context_masks.py \
  --config configs/fold0_5090.json \
  --output artifacts/context_v2_mask_report.json
```

结构检查必须包含：

```text
context_architecture = geometry_warped_context_v2
context_parameters = 18614162
total_latent_elements = 30720
full_resolution_check = true
```

掩码检查应为 `PASS`，最近观测距离中位数通常约 `6.4 m`。

## 7. 可选 GPU 冒烟

正式训练前建议执行一次。它只使用 4 条训练和 2 条验证样本，产物写入 `artifacts/smoke`，不会覆盖 Fold0 AE：

```bash
python scripts/smoke_test.py \
  --config configs/smoke.json \
  --device cuda
python scripts/verify_completion.py --stage smoke
```

冒烟分数没有准确率意义，只检查训练、存权重、重新加载和测试推理能否完成。

## 8. 清理旧 Context 时的正确方式

不需要手动删除。正式脚本会运行：

```bash
python scripts/ensure_context_compatibility.py \
  --config configs/fold0_5090.json
```

如果检测到 Context V1 checkpoint，会重命名为：

```text
artifacts/fold0/context_archived_full_resolution_context_v1_时间戳
```

AE 目录不会被移动或删除。

## 9. 前台正式运行

如果白天可以观察日志：

```bash
mkdir -p logs
set -o pipefail
RESUME=1 bash scripts/run_fold0.sh \
  2>&1 | tee logs/context_v2_fold0.log
```

`run_fold0.sh` 顺序如下：

1. 检查数据和 AE 容量门禁。
2. 验证已有正式 AE，验证通过则跳过 AE 重训。
3. 检查 Context 训练掩码分布。
4. 用 AE best checkpoint 重新编码 4000 条 float32 latent。
5. 归档不兼容的旧 Context checkpoint。
6. 训练单阶段端到端 Context V2。
7. 扫描 outage threshold。
8. 重新计算正式 Fold0 Context 指标。
9. 输出 AE 到 Context 的 stage gap。
10. 生成 500 条测试信道并检查格式。

日志不应再出现 `Joint epoch=`。

## 10. 夜间自动运行并关机

先确认 AutoDL 文件存储目录存在：

```bash
ls -ld /root/autodl-fs
```

然后启动 Fold0 模式：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
mkdir -p logs
rm -f NO_AUTO_SHUTDOWN

nohup env \
  PIPELINE_MODE=fold0 \
  CONFIRM_AUTODL_SHUTDOWN=YES \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  BACKUP_DIR=/root/autodl-fs \
  bash scripts/run_overnight_pipeline.sh \
  > logs/context_v2_launcher.log 2>&1 &

echo $! | tee logs/context_v2.pid
```

该模式只完成 Fold0，不会继续训练全量 4000 条最终模型。

## 11. 监控日志

实时查看主日志：

```bash
tail -f logs/overnight_pipeline.log
```

退出实时查看但不停止训练：

```text
Ctrl+C
```

查看最近 50 行：

```bash
tail -n 50 logs/overnight_pipeline.log
```

查看状态：

```bash
cat logs/overnight_status.txt
ps -fp "$(cat logs/context_v2.pid)"
nvidia-smi
```

至少完成 3 个 epoch 后，可按真实日志估算剩余时间：

```bash
python scripts/estimate_runtime.py --config configs/fold0_5090.json --recent 3
```

日志中的关键字段：

```text
train=...
score=...
threshold=...
seconds=...
```

非验证 epoch 的 `score=nan` 只表示该 epoch 没有运行验证。

## 12. 取消自动关机

训练过程中若决定保持实例开启：

```bash
touch NO_AUTO_SHUTDOWN
```

脚本结束时发现该文件，就会跳过关机。

重新允许自动关机：

```bash
rm -f NO_AUTO_SHUTDOWN
```

## 13. 中断后续训

重新开机后：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
RESUME=1 bash scripts/run_fold0.sh \
  2>&1 | tee -a logs/context_v2_fold0.log
```

它会从 `artifacts/fold0/context/last.pt` 恢复模型、AE decoder、优化器、调度器和 AMP scaler。

不要用 V1 的 `last.pt` 强行续训。兼容性脚本会自动识别并归档。

## 14. 训练完成后的检查

```bash
python scripts/verify_completion.py \
  --stage fold0 \
  --output artifacts/fold0/completion_report.json

cat artifacts/fold0/context/evaluation.json
cat artifacts/fold0/context/outage_scan.json
cat artifacts/fold0/stage_gap.json
cat artifacts/fold0/context/summary.json
```

重点查看：

- `score` 是否达到或接近 0.70；
- PAS、PDP、NMSE 分别是多少；
- `phase_latent_mse_z` 是否明显高于 Spectrum；
- `router_entropy` 是否长期接近 0 或 1；
- `detail_warp_bins` 是否长期接近 0 或最大范围；
- best outage threshold 与预测 outage 数；
- `stop_reason` 是早停、时间限制还是最大 epoch。

## 15. 打包和持久化

夜间脚本会自动执行。手动执行方式：

```bash
bash scripts/package_fold0.sh
ls -lh schemeC_fold0_*.tar.gz*
sha256sum -c schemeC_fold0_*.tar.gz.sha256
```

复制到 AutoDL 文件存储：

```bash
cp -f schemeC_fold0_*.tar.gz* /root/autodl-fs/
sync
```

确认 `/root/autodl-fs` 中存在压缩包后再释放实例。

## 16. 下载到本地的最小文件

至少下载：

```text
schemeC_fold0_*.tar.gz
schemeC_fold0_*.tar.gz.sha256
artifacts/fold0/context/evaluation.json
artifacts/fold0/context/best.pt
artifacts/fold0/context/outage_scan.json
outputs/fold0/Round2_Test_Channel.npy
```

压缩包已经包含 AE、Context、配置、源码、日志环境信息和测试输出，因此通常下载压缩包与校验文件即可。

## 17. 何时运行最终全量训练

先把 Fold0 压缩包传回并分析。只有确认 Context V2 分数和诊断值得继续后，再在另一个租卡时段执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
mkdir -p logs
rm -f NO_AUTO_SHUTDOWN

nohup env \
  PIPELINE_MODE=final \
  CONFIRM_AUTODL_SHUTDOWN=YES \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  BACKUP_DIR=/root/autodl-fs \
  bash scripts/run_overnight_pipeline.sh \
  > logs/context_v2_final_launcher.log 2>&1 &

echo $! | tee logs/context_v2_final.pid
```

夜间脚本会先验证 Fold0，再自动生成 `configs/final_selected.json`，不需要提前手工运行 `prepare_final_config.py`。

Fold0 与最终全量训练不应塞进同一个 8 小时租卡窗口。

## 18. 常见错误

### 找不到 AE checkpoint

```bash
ls -lh artifacts/fold0/autoencoder/
```

不要开始 Context，先恢复 AE 分析包。

### CUDA out of memory

优先把两个正式配置中的：

```json
"inference_query_batch_size": 16
```

降为 8。若训练仍溢出，再把 `maximum_targets` 从 12 降为 8。不要先缩小 30,720 维 latent。

### encoded.npz 很大

这是预期行为。V2 使用 float32 和两个基站独立统计，避免 Detail 在进入 Context 前被再次量化。确保临时盘至少保留 10 GB 可用空间：

```bash
df -h /root/autodl-tmp
```

### 日志仍出现 Joint

如果日志是 `AE training stage -> joint` 或 `stage=joint`，它只表示 AE 内部的 Spectrum/Detail 两个分支一起训练，不是旧 Context Joint 阶段。

如果日志出现独立的 `Joint epoch=...`，才说明代码没有更新到 Context V2。执行：

```bash
git log -1 --oneline
grep -n 'geometry_warped_context_v2' configs/fold0_5090.json
```

### GitHub 拉取超时

```bash
git config --global http.version HTTP/1.1
git pull origin 0817_schemeC
```

仍失败时使用本地上传源码压缩包，不要反复删除已有 AE 权重。
