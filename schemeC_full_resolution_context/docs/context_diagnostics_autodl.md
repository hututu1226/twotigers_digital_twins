# Scheme C Context 诊断实验 AutoDL 教程

## 1. 这次会做什么

诊断脚本只读取已经训练好的 Context、AE 和 encoded latent，不会继续训练，也不会改写
`best.pt`。它会一次完成：

1. 分别把 Spectrum、Detail、Power 和 outage 换成验证集真值；
2. 计算最近邻真实 latent 与原始信道复制基线；
3. 比较 Context 功率、最近邻功率和 8 邻居功率基线；
4. 临时关闭 Warp，或者减弱 Router 对 Attention 的强制影响；
5. 生成 JSON、Markdown 和可下载压缩包。

最近邻只是一把诊断用的尺子，不会重新成为最终算法。

## 2. 预计资源和时间

- 推荐设备：AutoDL 5090；
- 正式 Fold0 样本：565 条；
- 预计时间：5～15 分钟，建议预留 20 分钟；
- 不需要重新训练 AE 或 Context；
- 不会生成新的大模型权重。

## 3. 更新代码

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0817_schemeC
git pull origin 0817_schemeC
git lfs pull
cd schemeC_full_resolution_context
```

`git pull` 不会删除 `artifacts` 中已经训练好的忽略文件。
正式 Fold0 的 Context `best.pt` 已通过 Git LFS 保存；`git lfs pull` 会把它从
约 130 字节的指针文件替换成约 275 MiB 的真实权重。

## 4. 运行前检查

```bash
nvidia-smi

ls -lh artifacts/fold0/context/best.pt
ls -lh artifacts/fold0/autoencoder/best.pt
ls -lh artifacts/fold0/encoded.npz
ls -lh artifacts/preprocessed_scheme_c/metadata.npz
ls -lh ../Round2_Map/Round2_Train_Channel.npy
```

五个文件都必须存在。再检查 CUDA：

```bash
python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

`cuda_available` 必须为 `True`。

## 5. 先跑 16 条快速检查

```bash
LIMIT=16 bash scripts/run_context_diagnostics.sh
```

结束时必须看到：

```text
"status": "PASS"
Created schemeC_context_diagnostics_时间戳.tar.gz
```

16 条样本的分数没有统计意义，只检查正式权重、数据和所有诊断分支能否跑通。

## 6. 跑完整 565 条诊断

```bash
LIMIT=0 bash scripts/run_context_diagnostics.sh
```

脚本默认使用：

```text
configs/fold0_5090.json
artifacts/fold0/context/best.pt
```

不需要手动填写 outage threshold。脚本优先读取 checkpoint 保存的最佳阈值。

## 7. 查看进度

运行过程中会依次显示：

```text
[1/4] Predicting the trained Context baseline...
[2/4] Replacing Spectrum, Detail, Power, and outage with truth...
[3/4] Measuring nearest-latent and low-dimensional power baselines...
[4/4] Running no-Warp and weaker Router-prior counterfactuals...
```

另开一个 SSH 窗口可以实时查看日志：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
tail -f logs/context_diagnostics.log
```

按 `Ctrl+C` 只退出日志查看，不会停止另一个窗口中的诊断进程。

## 8. 输出文件

```bash
ls -lh artifacts/fold0/context_diagnostics/report.json
ls -lh artifacts/fold0/context_diagnostics/SUMMARY.md
ls -lh schemeC_context_diagnostics_*.tar.gz
ls -lh schemeC_context_diagnostics_*.tar.gz.sha256
```

文件含义：

- `report.json`：完整数字，后续分析以它为准；
- `SUMMARY.md`：便于人工浏览的表格；
- `tar.gz`：需要下载并发回分析的压缩包；
- `sha256`：检查压缩包下载后是否损坏。

压缩包不包含大模型权重，因此体积很小。

## 9. 报告中最重要的字段

```text
baseline.metrics.score
oracle_replacements.oracle_spectrum.score_delta
oracle_replacements.oracle_detail.score_delta
oracle_replacements.oracle_power.score_delta
spatial_baselines.nearest_latent.metrics.score
counterfactuals.no_warp.score_delta
counterfactuals.weak_router_prior_0_25.score_delta
baseline.routing.router_top1_mass
baseline.routing.router_effective_neighbors
```

通俗地说：

- `oracle_detail.score_delta` 最大：Detail 是第一优先问题；
- `nearest_latent` 高于 Context：Router/融合没有利用好邻居；
- `no_warp` 几乎不掉分：现有 Warp 基本没有贡献；
- 弱 Router prior 提升：当前 Router 对 Attention 压制过强；
- `oracle_power` 明显提升：值得单独建立 PowerNet。

反事实分支没有重新训练，所以小幅提升只能用于判断方向，不能当作新模型最终成绩。

## 10. 可选自动关机

确认想在诊断和打包成功后自动关闭 AutoDL 实例，再执行：

```bash
CONFIRM_AUTODL_SHUTDOWN=YES LIMIT=0 bash scripts/run_context_diagnostics.sh
```

脚本只有在 `report.json` 的状态为 `PASS` 且压缩包生成成功后才调用
`/usr/bin/shutdown`。第二天仍应在 AutoDL 控制台确认实例状态为“已关机”。

## 11. 常见错误

### 缺少 encoded.npz

```bash
python scripts/encode_latents.py --config configs/fold0_5090.json
```

然后重新运行诊断。

### 缺少 Context best.pt

说明正式 Context 权重不在当前实例，不能通过重新 encode 解决。需要把之前保存的
`artifacts/fold0/context/best.pt` 上传或解压回原路径。

### 报告停留在 RUNNING

说明任务中途被终止。旧的部分报告可以保留，但不能用于最终决策；直接重新执行完整命令即可覆盖。
