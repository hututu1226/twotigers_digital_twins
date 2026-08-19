# Scheme D AutoDL 端到端运行说明

## 1. 拉取分支和 AE LFS 权重

```bash
cd /root/autodl-tmp
git clone https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins
git switch 0819

apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull --include="schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"
stat -c '%s bytes' schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

权重应约 100 MB，不是 134 字节指针。

## 2. 数据与环境

```bash
ls -lh Round2_Map/Round2_Train_Channel.npy
cd schemeD_transport_residual_context
python -m pip install -r requirements.txt
python scripts/check_environment.py --config configs/fold0_5090.json --require-cuda
python scripts/inspect_architecture.py \
  --config configs/fold0_5090.json \
  --output reports/generated/architecture.json
```

结构报告必须是：`total_latent_elements=30720`、`full_resolution_check=true`、`transport_base_is_direct_full_resolution=true`、`residual_heads_zero_initialized=true`。

## 3. 冒烟测试

```bash
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log
```

冒烟会运行 21 个单元测试，再用少量真实数据训练 1 epoch 小 AE 和 1 epoch Context，最后生成 2 条测试信道。它验证代码可运行，不代表正式准确率。

## 4. 正式一键运行

```bash
set -o pipefail
bash scripts/run_all_5090.sh 2>&1 | tee logs/formal.log
```

顺序：环境/结构检查 → 复用 AE 编码 → Fold0 Context → outage 扫描 → Fold0 实验报告 → 选择 epoch → 4000 条全量编码 → 全量 Context → 500 测试输出 → 校验 → 最终报告 → 压缩。

## 5. 无人值守和自动关机

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeD_transport_residual_context
mkdir -p logs
nohup env SHUTDOWN_ON_SUCCESS=1 SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_unattended.sh \
  > logs/launcher.log 2>&1 &
echo $! > logs/unattended.pid
```

默认成功和失败都备份到 `/root/autodl-fs/schemeD_latest/` 后关机，避免训练报错后整夜计费。

## 6. 监控

```bash
tail -f logs/launcher.log
cat logs/unattended_status.txt
ps -fp "$(cat logs/unattended.pid)"
nvidia-smi
```

`Ctrl+C` 只退出日志查看，不终止后台训练。

日志中的关键字段：

- `score`：只有验证 epoch 有值；其他 epoch 为 `nan` 是正常的；
- `router_effective_neighbors`：目标是不再接近 1；
- `router_top1_mass`：越接近 1 越像退化最近邻；
- `spectrum_warp/detail_warp`：应非零但不能长期撞上限；
- `warp_saturation`：应接近 0；
- `spectrum_residual_rms/phase_residual_rms`：应平稳增长而非爆炸。

## 7. 中断恢复

再次执行相同无人值守命令。脚本会复用已有预处理/编码；Context 若有 `last.pt` 会恢复 optimizer、scheduler、AMP scaler 和早停状态；已完成并写出 `summary.json` 的阶段会跳过训练。

不要手工把 Fold0 `best.pt` 放到 Final 目录。Final 使用全部 4000 条重新训练，epoch 由 `prepare_final_config.py` 自动写入 `configs/final_selected.json`。

## 8. 结果

```text
artifacts/fold0/context/best.pt
artifacts/fold0/context/evaluation.json
artifacts/fold0/context/outage_scan.json
reports/generated/schemeD_fold0_EXPERIMENT_REPORT.md
artifacts/final/context/final.pt
outputs/final/Round2_Test_Channel.npy
reports/generated/schemeD_final_EXPERIMENT_REPORT.md
schemeD_results_YYYYmmdd_HHMMSS.tar.gz
```

验证：

```bash
python scripts/verify_completion.py \
  --config configs/final_selected.json --stage final
sha256sum -c schemeD_results_*.tar.gz.sha256
```

## 9. 5090 时间预估

结合旧日志每 epoch 约 10–15 秒，但 Scheme D 每 epoch 从 24 增到 32 steps，估计：

- 预处理复用时 0 分钟；首次预处理 5–15 分钟；
- 全量 AE latent 编码 5–20 分钟；
- Fold0 Context 通常 2–5 小时，最坏受 6.75 小时限制；
- outage 扫描和验证 20–50 分钟；
- Final Context 约 2–5.5 小时；
- 500 点推理、检查和打包 25–70 分钟。

总计约 4.5–11 小时。若最佳 epoch 在 200–400，通常更接近下半区；若持续提升到很晚，会接近上限。

## 10. 无卡上传 Git LFS

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0819
apt-get update && apt-get install -y git-lfs
bash schemeD_transport_residual_context/scripts/publish_lfs_results.sh
git lfs ls-files
```

LFS 上传不需要 GPU。测试 NPY 约 0.73 GiB，先确认 GitHub LFS 配额；上传过程中不要关闭无卡实例。
