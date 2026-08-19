# Scheme E AutoDL 端到端运行说明

## 1. 前提

- 仓库分支：`0819`。
- 数据目录：仓库根目录 `Round2_Map/`。
- 已安装支持当前显卡的 CUDA PyTorch。
- Scheme C AE 权重存在于 `schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt`。
- 正式建议 5090，至少预留 8 GiB 磁盘；生成的测试 NPY 单文件约 0.73 GiB。

## 2. 拉代码和 LFS 权重

```bash
cd /root/autodl-tmp
git clone https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins
git switch 0819

apt-get update
apt-get install -y git-lfs
git lfs install
git lfs pull --include="schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"
```

验证下载到的不是 134 字节 LFS 指针：

```bash
stat -c '%s bytes' schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
head -c 40 schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

大小应约 100 MB；开头不应是 `version https://git-lfs...`。

## 3. 检查数据

```bash
ls -lh Round2_Map/Round2_Setup.json
ls -lh Round2_Map/Round2_Map.ply
ls -lh Round2_Map/Round2_Train_Pos.npy
ls -lh Round2_Map/Round2_Train_Channel.npy
ls -lh Round2_Map/Round2_Test_Pos.npy
```

数据不通过 Git 上传，需要用 AutoDL 文件管理器、网盘或持久盘放到上述位置。

## 4. 安装依赖

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
python -m pip install -U pip
python -m pip install -r requirements.txt
```

不要在已经可用的 AutoDL CUDA 镜像里随意覆盖 `torch`。先检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 5. 冒烟测试

```bash
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log
```

它使用 32 条合成信道、2 折和 1 epoch，完整跑过预处理、教师模型、Fold0、全量阶段和 NPY 检查。最后必须看到 `Scheme E smoke test PASS`。

## 6. 正式预检

```bash
python scripts/check_environment.py \
  --config configs/fold0_5090.json \
  --require-cuda --strict-boosters
python scripts/preprocess.py --config configs/fold0_5090.json
python scripts/inspect_architecture.py --config configs/fold0_5090.json
```

结构检查依赖预处理产生的 `manifest.json`，因此不能交换后两条命令。必须满足：CUDA 可用、XGBoost/LightGBM 均存在、AE 大于 1 MB、`total_latent_elements=30720`、`full_resolution_check=true`。

## 7. 一条命令跑完

前台运行：

```bash
set -o pipefail
bash scripts/run_all_5090.sh 2>&1 | tee logs/formal.log
```

无人值守并自动关机：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
mkdir -p logs
nohup env SHUTDOWN_ON_SUCCESS=1 SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_unattended.sh \
  > logs/launcher.log 2>&1 &
echo $! > logs/unattended.pid
```

串行顺序固定为：预检 → 高斯/71 维特征 → 频谱目标 → 八折 OOF 教师 → Fold0 Hybrid → 分项报告 → 选择 epoch/投影轮数 → 全量教师 → 全量 Hybrid → 500 测试输出 → 校验 → 打包 → 持久盘备份 → 关机。

## 8. 查看实时日志

```bash
tail -f logs/launcher.log
```

或另开终端：

```bash
cat logs/unattended_status.txt
tail -f "$(awk -F= '/^log=/{print $2}' logs/unattended_status.txt)"
```

按 `Ctrl+C` 只退出 `tail`，不会停止训练。确认进程：

```bash
ps -fp "$(cat logs/unattended.pid)"
nvidia-smi
```

## 9. 中断后继续

重新开机后再次执行无人值守命令即可。程序会：

- 跳过完整的预处理/频谱目标；
- 按“折 × 基站”读取 GP 进度缓存；
- 从 Hybrid `last.pt` 继续；
- 跳过已通过完成性检查的阶段。

不要删除 `artifacts/*/progress/`、`last.pt` 和 `history.jsonl`。

## 10. 结果位置

```text
artifacts/fold0/spectral_teacher/oof_report.json
artifacts/fold0/hybrid/best.pt
artifacts/fold0/hybrid/summary.json
reports/generated/schemeE_fold0_EXPERIMENT_REPORT.md
artifacts/final/spectral_teacher/model.pkl
artifacts/final/hybrid/best.pt
outputs/final/Round2_Test_Channel.npy
reports/generated/schemeE_final_EXPERIMENT_REPORT.md
schemeE_results_YYYYmmdd_HHMMSS.tar.gz
```

持久盘备份默认在 `/root/autodl-fs/schemeE_latest/`。关机不会删除 `/root/autodl-tmp` 和 `/root/autodl-fs` 中的文件，但仍应以压缩包和 SHA256 为最终备份依据。

## 11. 5090 时间预估

这是估计，不是实测承诺：

- 预处理和频谱提取：10–30 分钟；
- 八折 OOF GP：45–120 分钟；
- Fold0 Hybrid：1.5–4 小时，配置有 4 小时训练上限；
- Fold0 分项验证：10–25 分钟；
- 全量 GP：20–60 分钟；
- 全量 Hybrid：约 1–3 小时，取决于 Fold0 最佳 epoch；
- 推理、校验、压缩：25–70 分钟。

合计通常约 4.5–9 小时。Exact GP、反复解码官方指标和 0/2/4/8 轮投影验证是时间较长的原因。

## 12. 在无卡实例上传 LFS

训练关机后可开无 GPU 实例，避免显卡计费：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0819
apt-get update && apt-get install -y git-lfs
bash schemeE_spectral_gaussian_hybrid/scripts/publish_lfs_results.sh
git lfs ls-files
```

脚本使用 `git add -f` 添加被 `.gitignore` 忽略的正式权重/NPY，并由根目录 `.gitattributes` 交给 LFS。单个测试 NPY 约 0.73 GiB，请先确认 GitHub LFS 存储和流量额度。LFS 上传不需要 GPU。
