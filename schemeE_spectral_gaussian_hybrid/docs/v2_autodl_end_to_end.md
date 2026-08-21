# Scheme E-v2 AutoDL 端到端教程

## 1. 分支和前提

代码分支：

```text
codex/0821_schemeE_v2
```

仓库根目录必须有：

```text
Round2_Map/Round2_Setup.json
Round2_Map/Round2_Map.ply
Round2_Map/Round2_Train_Pos.npy
Round2_Map/Round2_Train_Channel.npy
Round2_Map/Round2_Test_Pos.npy
```

AE 权重必须是实际二进制文件：

```text
schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

## 2. 拉取代码与 LFS

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git fetch origin
git switch codex/0821_schemeE_v2
git pull origin codex/0821_schemeE_v2

apt-get update
apt-get install -y git-lfs
git lfs install
git lfs pull --include="schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"
```

验证权重不是 134 字节的 LFS 指针：

```bash
stat -c '%s bytes' schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
head -c 40 schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

文件应大于 1 MB，开头不应出现 `version https://git-lfs...`。

## 3. 进入项目并检查 GPU

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 4. 安装依赖

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

不要随意覆盖 AutoDL 镜像中已经可用的 CUDA PyTorch。

## 5. 正式预检

```bash
python scripts/check_environment.py \
  --config configs/v2_5090.json \
  --require-cuda --strict-boosters
python scripts/preprocess.py --config configs/v2_5090.json
python scripts/inspect_architecture.py --config configs/v2_5090.json
python -m unittest discover -s tests -v
bash scripts/run_v2_smoke.sh
```

必须看到：

```text
full_resolution_check = true
Scheme E-v2 reference-aware smoke PASS
```

## 6. 无人值守正式运行

```bash
mkdir -p logs
nohup env \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  BACKUP_ROOT=/root/autodl-fs/schemeE_v2_latest \
  bash scripts/run_v2_unattended.sh \
  > logs/v2_launcher.log 2>&1 &
echo $! > logs/v2_unattended.pid
```

流水线没有人工等待点，顺序固定为：

```text
环境检查
-> v2 冒烟
-> 严格 Fold0 六核先验
-> Attempt 1/2/3
-> 每次分基站策略扫描
-> 自动选最高 Fold0
-> 全 4000 条 OOF 教师
-> 全量测试教师
-> 全量 Hybrid
-> 500 条测试推理
-> 格式与 SHA256
-> 打包和持久盘备份
-> 自动关机
```

## 7. 查看日志

```bash
cat logs/v2_unattended_status.txt
tail -f logs/v2_launcher.log
```

若主日志路径已写入状态文件：

```bash
tail -f "$(awk -F= '/^log=/{print $2}' logs/v2_unattended_status.txt)"
```

按 `Ctrl+C` 只退出实时查看，不会停止训练。

检查进程和 GPU：

```bash
ps -fp "$(cat logs/v2_unattended.pid)"
nvidia-smi
```

## 8. 中断后恢复

重新开机后再次执行第 6 节命令。完整阶段会被跳过，Hybrid 从 `last.pt` 继续。不要删除：

```text
artifacts/v2/
artifacts/spectral/channel_targets.npz
artifacts/preprocessed_scheme_e/
logs/
```

## 9. 关键结果

```text
artifacts/v2/fold0/spectral_teacher/strict_report.json
artifacts/v2/fold0_attempt1/hybrid/summary.json
artifacts/v2/fold0_attempt2/hybrid/summary.json
artifacts/v2/fold0_attempt3/hybrid/summary.json
reports/generated/v2_attempt1_policy.json
reports/generated/v2_attempt2_policy.json
reports/generated/v2_attempt3_policy.json
reports/generated/v2_attempt_selection.json
configs/v2_final_selected.json
artifacts/v2/final/hybrid/best.pt
outputs/v2/Round2_Test_Channel.npy
reports/generated/schemeE_v2_final_EXPERIMENT_REPORT.md
schemeE_v2_results_YYYYmmdd_HHMMSS.tar.gz
```

持久盘备份：

```text
/root/autodl-fs/schemeE_v2_latest/
```

## 10. 完成后检查

无卡开机后：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
cat logs/v2_unattended_status.txt
cat reports/generated/v2_attempt_selection.json
cat reports/generated/v2_final_output_check.json
ls -lh outputs/v2/Round2_Test_Channel.npy
ls -lh artifacts/v2/final/hybrid/best.pt
ls -lh schemeE_v2_results_*.tar.gz
```

状态必须为 `SUCCESS`，输出 shape 必须是 `[500,256,4,192]`，dtype 必须是 `complex64`，且 finite 为 true。

## 11. 下载到本地

在 JupyterLab 左侧文件栏进入：

```text
/root/autodl-fs/schemeE_v2_latest/
```

下载压缩包、`.sha256` 和 `Round2_Test_Channel.npy`。本地校验：

```powershell
Get-FileHash .\schemeE_v2_results_*.tar.gz -Algorithm SHA256
```

结果应与 `.sha256` 中记录一致。

## 12. Git LFS 发布结果

训练产物很大，普通 `git add` 不适合。先确认 GitHub LFS 额度，然后在无卡实例执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch codex/0821_schemeE_v2
git lfs install
git lfs track "schemeE_spectral_gaussian_hybrid/outputs/v2/*.npy"
git lfs track "schemeE_spectral_gaussian_hybrid/artifacts/v2/final/hybrid/*.pt"

git add .gitattributes
git add -f \
  schemeE_spectral_gaussian_hybrid/outputs/v2/Round2_Test_Channel.npy \
  schemeE_spectral_gaussian_hybrid/artifacts/v2/final/hybrid/best.pt
git add \
  schemeE_spectral_gaussian_hybrid/reports/generated/schemeE_v2_final_EXPERIMENT_REPORT.md \
  schemeE_spectral_gaussian_hybrid/reports/generated/schemeE_v2_final_experiment_report.json

git commit -m "Publish Scheme E-v2 trained results"
git push origin codex/0821_schemeE_v2
git lfs ls-files
```

若 LFS 网络慢，优先保留 AutoDL 持久盘和本地压缩包，不要反复重新训练。
