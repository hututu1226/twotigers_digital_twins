# Scheme E-v3 AutoDL 5090 端到端教程

## 1. 最终要运行的命令

正式无人值守入口只有一个：

```bash
bash scripts/run_v3_unattended.sh
```

它会串行完成环境检查、冒烟测试、严格 Fold0 三次实验、自动选优、4,000 条全量训练、500 条测试生成、报告、压缩备份和关机。中间不需要人工接命令。

## 2. 更新代码

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git fetch origin
git switch codex/0821_schemeE_v3
git pull origin codex/0821_schemeE_v3
```

如果分支第一次拉取，`git switch` 会在本地建立跟踪分支。若 GitHub 网络超时，先重试 `git fetch origin`，不要删除现有仓库或数据。

## 3. 安装 Git LFS 并拉 AE

AE checkpoint 约 100 MB，普通 Git 中保存的是 LFS 指针：

```bash
apt-get update
apt-get install -y git-lfs
git lfs install
git lfs pull --include="schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"
```

验证它不是一百多字节的指针：

```bash
ls -lh schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
stat -c '%s bytes' schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

正常应约 100 MB。若显示约 130 字节，LFS 尚未下载成功。

## 4. 检查数据

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
nvidia-smi
ls -lh ../Round2_Map/Round2_Train_Channel.npy
ls -lh ../Round2_Map/Round2_Train_Pos.npy
ls -lh ../Round2_Map/Round2_Test_Pos.npy
ls -lh ../Round2_Map/Round2_Map.ply
```

训练信道应约 5.9 GiB。数据默认不上传 GitHub；若实例中没有 `Round2_Map`，应从 AutoDL 文件存储或原始上传恢复，而不是提交到普通 Git。

## 5. 建议先做一次预检

```bash
python scripts/prepare_v3_config.py
python scripts/check_environment.py --config configs/v3_5090.json --require-cuda --strict-boosters
python scripts/inspect_architecture.py --config configs/v3_5090.json
```

架构检查应包含：

```text
architecture: spectral_gaussian_dual_seed_transport_v3
total_latent_elements: 30720
full_latent_linear_layers: []
full_resolution_check: true
dual_seed_transport: true
```

## 6. 后台无人值守启动

```bash
mkdir -p logs
nohup env \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  BACKUP_ROOT=/root/autodl-fs/schemeE_v3_latest \
  bash scripts/run_v3_unattended.sh \
  > logs/v3_launcher.log 2>&1 &
echo $! > logs/v3_unattended.pid
```

解释：

- `nohup`：SSH/Jupyter 页面关闭后程序继续运行；
- `SHUTDOWN_ON_SUCCESS=1`：成功后自动关机；
- `SHUTDOWN_ON_FAILURE=1`：失败时先备份已有证据，再自动关机；
- `BACKUP_ROOT`：保存到 AutoDL 文件存储，不随实例释放而丢失；
- `echo $!`：记录后台进程号。

## 7. 查看实时日志

```bash
tail -f logs/v3_launcher.log
```

退出实时查看按：

```text
Ctrl+C
```

这只退出 `tail`，不会停止后台训练。

状态和最近日志：

```bash
cat logs/v3_unattended_status.txt
tail -n 80 logs/v3_launcher.log
ps -fp "$(cat logs/v3_unattended.pid)"
```

GPU 使用情况：

```bash
watch -n 2 nvidia-smi
```

退出 `watch` 同样按 `Ctrl+C`。

## 8. 如何理解关键日志

Fold0 每次验证会出现 `score=`。这是真实空间洞验证分数，不是训练准确率，也不是官方测试分数。

最终报告重点检查：

```bash
cat reports/generated/v3_attempt_selection.json
cat artifacts/v3/final/hybrid/summary.json
cat reports/generated/schemeE_v3_final_EXPERIMENT_REPORT.md
```

`transport_spectrum_gate_by_cell` 和 `transport_detail_gate_by_cell` 表示两个基站使用多邻居候选的平均比例。例如 `[0.70,0.20]` 表示 BS0 更信多邻居，BS1 更信单邻居。它不是越大越好，最终以分数为准。

## 9. 成功产物

主要提交文件：

```text
outputs/v3/Round2_Test_Channel.npy
```

检查：

```bash
ls -lh outputs/v3/Round2_Test_Channel.npy
python scripts/inspect_output.py outputs/v3/Round2_Test_Channel.npy \
  --samples 500 --report reports/generated/v3_final_output_check.json
cat reports/generated/v3_final_output_check.json
```

应满足：

```text
shape = [500,256,4,192]
dtype = complex64
finite = true
```

模型与报告：

```text
artifacts/v3/final/hybrid/best.pt
artifacts/v3/final/hybrid/summary.json
artifacts/v3/final/carrier_fit.json
reports/generated/v3_attempt_selection.json
reports/generated/schemeE_v3_final_EXPERIMENT_REPORT.md
schemeE_v3_results_*.tar.gz
```

持久化备份：

```text
/root/autodl-fs/schemeE_v3_latest/
```

## 10. 下载回本地

在 JupyterLab 左侧文件区进入：

```text
/root/autodl-fs/schemeE_v3_latest/
```

优先下载：

```text
Round2_Test_Channel.npy
schemeE_v3_results_*.tar.gz
schemeE_v3_results_*.tar.gz.sha256
v3_final_selected.json
reports/
```

也可以在本地 PowerShell 使用实例 SSH 地址：

```powershell
scp -P <SSH端口> root@<SSH主机>:/root/autodl-fs/schemeE_v3_latest/Round2_Test_Channel.npy .
```

## 11. 中断后恢复

重新开机并进入项目后，使用同一条后台命令。训练脚本会检查已有 checkpoint 和完成标记，并使用 `--resume` 恢复未完成的 Fold 尝试。

不要删除：

```text
artifacts/v2/fold0/spectral_teacher/
artifacts/v2/final/spectral_teacher/
artifacts/v3/
```

前两项是 E-v2/E-v3 可共享的严格频谱先验，删除后会重新计算。

## 12. 预计耗时

5090 常见墙钟时间约 `2.2--5.5 h`。已有 E-v2 频谱教师缓存时通常更快。三个 Fold 尝试会自动早停；`900 epoch` 是保护上限，不代表一定全部跑完。

无人值守脚本无论成功或失败都会先复制已有结果到 `/root/autodl-fs`，随后按环境变量自动关机，避免训练结束后继续计费。
