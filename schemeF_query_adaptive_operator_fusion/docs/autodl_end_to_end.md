# Scheme F AutoDL 5090 端到端操作教程

## 1. 这次最终要得到什么

一次无人值守任务会自动完成：

```text
D/E 诊断
-> Scheme F GPU 冒烟
-> Fold0 正式训练（必要时仅一次修复）
-> 自动选优
-> 4000 条 full-data final
-> 500 条测试推理
-> 权重、报告、NPY、SHA256 打包
-> 复制到 /root/autodl-fs
-> 自动关机
```

主提交文件：

```text
outputs/final/Round2_Test_Channel.npy
shape = [500,256,4,192]
dtype = complex64
约 750 MiB
```

## 2. Git 与 Git LFS 分别负责什么

普通 Git 适合 `.py/.json/.md/.sh`。模型权重和 NPY 很大，必须使用 Git LFS 或直接通过 AutoDL 持久盘下载。

仓库已经有 Scheme F 的 `.gitattributes`：

```gitattributes
*.pt filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
```

注意：`artifacts/`、`outputs/` 默认又被 `.gitignore` 忽略。大结果通常不要 push 到 GitHub，直接下载更快。如果必须 push 某个权重，需要同时执行：

```bash
git lfs install
git add -f schemeF_query_adaptive_operator_fusion/artifacts/fold0/context/best.pt
git commit -m "Add Scheme F Fold0 checkpoint via LFS"
git push origin 0820_schemeF
```

`-f` 只是在明确选择这个被忽略的大文件，不要对整个 `artifacts/` 使用。

## 3. AutoDL 实例和磁盘选择

建议：

- GPU：RTX 5090；
- PyTorch：2.2 或更高，CUDA 可用；
- 系统盘只放临时依赖；
- 仓库、权重、结果全部放 `/root/autodl-fs` 持久盘；
- 至少预留 12 GiB 可用磁盘，建议 25 GiB。

不要把唯一结果只放 `/root/autodl-tmp`。实例释放或环境变化时，临时盘可能不再存在。

## 4. 第一次进入终端

```bash
cd /root/autodl-fs
pwd
nvidia-smi
df -h /root/autodl-fs
```

必须看到 5090，且 `df` 有足够空间。

## 5. 安装 Git LFS

先检查：

```bash
git lfs version
```

若提示 `git: 'lfs' is not a git command`：

```bash
apt-get update
apt-get install -y git-lfs
git lfs install
git lfs version
```

Git LFS 下载 GitHub 大文件较慢时，先切换 HTTP/1.1 并允许续传：

```bash
git config --global http.version HTTP/1.1
git config --global lfs.concurrenttransfers 1
git config --global lfs.activitytimeout 180
```

超时后重新执行 `git lfs pull` 会继续或重试，不代表权重损坏。

## 6. 克隆 0820 分支

```bash
cd /root/autodl-fs
git clone --branch 0820_schemeF --single-branch \
  https://github.com/hututu1226/twotigers_digital_twins.git \
  twotigers_0820
cd twotigers_0820
git branch --show-current
git log -1 --oneline
```

应显示 `0820_schemeF`。

若 GitHub 443 超时：

```bash
git config --global http.version HTTP/1.1
git clone --depth 1 --branch 0820_schemeF \
  https://github.com/hututu1226/twotigers_digital_twins.git \
  twotigers_0820
```

仍失败时，不要反复计费等待。可在本机将源码压缩后通过 Jupyter 上传到 `/root/autodl-fs`，或使用已有持久盘 worktree 再执行 `git fetch`。

## 7. 拉取 Scheme C AE 权重

```bash
cd /root/autodl-fs/twotigers_0820
git lfs pull --include="schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"

ls -lh schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
stat -c '%s bytes' schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

如果只有约 `100--200 bytes`，它仍是 LFS pointer，不能训练。真实 checkpoint 应大于 `1,000,000 bytes`。

若持久盘旧任务已有真实 AE，可不走 GitHub，建立符号链接：

```bash
mkdir -p schemeC_full_resolution_context/artifacts/fold0/autoencoder
ln -s /root/autodl-fs/twotigers_0819_run/schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt \
  schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

目标已存在时不要重复 `ln -s`。

## 8. 准备 Round2 数据

Scheme F 配置要求仓库根目录存在：

```text
Round2_Map/Round2_Setup.json
Round2_Map/Round2_Map.ply
Round2_Map/Round2_Train_Pos.npy
Round2_Map/Round2_Train_Channel.npy
Round2_Map/Round2_Test_Pos.npy
```

检查：

```bash
cd /root/autodl-fs/twotigers_0820
ls -lh Round2_Map/Round2_Setup.json
ls -lh Round2_Map/Round2_Train_Channel.npy
```

若旧持久 worktree 已有数据，建议链接，不复制几 GB：

```bash
cd /root/autodl-fs/twotigers_0820
ln -s /root/autodl-fs/twotigers_0819_run/Round2_Map Round2_Map
```

先用 `ls -ld Round2_Map` 确认目标不存在；不要覆盖真实目录。

## 9. 安装 Python 依赖

```bash
cd /root/autodl-fs/twotigers_0820
python -m pip install -U pip
python -m pip install -r schemeF_query_adaptive_operator_fusion/requirements.txt
python -m pip install scipy scikit-learn xgboost lightgbm
```

E prior 已存在时通常不会重新训练 booster，但安装这些依赖可以保证缓存缺失时仍能自动重建。

检查：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 10. 先做只读预检

```bash
cd /root/autodl-fs/twotigers_0820/schemeF_query_adaptive_operator_fusion
python -m unittest discover -s tests -v
python scripts/inspect_architecture.py --config configs/fold0_5090.json
```

architecture 输出必须包含：

```text
context_architecture = query_adaptive_operator_fusion_v1
total_latent_elements = 30720
full_latent_linear_layers = []
full_resolution_check = true
router.top_k = 8
router.token_top_k = 2
```

`inspect_architecture` 不依赖正式 priors；正式环境检查会在无人值守脚本里完成。

## 11. 启动无人值守任务

```bash
cd /root/autodl-fs/twotigers_0820/schemeF_query_adaptive_operator_fusion
mkdir -p logs

nohup env \
  BACKUP_ROOT=/root/autodl-fs/schemeF_0820 \
  LEGACY_RUN_ROOT=/root/autodl-fs/twotigers_0819_run \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_unattended.sh \
  > logs/launcher.log 2>&1 &

echo $! | tee logs/launcher.pid
```

解释：

- `nohup`：关闭网页或 SSH 后进程继续；
- `BACKUP_ROOT`：关机前复制结果到持久盘；
- `LEGACY_RUN_ROOT`：复用 D/E/AE 旧资产；路径不存在时脚本会寻找其他默认目录；
- 两个 `SHUTDOWN...=1`：成功或失败完成备份后都关机；
- `$!`：刚启动后台进程的 PID。

## 12. 看实时日志与退出 tail

状态摘要：

```bash
cat logs/unattended_status.txt
```

启动器日志：

```bash
tail -n 50 logs/launcher.log
```

实时主日志：

```bash
tail -f logs/unattended_*.log
```

退出实时显示：按 `Ctrl+C`。这只退出 `tail`，不会停止训练。

确认训练仍在运行：

```bash
ps -fp "$(cat logs/launcher.pid)"
nvidia-smi
```

不要用 `Ctrl+C` 杀掉后台训练，也不要关闭正在运行的 AutoDL 实例。

## 13. 如何判断已经结束

成功前状态文件为：

```text
status=RUNNING
```

完成后持久盘查看：

```bash
cat /root/autodl-fs/schemeF_0820/unattended_status.txt
find /root/autodl-fs/schemeF_0820 -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
```

成功应有：

```text
status=SUCCESS
Round2_Test_Channel.npy
schemeF_results_*.tar.gz
schemeF_results_*.tar.gz.sha256
reports/
logs/
```

实例自动关机后 Jupyter 连接断开是预期行为，不等于结果丢失；结果已经放在 `/root/autodl-fs/schemeF_0820`。

## 14. 失败时如何处理

失败状态：

```text
status=FAILED
message=pipeline exited with code ...
```

重新开机后先看：

```bash
cd /root/autodl-fs/schemeF_0820
tail -n 100 logs/unattended_*.log
```

再回仓库检查：

```bash
cd /root/autodl-fs/twotigers_0820/schemeF_query_adaptive_operator_fusion
find artifacts -name last.pt -o -name summary.json -o -name evaluation.json
```

修复环境问题后，用同一条 `nohup` 命令重启。各阶段会复用已有文件，Context 有 `last.pt` 时会 resume，不会默认从零重训。

不要手工创建空的 `best.pt`、`final.pt` 或 NPY 来绕过检查。

## 15. 下载结果到本机

### 方法 A：Jupyter 文件浏览器

在 Jupyter 左侧进入：

```text
/root/autodl-fs/schemeF_0820
```

优先下载：

```text
Round2_Test_Channel.npy
schemeF_results_*.tar.gz
schemeF_results_*.tar.gz.sha256
```

### 方法 B：scp

在本机 PowerShell 执行，主机和端口替换成 AutoDL SSH 信息：

```powershell
scp -P 端口 root@主机:/root/autodl-fs/schemeF_0820/Round2_Test_Channel.npy `
  "D:\华为算法大赛复赛\autodl_results\20260820_schemeF\"

scp -P 端口 root@主机:/root/autodl-fs/schemeF_0820/schemeF_results_*.tar.gz* `
  "D:\华为算法大赛复赛\autodl_results\20260820_schemeF\"
```

Windows 路径有中文或空格时必须加双引号。

## 16. 本机校验下载结果

PowerShell：

```powershell
cd "D:\华为算法大赛复赛\autodl_results\20260820_schemeF"
Get-FileHash .\schemeF_results_*.tar.gz -Algorithm SHA256
```

NPY 检查：

```powershell
python -c "import numpy as np; a=np.load('Round2_Test_Channel.npy',mmap_mode='r'); print(a.shape,a.dtype,np.isfinite(a).all())"
```

正确输出应为：

```text
(500, 256, 4, 192) complex64 True
```

## 17. 结果如何解释

查看：

```text
reports/schemeF_fold0_EXPERIMENT_REPORT.md
reports/schemeF_final_EXPERIMENT_REPORT.md
reports/fold0_breakdown.json
reports/fold_attempt_selection.json
```

重点字段：

- `Score/PAS/PDP/NMSE`：Fold0 准确率；
- `bs0/bs1`：是否某一个基站单独失控；
- `power_p99_log10`：极端功率误差；
- `detail_token_effective_neighbors`：Top2 是否塌成 Top1；
- `recommended_for_submission`：低于 `0.63` 时为 false；
- final 报告没有 test accuracy，因为比赛测试标签不可见。

## 18. 计费边界

脚本最大允许一次主 Fold、一次修复 Fold和一次 final。配置墙钟保护约为 `3.25 + 3.0 + 3.0` 小时，再加诊断、预处理和打包，最坏仍应控制在约 12 小时预算内。

自动关机只在以下动作之后发生：

1. 写状态；
2. 复制权重、报告、日志和 NPY 到持久盘；
3. `sync`；
4. 调用 `/usr/bin/shutdown`。

时间是估计，AutoDL 具体计费、关机和持久盘规则可能变化，运行前应以控制台当前显示为准。
