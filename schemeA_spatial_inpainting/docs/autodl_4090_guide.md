# AutoDL RTX 4090 完整操作教程

本文从本地 Git 提交开始，覆盖代码克隆、比赛数据上传、CUDA 环境、fold0 选模、全量训练、测试生成、结果备份、下载和停止计费。

本文按当前实际仓库编写：

```text
GitHub: https://github.com/hututu1226/twotigers_digital_twins
当前开发分支示例: results
AutoDL 仓库: /root/autodl-tmp/twotigers_digital_twins
Scheme A: /root/autodl-tmp/twotigers_digital_twins/schemeA_spatial_inpainting
比赛数据: /root/autodl-tmp/twotigers_digital_twins/Round2_Map
持久文件存储: /root/autodl-fs
```

如果新代码最终提交到了其他分支，把所有 `results` 替换为 GitHub 网页中实际包含 `schemeA_spatial_inpainting` 的分支。

## 1. 先理解代码、数据和结果分别放在哪里

| 内容 | 管理方式 | 原因 |
|---|---|---|
| Python、固定 JSON、Markdown | GitHub | 小、适合版本管理 |
| `Round2_Map.zip` | AutoDL 文件存储 | 约 5.07 GiB，不进入 Git |
| 解压后的训练 NPY | AutoDL 本地 NVMe | 训练 I/O 更快 |
| checkpoint、history、输出 NPY | 本地 NVMe 训练，结束后复制文件存储 | 兼顾速度和可靠性 |
| 最终 `tar.gz` | 文件存储 + 下载回本地 | 防止实例释放或本地盘故障 |

AutoDL 官方建议高 I/O 数据训练前从文件存储复制到实例本地盘；文件存储挂载在 `/root/autodl-fs`。[文件存储说明](https://www.autodl.com/docs/fs/)

## 2. 本地 Windows：只提交新工程代码

打开 PowerShell：

```powershell
cd "D:\华为算法大赛复赛"

git branch --show-current
git status --short
git remote -v
```

当前机器实际分支是 `results`，远端名是 `origin`。确认新目录存在：

```powershell
Get-ChildItem schemeA_spatial_inpainting
```

只暂存新工程，不把其他未完成修改一起提交：

```powershell
git add schemeA_spatial_inpainting
git status --short
```

`git add` 后不应出现：

```text
Round2_Map/
schemeA_spatial_inpainting/artifacts/
schemeA_spatial_inpainting/outputs/
schemeA_spatial_inpainting/logs/
```

这些大文件已经被 `.gitignore` 排除。提交并推送：

```powershell
git commit -m "feat: add Scheme A spatial inpainting"
git push origin results
```

每行含义：

- `git add schemeA_spatial_inpainting`：只把新工程当前内容放入待提交区。
- `git status --short`：提交前核对范围。
- `git commit -m ...`：在本地创建一个可追踪版本。
- `git push origin results`：把本地 `results` 分支上传到 GitHub 的同名分支。

刷新 GitHub，切换到 `results` 分支，确认能看到：

```text
schemeA_spatial_inpainting/README.md
schemeA_spatial_inpainting/configs/
schemeA_spatial_inpainting/spatial_inpainting/
schemeA_spatial_inpainting/scripts/
schemeA_spatial_inpainting/docs/
```

## 3. 本地 Windows：准备数据压缩包

当前本地文件：

```text
D:\华为算法大赛复赛\Round2_Map.zip
大小: 5.072 GiB
SHA-256: 4DECAA8CCF6E2AB6D8015C46A89916223DF6919BE95101AFFC02123F140B9748
```

再次检查：

```powershell
Get-FileHash -Algorithm SHA256 "D:\华为算法大赛复赛\Round2_Map.zip"
```

压缩包内部应直接包含 5 个文件，而不是再套一层未知目录：

```text
Round2_Train_Channel.npy
Round2_Train_Pos.npy
Round2_Test_Pos.npy
Round2_Setup.json
Round2_Map.ply
```

## 4. AutoDL 网页：先上传数据，再租 GPU

推荐顺序：

1. 在 AutoDL 控制台选择准备租卡的地区。
2. 打开该地区的“文件存储”并初始化。
3. 将 `Round2_Map.zip` 上传到文件存储。
4. 上传完成后再租 RTX 4090，避免上传 5 GB 时支付 GPU 费用。

文件存储前 20 GB 的当前计费规则及超额价格以官方页面为准。[AutoDL 计费说明](https://www.autodl.com/docs/price/)

## 5. 创建 4090 实例

建议选择：

```text
GPU: RTX 4090 24GB，单卡
系统: Ubuntu 22.04
镜像: PyTorch >= 2.2，Python 3.10 或 3.11，CUDA 12.x
内存: 32GB 或以上
数据盘: 免费 50GB 已足够当前工程；不必为模型扩容
地区: 与文件存储一致
计费: 短期实验优先按量
```

实例一旦处于开机状态就按实例时长计费，不是等 Python 使用 GPU 才计费；按量实例关机结束 GPU 计费。[官方计费规则](https://www.autodl.com/docs/price/)

### 5.1 先设置定时关机

开机后立即在控制台设置定时关机。首次完整流程建议设在 8 小时后：

- 正常预估 3.5 到 7 小时；
- 即使命令异常退出，定时关机也能阻止无限空转计费；
- 若 8 小时不足，每 epoch 已保存 `last.pt`，下次可续训。

命令链成功后还会主动调用 `/usr/bin/shutdown`，定时关机是第二道保险。AutoDL 也明确推荐训练完成后用该命令关机。[省钱说明](https://www.autodl.com/docs/save_money/)

## 6. AutoDL Terminal：检查资源

打开 JupyterLab Terminal 或 SSH：

```bash
nvidia-smi
df -h /root/autodl-tmp
df -h /root/autodl-fs
free -h
ls -lh /root/autodl-fs/Round2_Map.zip
```

确认：

- `nvidia-smi` 能看到 RTX 4090；
- `/root/autodl-tmp` 至少还有 15 GB，建议 25 GB；
- `/root/autodl-fs/Round2_Map.zip` 存在；
- 内存约 32 GB 或更高。

## 7. 克隆 GitHub 代码

AutoDL 访问 GitHub 超时时，官方提供学术资源加速：

```bash
source /etc/network_turbo
```

[AutoDL 学术资源加速说明](https://www.autodl.com/docs/network_turbo/)

当前 `results` 分支包含历史 Git LFS 大结果，Scheme A 不需要下载它们。执行：

```bash
cd /root/autodl-tmp
export GIT_LFS_SKIP_SMUDGE=1
git clone --branch results --single-branch \
  https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins
```

检查：

```bash
git branch --show-current
git log -1 --oneline
ls schemeA_spatial_inpainting
```

应看到当前分支 `results` 和新工程目录。

不再需要网络加速时关闭代理，避免影响其他网络：

```bash
unset http_proxy
unset https_proxy
```

### 7.1 如果仍然 clone 超时

先重试：

```bash
source /etc/network_turbo
git ls-remote https://github.com/hututu1226/twotigers_digital_twins.git HEAD
```

如果能返回 commit，再 clone。如果 Git LFS 命令缺失导致 checkout 失败，可让 Git 只保留 LFS 指针：

```bash
rm -rf /root/autodl-tmp/twotigers_digital_twins
git -c filter.lfs.smudge= \
    -c filter.lfs.process= \
    -c filter.lfs.required=false \
    clone --branch results --single-branch \
    https://github.com/hututu1226/twotigers_digital_twins.git
```

旧 Scheme1/Scheme2 的 LFS NPY 会保持小型指针，这不影响 Scheme A。

## 8. 校验并解压比赛数据到本地 NVMe

校验文件存储中的压缩包：

```bash
sha256sum /root/autodl-fs/Round2_Map.zip
```

必须得到：

```text
4decaa8ccf6e2ab6d8015c46a89916223df6919be95101affc02123f140b9748
```

解压：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
mkdir -p Round2_Map
unzip -q /root/autodl-fs/Round2_Map.zip -d Round2_Map
ls -lh Round2_Map
```

不要让路径变成 `Round2_Map/Round2_Map/Round2_*.npy`。如果出现双层目录，移动里面 5 个文件到外层。

验证 shape：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

root = Path('Round2_Map')
train = np.load(root / 'Round2_Train_Channel.npy', mmap_mode='r')
train_pos = np.load(root / 'Round2_Train_Pos.npy', mmap_mode='r')
test_pos = np.load(root / 'Round2_Test_Pos.npy', mmap_mode='r')
print('channel:', train.shape, train.dtype)
print('train pos:', train_pos.shape, train_pos.dtype)
print('test pos:', test_pos.shape, test_pos.dtype)
PY
```

预期：

```text
channel: (4000, 256, 4, 192) complex64
train pos: (4000, 3)
test pos: (500, 3)
```

## 9. 配置 Python/CUDA 环境

优先使用 AutoDL PyTorch 镜像已经安装的 CUDA 版 torch，不要随意覆盖：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

最后一项应为 RTX 4090，`torch.cuda.is_available()` 应为 `True`。

安装本工程：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeA_spatial_inpainting
python -m pip install --upgrade pip
python -m pip install "numpy>=1.26,<3"
python -m pip install -e . --no-deps
```

`--no-deps` 的目的是保留镜像中已经匹配 CUDA 的 torch。基础测试：

```bash
python -m unittest discover -s tests -v
```

应为 7/7 通过。

## 10. 先做 CUDA 冒烟

```bash
python scripts/smoke_test.py \
  --config configs/smoke.json \
  --device cuda
```

最后必须出现：

```text
"status": "PASS"
```

同时检查 `device` 为 `cuda`，推理的 `cell_counts` 为 `[1,1]`。这证明：

- CUDA forward/backward 可用；
- AE 和空间 U-Net 都跑过一轮；
- 双基站都进入推理；
- checkpoint 和输出 NPY 可读。

如果冒烟失败，不要直接启动正式训练。按 traceback 修正环境或代码。

## 11. fold0 严格开发训练

### 11.1 前台运行方式

最简单：

```bash
bash scripts/run_fold0.sh
```

该脚本依次执行：

```text
正式 1 m 预处理（首次）
-> AE 训练
-> latent 编码
-> Spatial U-Net 训练
-> 固定空间验证
-> outage 阈值扫描
-> 开发模型测试生成
-> NPY 检查
```

SSH 断开会终止普通前台进程，因此长训练建议用下一节的后台方式。

### 11.2 后台运行方式

```bash
mkdir -p logs
nohup bash scripts/run_fold0.sh > logs/fold0.log 2>&1 &
echo $! | tee logs/fold0.pid
```

查看：

```bash
tail -f logs/fold0.log
```

按 `Ctrl+C` 只退出日志查看，不会停止后台训练。确认进程：

```bash
ps -fp "$(cat logs/fold0.pid)"
nvidia-smi
```

当 `ps` 不再显示进程时：

```bash
tail -n 80 logs/fold0.log
```

必须看到测试 NPY `valid: true`，不能只看到进程结束。

### 11.3 查看准确 ETA

训练至少 5 个 epoch 后：

```bash
python scripts/estimate_runtime.py \
  --config configs/fold0_4090.json \
  --recent 5
```

静态时间预估见 [4090 时长说明](runtime_4090_estimate.md)，当前完整 fold0 预计 1.5 到 3.5 小时。

## 12. 检查 fold0 结果

```bash
cat artifacts/fold0/autoencoder/summary.json
cat artifacts/fold0/spatial/summary.json
cat artifacts/fold0/spatial/outage_scan.json
python scripts/inspect_output.py outputs/fold0/Round2_Test_Channel.npy
```

查看最佳 epoch：

```bash
python - <<'PY'
import torch
for path in [
    'artifacts/fold0/autoencoder/best.pt',
    'artifacts/fold0/spatial/best.pt',
]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(path, 'epoch=', ckpt['epoch'] + 1, 'metrics=', ckpt['metrics'])
PY
```

此时应重点判断：

- AE ceiling 是否可接受；
- spatial best score 是否稳定高于早期 epoch；
- NMSE 是否在 PAS/PDP 上升时恶化；
- outage F1 和预测全零数量是否合理。

## 13. 生成全量配置并重训 4000 条

自动读取 fold0 最佳 epoch 与阈值：

```bash
python scripts/prepare_final_config.py
cat configs/final_selected.json
```

然后后台运行：

```bash
nohup env CONFIG=configs/final_selected.json \
  bash scripts/run_final.sh > logs/final.log 2>&1 &
echo $! | tee logs/final.pid
```

监控：

```bash
tail -f logs/final.log
ps -fp "$(cat logs/final.pid)"
nvidia-smi
```

全量配置没有独立验证集，固定轮数训练全部 4000 条，推理使用：

```text
artifacts/final/autoencoder/final.pt
artifacts/final/spatial/final.pt
```

训练完成后必须检查：

```bash
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
```

预期：

```text
shape: [500, 256, 4, 192]
dtype: complex64
finite: true
valid: true
size: 约 0.732 GiB
```

## 14. 训练中断后的恢复

### 14.1 fold0 AE 中断

```bash
python scripts/train_autoencoder.py --config configs/fold0_4090.json --resume
python scripts/encode_latents.py --config configs/fold0_4090.json
```

AE 完成后要重新编码 latent。

### 14.2 fold0 U-Net 中断

```bash
python scripts/train_spatial.py --config configs/fold0_4090.json --resume
```

### 14.3 final 中断

先查看最后完成到哪一阶段：

```bash
tail -n 100 logs/final.log
```

若 AE 中断：

```bash
python scripts/train_autoencoder.py --config configs/final_selected.json --resume
python scripts/encode_latents.py --config configs/final_selected.json
python scripts/train_spatial.py --config configs/final_selected.json
```

若 U-Net 中断：

```bash
python scripts/train_spatial.py --config configs/final_selected.json --resume
python scripts/infer.py --config configs/final_selected.json
```

不要在已有断点时误用不带 `--resume` 的同一训练命令；不带 `--resume` 会把该阶段旧 checkpoint 和 history 清理后重训。

## 15. 打包结果

```bash
bash scripts/package_results.sh
```

脚本先检查必需文件，成功后显示：

```text
Created schemeA_results_YYYYMMDD_HHMMSS.tar.gz
Checksum: ...
```

找出文件：

```bash
ARCHIVE=$(ls -t schemeA_results_*.tar.gz | head -1)
echo "$ARCHIVE"
ls -lh "$ARCHIVE" "$ARCHIVE.sha256"
sha256sum -c "$ARCHIVE.sha256"
```

必须显示 `OK`。

## 16. 复制到文件存储，确认后再关机

创建带时间的持久目录：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP=/root/autodl-fs/schemeA_$STAMP
mkdir -p "$BACKUP"

cp "$ARCHIVE" "$ARCHIVE.sha256" "$BACKUP/"
cp -a logs "$BACKUP/"
cp -a artifacts/fold0/spatial/outage_scan.json "$BACKUP/"
cp -a configs/final_selected.json "$BACKUP/"
sync

find "$BACKUP" -maxdepth 2 -type f -printf '%p %s bytes\n'
(cd "$BACKUP" && sha256sum -c "$ARCHIVE.sha256")
```

确认文件存储中压缩包存在且校验 `OK` 后关机：

```bash
/usr/bin/shutdown
```

关机后页面状态应变为“已关机”。只退出 SSH/JupyterLab 不会停止计费。

关机会保留普通容器实例数据，但本地盘没有冗余，且实例连续关机达到平台释放周期后会清空；重要结果必须已复制文件存储或下载本地。[实例数据保留规则](https://www.autodl.com/docs/instance_data/)

付费扩容数据盘即使关机也可能继续计费；如果曾扩容且不再需要，应在控制台缩容或释放，按当前官方规则核对。[本地数据盘计费](https://www.autodl.com/docs/local_disk/)

## 17. 可选：成功后自动备份并关机

只在 fold0 已审查、准备无人值守跑 final 时使用。先保持控制台 8 小时定时关机，再执行：

```bash
nohup bash -lc '
set -Eeuo pipefail
cd /root/autodl-tmp/twotigers_digital_twins/schemeA_spatial_inpainting
env CONFIG=configs/final_selected.json bash scripts/run_final.sh
bash scripts/package_results.sh
ARCHIVE=$(ls -t schemeA_results_*.tar.gz | head -1)
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP=/root/autodl-fs/schemeA_final_$STAMP
mkdir -p "$BACKUP"
cp "$ARCHIVE" "$ARCHIVE.sha256" "$BACKUP/"
cp -a logs configs/final_selected.json "$BACKUP/"
sync
cd "$BACKUP"
sha256sum -c "$ARCHIVE.sha256"
/usr/bin/shutdown
' > logs/final_backup_shutdown.log 2>&1 &
```

这里使用严格模式和顺序命令：训练、检查、打包、复制或校验任一步失败，都不会主动关机，便于保留现场；控制台定时关机仍负责最终费用兜底。

## 18. 下载回 Windows

最简单是在 AutoDL 文件存储网页下载压缩包和 `.sha256`。也可以用 FileZilla/SFTP，或在本地 PowerShell 执行 scp。

假设 AutoDL 页面给出的 SSH 命令为：

```text
ssh -p 35394 root@region-1.autodl.com
```

本地 PowerShell 下载：

```powershell
scp -P 35394 `
  root@region-1.autodl.com:/root/autodl-fs/schemeA_时间目录/schemeA_results_时间.tar.gz `
  "D:\华为算法大赛复赛\downloads\"

scp -P 35394 `
  root@region-1.autodl.com:/root/autodl-fs/schemeA_时间目录/schemeA_results_时间.tar.gz.sha256 `
  "D:\华为算法大赛复赛\downloads\"
```

端口和主机必须替换为自己实例页面显示的值。官方也支持 JupyterLab、FileZilla 和 scp 下载。[下载说明](https://www.autodl.com/docs/down/)

Windows 校验：

```powershell
Get-FileHash -Algorithm SHA256 "D:\华为算法大赛复赛\downloads\schemeA_results_时间.tar.gz"
Get-Content "D:\华为算法大赛复赛\downloads\schemeA_results_时间.tar.gz.sha256"
```

两边 hash 必须相同，再解压归档。

## 19. 下次代码更新

本地修改并推送后，AutoDL 仓库中执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
source /etc/network_turbo
git status --short
git pull --ff-only origin results
unset http_proxy
unset https_proxy
```

`artifacts/`、`outputs/`、`logs/` 被忽略，不会阻止 pull。若 Git 提示源码有本地修改，不要使用 `git reset --hard`；先用 `git diff` 查明是否有云端手工改动并保存。

## 20. 最终检查表

关机前逐项确认：

```text
[ ] 7 项单元测试通过
[ ] CUDA smoke 为 PASS，两个基站各走过一次
[ ] fold0 AE best.pt 存在
[ ] fold0 spatial best.pt 存在
[ ] outage_scan.json 存在
[ ] final_selected.json 使用 validation_fold=null
[ ] final AE final.pt 存在
[ ] final spatial final.pt 存在
[ ] 最终 NPY shape=(500,256,4,192)
[ ] 最终 NPY dtype=complex64 且 finite=true
[ ] tar.gz 和 sha256 已生成
[ ] 文件存储中的 tar.gz 校验 OK
[ ] 重要结果已下载或处于文件存储
[ ] AutoDL 实例已关机
```
