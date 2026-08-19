# 0819：Scheme E 先跑、Scheme D 后跑的 AutoDL 全流程

这份教程只对应 Git 分支 `0819`。目标是你启动一次后离开电脑，机器自动完成：

```text
Scheme E 冒烟已在本地通过
-> Scheme E 正式 Fold0 / 全量 / Test / 报告 / 打包
-> 即使 Scheme E 失败，仍继续 Scheme D
-> Scheme D 正式 Fold0 / 全量 / Test / 报告 / 打包
-> 复制到 /root/autodl-fs
-> 自动关机
```

## 1. 重要结论

- 两套方案都会复用 `schemeC_full_resolution_context/.../autoencoder/best.pt`，必须用 Git LFS 拉到真实约 100 MB 权重。
- `Round2_Map` 原始数据不上传 GitHub，需要保留在 AutoDL 数据盘。
- 不要在 GPU 训练期间上传 Git LFS。训练结束自动关机后，用无卡模式上传，显卡不计费。
- 两个最终测试文件同名，主脚本备份时会分别改名为 `schemeE_Round2_Test_Channel.npy` 和 `schemeD_Round2_Test_Channel.npy`。
- `>0.65` 是设计目标。只有 Fold0 报告是真实可测泛化分数；没有测试标签，不能把输出格式通过叫作测试准确率。

## 2. GitHub 拉取失败时

正常方式：

```bash
cd /root/autodl-tmp
git clone https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins
git switch 0819
```

若 GitHub 443 超时：

1. 先重试，不要在失败后继续 `cd` 一个不存在的目录。
2. 若镜像提供 `/etc/network_turbo`，可临时执行 `source /etc/network_turbo` 后重试。
3. 仍失败时，在本地把仓库压缩后用 AutoDL 文件管理器上传到 `/root/autodl-tmp`，解压后执行 `git remote -v` 检查远端。

确认分支：

```bash
git branch --show-current
git log -1 --oneline
```

必须显示 `0819`。

## 3. 安装 Git LFS 并取 AE

```bash
apt-get update
apt-get install -y git-lfs
git lfs install
git lfs pull --include="schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"

stat -c '%s bytes' schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
head -c 48 schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt
```

错误状态是 100 多字节且能看到 `version https://git-lfs.github.com/spec/v1`；正确状态约 100 MB 且是二进制内容。

GitHub 普通 Git 会阻止大于 100 MiB 的文件，因此权重和约 0.73 GiB 的 NPY 必须使用 LFS；GitHub Free/Pro 的 LFS 单文件上限为 2 GiB，两个测试 NPY 分别低于该限制。参考 [GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) 和 [Git LFS 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

## 4. 确认比赛数据

```bash
cd /root/autodl-tmp/twotigers_digital_twins
ls -lh Round2_Map/Round2_Setup.json
ls -lh Round2_Map/Round2_Map.ply
ls -lh Round2_Map/Round2_Train_Pos.npy
ls -lh Round2_Map/Round2_Train_Channel.npy
ls -lh Round2_Map/Round2_Test_Pos.npy
```

缺文件时先停止，不要启动正式脚本。

## 5. 初始化 AutoDL 文件存储

在 AutoDL 控制台的“文件存储”中初始化当前地区，然后重启实例并检查：

```bash
ls -ld /root/autodl-fs
df -h /root/autodl-tmp /root/autodl-fs
```

官方说明 `/root/autodl-tmp` 是高 IO 数据盘，关机不丢失；`/root/autodl-fs` 是跨实例共享网络存储，适合重要压缩备份。参考 [AutoDL 实例目录](https://www.autodl.com/docs/env/) 和 [AutoDL 文件存储](https://www.autodl.com/docs/fs/)。

训练数据和工作目录留在 `/root/autodl-tmp`，最终压缩包复制到 `/root/autodl-fs`。

## 6. 安装 Python 依赖

```bash
cd /root/autodl-tmp/twotigers_digital_twins
python -m pip install -U pip
python -m pip install -r schemeE_spectral_gaussian_hybrid/requirements.txt
python -m pip install -r schemeD_transport_residual_context/requirements.txt
```

检查 GPU，不要盲目重装 PyTorch：

```bash
python - <<'PY'
import torch
print('torch =', torch.__version__)
print('torch CUDA =', torch.version.cuda)
print('CUDA available =', torch.cuda.is_available())
print('GPU =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
nvidia-smi
```

## 7. 建议先各做一次冒烟

Scheme E：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
mkdir -p logs
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log
```

Scheme D：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeD_transport_residual_context
mkdir -p logs
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log
```

两个都 PASS 后再启动整夜任务。冒烟分数没有意义，只看是否完整跑通。

## 8. 启动两个方案的总任务

```bash
cd /root/autodl-tmp/twotigers_digital_twins
mkdir -p logs
chmod +x scripts/run_0819_schemeE_then_D.sh
chmod +x schemeE_spectral_gaussian_hybrid/scripts/*.sh
chmod +x schemeD_transport_residual_context/scripts/*.sh

nohup env \
  REPO_ROOT=/root/autodl-tmp/twotigers_digital_twins \
  BACKUP_ROOT=/root/autodl-fs/0819_results \
  SHUTDOWN_WHEN_DONE=1 \
  bash scripts/run_0819_schemeE_then_D.sh \
  > logs/0819_launcher.log 2>&1 &
echo $! > logs/0819_master.pid
cat logs/0819_master.pid
```

主脚本故意不使用“Scheme E 失败就全停”的行为。它会记录 `schemeE_exit`，备份能找到的文件，然后继续 Scheme D。两个都结束后，无论成功与否都执行关机。

AutoDL 官方推荐使用完整路径 `/usr/bin/shutdown` 自动关机；本项目脚本已这样实现。参考 [AutoDL 省钱与自动关机](https://www.autodl.com/docs/save_money/)。

## 9. 实时监控与退出

```bash
tail -f logs/0819_launcher.log
```

按 `Ctrl+C` 只退出日志，不停止后台任务。

查看阶段：

```bash
cat logs/0819_master_status.txt
ps -fp "$(cat logs/0819_master.pid)"
nvidia-smi
```

Scheme E 和 D 自己也有 `logs/`、训练 `history.jsonl` 和生成报告。

## 10. 预计总时长

5090 单卡估计：

- Scheme E：4.5–9 小时；
- Scheme D：4.5–11 小时；
- 两套串行：约 9–20 小时。

这是按旧日志 10–15 秒/epoch、增加的 steps、Exact GP 和多次验证推算。早停较早时会明显缩短；持续提升或压缩 0.73 GiB NPY 时会接近上限。5090 不是 8 小时硬限制，因此总脚本优先完整性，不用经验 epoch 强行截断。

## 11. 关机后文件在哪里

工作目录：

```text
/root/autodl-tmp/twotigers_digital_twins/
```

额外备份：

```text
/root/autodl-fs/0819_results/schemeE/
/root/autodl-fs/0819_results/schemeD/
```

重点文件：

```text
schemeE/schemeE_Round2_Test_Channel.npy
schemeD/schemeD_Round2_Test_Channel.npy
schemeE/schemeE_results_*.tar.gz
schemeD/schemeD_results_*.tar.gz
```

重新开机后：

```bash
find /root/autodl-fs/0819_results -maxdepth 3 -type f -printf '%p %s bytes\n'
sha256sum -c /root/autodl-fs/0819_results/schemeE/*.sha256
sha256sum -c /root/autodl-fs/0819_results/schemeD/*.sha256
```

## 12. 中断恢复

若实例意外中断，重新执行第 8 节命令。Scheme E 会复用 GP 分块缓存和 Hybrid `last.pt`；Scheme D 会复用编码并从 Context `last.pt` 恢复。已完成阶段会跳过。

不要删除：

```text
schemeE_spectral_gaussian_hybrid/artifacts/*/progress/
schemeE_spectral_gaussian_hybrid/artifacts/*/hybrid/last.pt
schemeD_transport_residual_context/artifacts/*/context/last.pt
```

## 13. 用无卡模式上传 GitHub

训练结束后从 AutoDL 控制台以无卡模式开机。官方说明无卡模式不会影响实例原有数据，适合上传下载和调试，参考 [AutoDL 无卡模式](https://www.autodl.com/docs/save_money/)。

首次在这台 AutoDL 实例提交时，先设置提交者身份：

```bash
git config --global user.name "你的 GitHub 用户名"
git config --global user.email "你的 GitHub 邮箱"
```

HTTPS 推送需要 GitHub Personal Access Token（PAT），不能填写 GitHub 登录密码。发布脚本执行到 `git push` 时，如终端询问：用户名填 GitHub 用户名，密码位置粘贴 PAT。不要把 PAT 写进仓库、脚本、远端 URL 或教程截图；已有凭据时不会再次询问。PAT 创建说明见 [GitHub 官方文档](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)。

先上传 Scheme E：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0819
apt-get update && apt-get install -y git-lfs
bash schemeE_spectral_gaussian_hybrid/scripts/publish_lfs_results.sh
```

再上传 Scheme D：

```bash
bash schemeD_transport_residual_context/scripts/publish_lfs_results.sh
git lfs ls-files
git status --short
git log -3 --oneline
```

两个发布脚本各生成一个明确提交并 push 到 `origin/0819`。上传大 NPY 可能很慢，但不需要 GPU。若 LFS 配额不足，保留 `/root/autodl-fs` 压缩包，改用网盘或 GitHub Release，不能把 0.73 GiB 文件作为普通 Git blob 强推。
