# AutoDL RTX 4090 完整训练与结果回传教程

本文档面向第一次使用 Git、GitHub 和 AutoDL 的用户，覆盖以下完整流程：

```text
本地确认代码
  -> GitHub 私有仓库
  -> AutoDL 文件存储上传比赛数据
  -> 租用 RTX 4090 实例
  -> 克隆代码并准备 CUDA 环境
  -> 数据校验与预处理
  -> 两套方案 CUDA 冒烟测试
  -> 两套方案正式训练
  -> 验证集评估与 outage 阈值校准
  -> 生成两套测试集信道
  -> 打包训练日志、验证结果、测试结果和模型权重
  -> 下载到本地并校验
  -> 关闭实例停止 GPU 计费
```

本文所有命令均假定：

```text
GitHub 仓库：https://github.com/hututu1226/twotigers_digital_twins
AutoDL 项目目录：/root/autodl-tmp/twotigers_digital_twins
AutoDL 原始数据目录：/root/autodl-tmp/twotigers_digital_twins/Round2_Map
AutoDL 文件存储：/root/autodl-fs
```

命令中的仓库名、SSH 主机和端口若与你实际页面不同，以 AutoDL/GitHub 页面显示为准。

---

## 1. 先理解最终需要带回哪些文件

完成训练后，建议至少下载这些内容：

```text
方案一模型权重
artifacts/runs/scheme1_4070/best.pt

方案二模型权重
artifacts/runs/scheme2_4070/best.pt

方案一训练过程
artifacts/runs/scheme1_4070/history.jsonl
logs/scheme1_train.log

方案二训练过程
artifacts/runs/scheme2_4070/history.jsonl
logs/scheme2_train.log

训练时实际配置
artifacts/runs/scheme1_4070/resolved_config.json
artifacts/runs/scheme2_4070/resolved_config.json

验证集最终结果
reports/scheme1_validation.txt
reports/scheme2_validation.txt

outage 阈值扫描结果
reports/scheme1_outage_scan.txt
reports/scheme2_outage_scan.txt

测试集信道
outputs/Round2_Test_Channel_scheme1.npy
outputs/Round2_Test_Channel_scheme2.npy

最终推理配置
configs/scheme1_submit.json
configs/scheme2_submit.json

复现信息
reports/git_commit.txt
reports/environment.txt
reports/pip_freeze.txt
```

其中：

- `best.pt` 是空间验证综合分数最佳的模型，应优先使用；
- `last.pt` 是最近一次 epoch 的完整断点，用于中断恢复，不保证效果最佳；
- `final.pt` 是训练最后时刻的纯模型，不保证效果最佳；
- `history.jsonl` 每行记录一个 epoch 的训练损失和空间验证指标；
- 测试集没有真实标签，因此只能生成测试信道，不能在本地计算官方测试分数；
- 每个完整测试信道文件约 750 MiB；模型本身只有几十 MiB 左右。

---

## 2. 本地电脑：上传最新代码到 GitHub

打开 PowerShell：

```powershell
cd D:\华为算法大赛复赛
git status
git log -1 --oneline --decorate
git remote -v
```

需要确认：

1. 当前分支是 `main`；
2. 远端是 `origin`；
3. 远端地址是你的 GitHub 仓库；
4. 没有误提交 `Round2_Map`、`artifacts` 或 `.npy/.pt` 大文件。

将本教程提交到 GitHub：

```powershell
git add .gitignore docs\autodl_4090_full_guide.md README.md
git status --short
git commit -m "docs: add AutoDL 4090 training and download guide"
git push
```

如果 `git commit` 显示 `nothing to commit`，表示这些文件可能已经提交，可直接执行 `git push`。

刷新 GitHub 仓库网页，确认能看到：

```text
configs/
docs/
scripts/
src/
tests/
README.md
```

GitHub 上看不到 `Round2_Map`、`artifacts`、`outputs`、`downloads`、`logs`、`reports` 和临时的 `*_submit.json` 是正常的，因为它们被 `.gitignore` 排除了。这样即使误执行 `git add .`，训练权重、测试信道和下载包也不会进入 Git 仓库。

---

## 3. 本地电脑：准备并校验比赛数据压缩包

当前数据压缩包：

```text
D:\华为算法大赛复赛\Round2_Map.zip
大小约 5.07 GiB
```

本地计算 SHA-256：

```powershell
Get-FileHash -Algorithm SHA256 D:\华为算法大赛复赛\Round2_Map.zip
```

当前文件预期哈希：

```text
4DECAA8CCF6E2AB6D8015C46A89916223DF6919BE95101AFFC02123F140B9748
```

SHA-256 用于确认 5GB 文件上传后没有损坏。文件名和大小相同仍不能完全排除传输损坏，因此不要跳过哈希校验。

---

## 4. AutoDL：初始化文件存储并上传数据

推荐先上传数据，再租 GPU。这样上传 5GB 文件时不会支付 4090 费用。

操作顺序：

1. 登录 AutoDL；
2. 确定准备租卡的地区，例如某个北京区或内蒙古区；
3. 进入 `控制台 -> 文件存储`；
4. 初始化同一地区的文件存储；
5. 上传本地 `Round2_Map.zip`；
6. 等待上传任务显示完成。

AutoDL 文件存储挂载路径为：

```text
/root/autodl-fs
```

文件存储前 20GB 通常免费，且不需要启动 GPU 实例即可网页上传。正式训练时不要直接从网络文件存储读取 6.3GB 信道，应解压到本地 NVMe 数据盘 `/root/autodl-tmp`。

AutoDL 官方说明：

- [文件存储](https://www.autodl.com/docs/fs/)
- [上传数据](https://www.autodl.com/docs/scp/)

---

## 5. AutoDL：租用 RTX 4090 实例

推荐配置：

```text
计费方式：按量计费
GPU：RTX 4090，一张
显存：标准 24GB 足够
CPU 内存：建议 32GB 或以上
数据盘：默认免费 50GB 足够
系统：Ubuntu 22.04
镜像：PyTorch >= 2.4，Python 3.10/3.11，CUDA 12.1 或 12.4
地区：必须与文件存储所在地区一致
```

注意：

- 本项目当前是单 GPU 训练，租两张卡不会自动加速；
- 不需要租所谓的 4090 48GB 改卡，标准 24GB 足够；
- 配置文件名中包含 `4070` 只是最初硬件命名，不限制在 4070 上运行；
- 4090 会运行同一份 CUDA 配置，结果目录仍叫 `scheme1_4070/scheme2_4070`；
- 实例显示“运行中”后就开始计费，不是等 Python 使用 GPU 后才计费。

实例启动后可使用：

- AutoDL 页面中的 JupyterLab；
- AutoDL 页面提供的 SSH 命令；
- VS Code Remote-SSH。

第一次操作建议先使用 JupyterLab 的 Terminal。

AutoDL 官方说明：

- [快速开始](https://www.autodl.com/docs/quick_start/)
- [SSH 连接](https://www.autodl.com/docs/ssh/)
- [计费规则](https://www.autodl.com/docs/price/)

---

## 6. AutoDL：检查文件存储、磁盘和 GPU

打开 JupyterLab Terminal，执行：

```bash
nvidia-smi
df -h /root/autodl-tmp
df -h /root/autodl-fs
free -h
ls -lh /root/autodl-fs
```

需要看到：

- `nvidia-smi` 中存在 `NVIDIA GeForce RTX 4090`；
- `/root/autodl-tmp` 至少有 15GB 可用空间，建议 30GB 以上；
- `/root/autodl-fs` 能看到 `Round2_Map.zip`；
- 内存建议 32GB 左右。

如果 `/root/autodl-fs` 不存在：

1. 检查实例地区是否和文件存储地区一致；
2. 检查该地区是否已经初始化文件存储；
3. 关机后重新开机，使挂载生效。

---

## 7. AutoDL：克隆 GitHub 私有仓库

进入本地高速数据盘：

```bash
cd /root/autodl-tmp
git clone https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins
```

如果仓库是 Private，GitHub 会要求认证：

```text
Username：你的 GitHub 用户名
Password：GitHub Personal Access Token，不是 GitHub 登录密码
```

建议创建 Fine-grained Personal Access Token：

```text
GitHub -> Settings -> Developer settings
       -> Personal access tokens -> Fine-grained tokens
```

权限只需：

```text
Repository access：只选择 twotigers_digital_twins
Repository permissions -> Contents：Read-only
```

安全要求：

- 不要把 Token 写进 Python、Markdown 或 JSON；
- 不要把 Token 写在 clone URL 中；
- 在终端 Password 提示处粘贴 Token，终端不会显示字符，这是正常的；
- 训练结束后可在 GitHub 删除或撤销该 Token。

克隆后检查：

```bash
pwd
git status
git branch -a
git log -1 --oneline --decorate
git remote -v
```

预期：

```text
当前分支：main
远程分支：remotes/origin/main
工作区：干净
远端：GitHub twotigers_digital_twins
```

`main` 和 `origin/main` 是 Git 分支指针，不是文件夹，因此目录中只看到 `src/scripts/configs/docs` 是正常的。

---

## 8. AutoDL：校验并解压比赛数据

首先校验文件存储里的压缩包：

```bash
sha256sum /root/autodl-fs/Round2_Map.zip
```

输出必须与本地一致：

```text
4decaa8ccf6e2ab6d8015c46a89916223df6919be95101affc02123f140b9748
```

不一致时不要训练，应删除云端损坏文件并重新上传。

解压到项目的本地数据盘目录：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
mkdir -p Round2_Map
unzip -oq /root/autodl-fs/Round2_Map.zip -d Round2_Map
```

检查文件：

```bash
ls -lh Round2_Map
```

应包含：

```text
Round2_Setup.json
Round2_Map.ply
Round2_Train_Pos.npy
Round2_Train_Channel.npy
Round2_Test_Pos.npy
```

检查数组：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path("Round2_Map")
for name in [
    "Round2_Train_Pos.npy",
    "Round2_Train_Channel.npy",
    "Round2_Test_Pos.npy",
]:
    array = np.load(root / name, mmap_mode="r")
    print(name, array.shape, array.dtype, (root / name).stat().st_size)
PY
```

预期：

```text
Round2_Train_Pos.npy     (4000, 3)           float64
Round2_Train_Channel.npy (4000, 256, 4, 192) complex64
Round2_Test_Pos.npy      (500, 3)            float64
```

---

## 9. AutoDL：检查 Python 和 CUDA PyTorch

执行：

```bash
which python
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

必须满足：

```text
Python >= 3.10
PyTorch >= 2.4
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

如果 `CUDA available` 是 `False`，不要开始训练。优先重新创建正确的 PyTorch CUDA 镜像，不建议新手在错误镜像里手工拼 CUDA、cuDNN 和驱动。

安装项目自身代码，但不要覆盖镜像内的 CUDA PyTorch：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
python -m pip install -e . --no-deps
python -m pip install "numpy>=1.26,<3"
python -m pip check
```

再次检查：

```bash
python -c "import channel_ai,torch,numpy; print(channel_ai.__version__,torch.__version__,numpy.__version__,torch.cuda.is_available())"
```

最后一个值必须为 `True`。

---

## 10. AutoDL：运行预处理

执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
python scripts/preprocess.py
```

预期近似输出：

```text
Preprocessing complete: train=3369 validation=631 outage=262
```

生成：

```text
artifacts/preprocessed/manifest.json
artifacts/preprocessed/metadata.npz
artifacts/preprocessed/train_map_tokens.npy
artifacts/preprocessed/test_map_tokens.npy
```

检查：

```bash
ls -lh artifacts/preprocessed
```

预处理只需运行一次。后续重启实例只要数据仍在，就不必重复运行。

---

## 11. AutoDL：先跑单元测试

```bash
python -m unittest discover -s tests -v
```

必须全部显示 `ok`，最终应看到：

```text
Ran 3 tests
OK
```

这些测试检查：

- 复信道角度-时延正逆变换；
- PAS/PDP/NMSE 指标；
- 双小区标签推断。

任意测试失败时不要正式训练。

---

## 12. AutoDL：两套方案 CUDA 冒烟测试

执行：

```bash
python scripts/smoke_test.py --device cuda --samples 2
```

必须看到：

```text
PASS scheme1
PASS scheme2
PASS all smoke tests
```

冒烟测试会验证：

- GPU 能否训练；
- 方案一三个阶段能否跑通；
- 方案二 Token Transformer 能否反向传播；
- checkpoint 是否能保存和重新加载；
- 推理结果是否为 `(2,256,4,192) complex64`；
- 输出是否没有 NaN/Inf。

冒烟只训练极少样本和 1 个 epoch，分数低是正常的。冒烟 checkpoint 不能用于比赛提交。

同时执行：

```bash
nvidia-smi
```

确认 Python 进程确实占用了 4090 显存，而不是意外在 CPU 上运行。

---

## 13. AutoDL：准备后台训练和日志目录

SSH 或网页断开不应导致训练终止，因此使用 `tmux`。

检查 tmux：

```bash
tmux -V
```

如果不存在：

```bash
apt-get update && apt-get install -y tmux
```

创建目录：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
mkdir -p logs reports outputs
```

tmux 常用操作：

```text
创建会话：tmux new -s train_s1
退出但保持程序运行：先按 Ctrl+B，松开，再按 D
查看会话：tmux ls
重新进入：tmux attach -t train_s1
```

关闭浏览器或断开 SSH 不等于 AutoDL 关机，GPU 仍会继续计费。

---

## 14. AutoDL：正式训练方案一

创建方案一会话：

```bash
tmux new -s train_s1
```

进入 tmux 后执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
set -o pipefail
python -u scripts/train.py \
  --config configs/scheme1_4070.json \
  --device cuda \
  2>&1 | tee logs/scheme1_train.log
```

说明：

- `python -u`：让日志实时写出；
- `2>&1`：把错误日志和正常日志合并；
- `tee`：既在屏幕显示，也保存到文件；
- `set -o pipefail`：训练失败时整条管道能返回失败状态。

方案一依次执行：

```text
autoencoder：120 epochs，可能 early stopping
predictor：180 epochs，可能 early stopping
joint：80 epochs，可能 early stopping
```

看到训练开始后，按 `Ctrl+B`，再按 `D` 退出 tmux。

在普通终端监控：

```bash
tail -f logs/scheme1_train.log
```

退出 `tail`：按 `Ctrl+C`。这不会停止训练。

另一个终端查看 GPU：

```bash
watch -n 2 nvidia-smi
```

退出 `watch`：按 `Ctrl+C`。

训练完成的日志末尾应有：

```text
Training complete: artifacts/runs/scheme1_4070/final.pt
```

检查：

```bash
ls -lh artifacts/runs/scheme1_4070
tail -n 5 logs/scheme1_train.log
```

必须存在：

```text
best.pt
last.pt
final.pt
history.jsonl
resolved_config.json
```

### 14.1 方案一中断恢复

如果实例意外断开或训练进程结束：

```bash
python -u scripts/train.py \
  --config configs/scheme1_4070.json \
  --device cuda \
  --resume artifacts/runs/scheme1_4070/last.pt \
  2>&1 | tee -a logs/scheme1_train.log
```

`-a` 表示追加日志，不覆盖之前内容。

---

## 15. AutoDL：正式训练方案二

确认方案一已经完成，不要让两个模型同时争抢同一张 4090：

```bash
nvidia-smi
```

创建方案二会话：

```bash
tmux new -s train_s2
```

进入 tmux 后执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
set -o pipefail
python -u scripts/train.py \
  --config configs/scheme2_4070.json \
  --device cuda \
  2>&1 | tee logs/scheme2_train.log
```

方案二默认联合训练 350 epochs，并可能 early stopping。

退出 tmux：按 `Ctrl+B`，再按 `D`。

监控：

```bash
tail -f logs/scheme2_train.log
watch -n 2 nvidia-smi
```

训练完成的日志末尾应有：

```text
Training complete: artifacts/runs/scheme2_4070/final.pt
```

检查：

```bash
ls -lh artifacts/runs/scheme2_4070
```

### 15.1 方案二中断恢复

```bash
python -u scripts/train.py \
  --config configs/scheme2_4070.json \
  --device cuda \
  --resume artifacts/runs/scheme2_4070/last.pt \
  2>&1 | tee -a logs/scheme2_train.log
```

---

## 16. 查看训练集与空间验证历史

每个 `history.jsonl` 的一行代表一个 epoch，例如：

```json
{
  "phase": "joint",
  "epoch": 12,
  "seconds": 18.4,
  "train": {"total": 0.71},
  "validation": {
    "pas": 0.82,
    "pdp": 0.86,
    "nmse": 0.49,
    "score": 0.79,
    "gate_accuracy": 1.0,
    "outage_accuracy": 0.94
  }
}
```

输出每个方案验证分数最佳的 epoch：

```bash
python - <<'PY'
import json
from pathlib import Path

for scheme in ["scheme1", "scheme2"]:
    path = Path(f"artifacts/runs/{scheme}_4070/history.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    best = max(records, key=lambda row: row["validation"]["score"])
    print("\n", scheme)
    print("phase:", best["phase"])
    print("epoch:", best["epoch"] + 1)
    print("seconds:", best["seconds"])
    print("train:", best["train"])
    print("validation:", best["validation"])
PY
```

需要重点关注：

- `validation.score`：官方三指标加权综合值；
- `validation.pas`：越高越好；
- `validation.pdp`：越高越好；
- `validation.nmse`：越低越好；
- `gate_accuracy`：基站门控准确率；
- `outage_accuracy`：全零信道分类准确率；
- `train.total`：训练总损失，只用于观察收敛，不直接等于官方分数。

---

## 17. 独立计算并保存验证集结果

方案一：

```bash
python scripts/evaluate.py \
  --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --device cuda \
  --stage joint \
  2>&1 | tee reports/scheme1_validation.txt
```

方案二：

```bash
python scripts/evaluate.py \
  --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --device cuda \
  --stage joint \
  2>&1 | tee reports/scheme2_validation.txt
```

查看：

```bash
cat reports/scheme1_validation.txt
cat reports/scheme2_validation.txt
```

这些文件就是需要下载回本地的“验证集最终结果”。

---

## 18. 扫描并保存 outage 阈值

默认阈值 0.5 不一定最佳，应在空间验证集扫描。

方案一：

```bash
python scripts/calibrate_outage.py \
  --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --device cuda \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
  2>&1 | tee reports/scheme1_outage_scan.txt
```

方案二：

```bash
python scripts/calibrate_outage.py \
  --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --device cuda \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
  2>&1 | tee reports/scheme2_outage_scan.txt
```

每份文件末尾会显示：

```text
best_threshold=...
best_score=...
```

分别记下两个最佳阈值。不要假设两个方案使用同一阈值。

---

## 19. 创建最终推理配置

不要直接修改原训练配置，先复制：

```bash
cp configs/scheme1_4070.json configs/scheme1_submit.json
cp configs/scheme2_4070.json configs/scheme2_submit.json
```

推荐使用 JupyterLab 文件浏览器打开两个 `submit.json`，只修改：

```json
"outage_threshold": 0.5
```

将 0.5 替换为各自扫描得到的最佳阈值。不要修改模型参数：

```text
hidden_dim
latent_dim
base_channels
token_count
model_dim
transformer_layers
```

修改后检查 JSON 是否有效：

```bash
python -m json.tool configs/scheme1_submit.json > /dev/null
python -m json.tool configs/scheme2_submit.json > /dev/null
```

没有输出且返回终端，表示 JSON 格式正确。

---

## 20. 生成两套测试集信道

### 20.1 方案一测试结果

```bash
python scripts/infer.py \
  --config configs/scheme1_submit.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --output outputs/Round2_Test_Channel_scheme1.npy \
  --device cuda
```

### 20.2 方案二测试结果

```bash
python scripts/infer.py \
  --config configs/scheme2_submit.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --output outputs/Round2_Test_Channel_scheme2.npy \
  --device cuda
```

测试推理建议直接在 4090 上完成，额外时间很短，并能保持与训练完全相同的 PyTorch/CUDA 环境。

---

## 21. 严格检查测试结果

执行：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

for scheme in ["scheme1", "scheme2"]:
    path = Path(f"outputs/Round2_Test_Channel_{scheme}.npy")
    channel = np.load(path, mmap_mode="r")
    zero_count = int(np.all(channel == 0, axis=(1, 2, 3)).sum())
    mean_power = float(np.mean(np.abs(channel) ** 2))
    print("\n", path)
    print("file bytes:", path.stat().st_size)
    print("shape:", channel.shape)
    print("dtype:", channel.dtype)
    print("finite:", bool(np.isfinite(channel).all()))
    print("zero samples:", zero_count)
    print("mean power:", mean_power)
PY
```

两个文件都必须满足：

```text
shape: (500, 256, 4, 192)
dtype: complex64
finite: True
```

如果 shape、dtype 或 finite 任一项不正确，不要提交，也不要删除实例，应先排查。

比赛最终要求的文件名通常是：

```text
Round2_Test_Channel.npy
```

保留两套带方案名结果用于比较。决定提交某一方案后再复制：

```bash
cp outputs/Round2_Test_Channel_scheme1.npy outputs/Round2_Test_Channel.npy
```

如果最终选择方案二，将 `scheme1` 换成 `scheme2`。

---

## 22. 保存完整复现信息

```bash
mkdir -p reports
git rev-parse HEAD > reports/git_commit.txt
git status --short > reports/git_status.txt
python -m pip freeze > reports/pip_freeze.txt
nvidia-smi > reports/nvidia_smi.txt
python - <<'PY' > reports/environment.txt
import platform
import torch
print("platform:", platform.platform())
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

查看：

```bash
cat reports/git_commit.txt
cat reports/environment.txt
```

以后只有同时拥有 commit、配置和 checkpoint，才能可靠复现某次实验。

---

## 23. 打包需要下载的全部结果

在项目根目录执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins

tar -czf /root/autodl-fs/twotigers_round2_results.tar.gz \
  artifacts/runs/scheme1_4070/best.pt \
  artifacts/runs/scheme1_4070/last.pt \
  artifacts/runs/scheme1_4070/history.jsonl \
  artifacts/runs/scheme1_4070/resolved_config.json \
  artifacts/runs/scheme2_4070/best.pt \
  artifacts/runs/scheme2_4070/last.pt \
  artifacts/runs/scheme2_4070/history.jsonl \
  artifacts/runs/scheme2_4070/resolved_config.json \
  configs/scheme1_submit.json \
  configs/scheme2_submit.json \
  logs \
  reports \
  outputs/Round2_Test_Channel_scheme1.npy \
  outputs/Round2_Test_Channel_scheme2.npy
```

这里同时打包 `last.pt`，便于未来继续训练。如果只关心最终推理，可以不带 `last.pt`，但建议第一次完整保留。

检查压缩包：

```bash
ls -lh /root/autodl-fs/twotigers_round2_results.tar.gz
tar -tzf /root/autodl-fs/twotigers_round2_results.tar.gz | head -n 30
```

计算下载校验值：

```bash
sha256sum /root/autodl-fs/twotigers_round2_results.tar.gz \
  | tee /root/autodl-fs/twotigers_round2_results.tar.gz.sha256
```

确认 `.tar.gz` 和 `.sha256` 都位于 `/root/autodl-fs` 后，再进行下载或关机。

为什么放到文件存储：

- 实例关机后仍可保留；
- 本地数据盘没有冗余保障；
- 更换实例时也能访问；
- 可以先关掉昂贵 GPU，再慢慢下载。

---

## 24. 下载方法一：AutoDL 文件存储网页

最简单的方法：

1. 打开 AutoDL `控制台 -> 文件存储`；
2. 找到 `twotigers_round2_results.tar.gz`；
3. 下载到本地；
4. 同时下载 `.sha256` 文件。

如果浏览器下载大文件不稳定，使用下一节的 SCP。

---

## 25. 下载方法二：SCP 下载到本地

在 AutoDL 实例页面复制 SSH 命令，例如：

```text
ssh -p 35394 root@region-1.autodl.com
```

其中：

```text
端口：35394
主机：region-1.autodl.com
用户名：root
```

在本地 PowerShell 执行，不是在 AutoDL 终端执行：

```powershell
cd D:\华为算法大赛复赛
New-Item -ItemType Directory -Force downloads

scp -P 35394 root@region-1.autodl.com:/root/autodl-fs/twotigers_round2_results.tar.gz .\downloads\
scp -P 35394 root@region-1.autodl.com:/root/autodl-fs/twotigers_round2_results.tar.gz.sha256 .\downloads\
```

必须把示例端口和主机换成你自己实例页面显示的值。

注意：

- SCP 端口参数是大写 `-P`；
- 密码输入时不显示字符；
- 不要在远端路径前写 Windows 盘符；
- 如果实例已关机，SSH 无法连接，可使用文件存储网页下载，或低价无卡模式开机。

AutoDL 官方说明：[下载数据](https://www.autodl.com/docs/down/)

---

## 26. 本地校验并解压下载结果

本地 PowerShell：

```powershell
cd D:\华为算法大赛复赛
Get-FileHash -Algorithm SHA256 .\downloads\twotigers_round2_results.tar.gz
Get-Content .\downloads\twotigers_round2_results.tar.gz.sha256
```

两处 SHA-256 必须一致。

创建解压目录：

```powershell
New-Item -ItemType Directory -Force .\downloads\twotigers_round2_results
tar -xzf .\downloads\twotigers_round2_results.tar.gz -C .\downloads\twotigers_round2_results
```

检查文件：

```powershell
Get-ChildItem -Recurse .\downloads\twotigers_round2_results
```

本地验证两个测试文件：

```powershell
python -c "import numpy as np; a=np.load(r'downloads/twotigers_round2_results/outputs/Round2_Test_Channel_scheme1.npy',mmap_mode='r'); print(a.shape,a.dtype,np.isfinite(a).all())"

python -c "import numpy as np; a=np.load(r'downloads/twotigers_round2_results/outputs/Round2_Test_Channel_scheme2.npy',mmap_mode='r'); print(a.shape,a.dtype,np.isfinite(a).all())"
```

两条都应输出：

```text
(500, 256, 4, 192) complex64 True
```

---

## 27. 关机与停止计费

只有完成以下检查后再关机：

```text
[ ] 两套训练都显示 Training complete
[ ] 两套 best.pt 均存在
[ ] 两套 validation.txt 均存在
[ ] 两套 outage_scan.txt 均存在
[ ] 两套测试 NPY shape/dtype/finite 正确
[ ] 结果压缩包已写入 /root/autodl-fs
[ ] 压缩包 SHA-256 已保存
[ ] 最好已经下载并在本地验证
```

然后在 AutoDL 控制台点击“关机”。

重要区别：

- 关闭 JupyterLab 页面：不会停止计费；
- 断开 SSH：不会停止计费；
- tmux 退出：不会停止计费；
- AutoDL 控制台关机：按量 GPU 停止计费。

关机后数据通常保留，但连续关机达到平台释放期限后实例可能被释放；本地数据盘也无冗余保障。因此重要结果必须已下载或复制到文件存储。

AutoDL 官方说明：

- [省钱与自动关机](https://www.autodl.com/docs/save_money/)
- [实例数据保留](https://www.autodl.com/docs/instance_data/)

---

## 28. 常见问题与处理

### 28.1 `git clone` 提示 Repository not found

可能原因：

- 仓库是 Private；
- GitHub 用户名不正确；
- Token 没有该仓库读取权限；
- URL 拼写错误。

确认 URL：

```text
https://github.com/hututu1226/twotigers_digital_twins.git
```

### 28.2 GitHub Password 认证失败

GitHub 不再接受账号登录密码用于 Git HTTPS。Password 提示处必须输入 Personal Access Token。

### 28.3 `Round2_Map.zip` 哈希不一致

不要解压或训练。重新上传压缩包，并再次比较 SHA-256。

### 28.4 `/root/autodl-fs` 不存在

文件存储未初始化、地区不一致或实例在初始化前已开机。检查后重启实例。

### 28.5 `torch.cuda.is_available()` 为 False

镜像不是 CUDA PyTorch，或环境被错误的 CPU PyTorch 覆盖。不要继续训练；优先重建正确镜像。

### 28.6 `CUDA out of memory`

先确认没有另一个训练进程：

```bash
nvidia-smi
ps -ef | grep python
```

如果只有当前进程，在对应配置中把：

```json
"batch_size": 8
```

改为：

```json
"batch_size": 4
```

4090 24GB 正常情况下不应需要降低。

### 28.7 SSH 断开后训练消失

说明没有使用 tmux/screen，或者训练进程已经报错退出。查看：

```bash
tmux ls
tail -n 100 logs/scheme1_train.log
tail -n 100 logs/scheme2_train.log
```

### 28.8 不知道训练是否结束

```bash
grep "Training complete" logs/scheme1_train.log
grep "Training complete" logs/scheme2_train.log
```

两条都有输出才表示两套方案完成。

### 28.9 找不到 `best.pt`

可能是训练未进入最终阶段或中途失败。查看日志最后 100 行并从 `last.pt` 恢复。

### 28.10 `JSONDecodeError`

提交配置编辑时破坏了 JSON。检查：

```bash
python -m json.tool configs/scheme1_submit.json
```

常见原因是多写或少写逗号、引号。

### 28.11 测试结果中出现 NaN/Inf

不要提交。保留实例和日志，检查 checkpoint、推理配置是否匹配，并确认训练日志是否出现 NaN。

### 28.12 文件存储下载太慢

可以先关掉 4090，再用 AutoDL 文件存储网页下载；或者使用无卡模式开机后 SCP。不要为了等待下载一直开着 4090。

---

## 29. 最短命令清单

以下清单用于熟悉完整流程后快速复查，不代替前文的错误检查。

```bash
# 克隆
cd /root/autodl-tmp
git clone https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins

# 数据
sha256sum /root/autodl-fs/Round2_Map.zip
mkdir -p Round2_Map
unzip -oq /root/autodl-fs/Round2_Map.zip -d Round2_Map

# 环境
python -m pip install -e . --no-deps
python -m pip install "numpy>=1.26,<3"

# 验收
python scripts/preprocess.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cuda --samples 2

# 训练方案一
python -u scripts/train.py --config configs/scheme1_4070.json --device cuda \
  2>&1 | tee logs/scheme1_train.log

# 训练方案二
python -u scripts/train.py --config configs/scheme2_4070.json --device cuda \
  2>&1 | tee logs/scheme2_train.log

# 验证
python scripts/evaluate.py --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt --device cuda --stage joint \
  2>&1 | tee reports/scheme1_validation.txt

python scripts/evaluate.py --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt --device cuda --stage joint \
  2>&1 | tee reports/scheme2_validation.txt

# 阈值扫描
python scripts/calibrate_outage.py --config configs/scheme1_4070.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt --device cuda \
  2>&1 | tee reports/scheme1_outage_scan.txt

python scripts/calibrate_outage.py --config configs/scheme2_4070.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt --device cuda \
  2>&1 | tee reports/scheme2_outage_scan.txt

# 修改 submit 配置后推理
python scripts/infer.py --config configs/scheme1_submit.json \
  --checkpoint artifacts/runs/scheme1_4070/best.pt \
  --output outputs/Round2_Test_Channel_scheme1.npy --device cuda

python scripts/infer.py --config configs/scheme2_submit.json \
  --checkpoint artifacts/runs/scheme2_4070/best.pt \
  --output outputs/Round2_Test_Channel_scheme2.npy --device cuda
```

---

## 30. 最终验收清单

### 代码与环境

```text
[ ] GitHub main 是最新代码
[ ] AutoDL clone 后 git status 干净
[ ] 记录了 git commit hash
[ ] torch.cuda.is_available() 为 True
[ ] GPU 为 RTX 4090
```

### 数据

```text
[ ] Round2_Map.zip SHA-256 一致
[ ] 训练信道 shape 为 (4000,256,4,192)
[ ] 训练信道 dtype 为 complex64
[ ] 预处理得到约 3369/631/262
```

### 训练

```text
[ ] 单元测试 3/3 通过
[ ] 双方案 CUDA 冒烟通过
[ ] 方案一 Training complete
[ ] 方案二 Training complete
[ ] 两套 best.pt 存在
[ ] 两套 history.jsonl 和完整日志存在
```

### 验证和测试

```text
[ ] 两套 validation.txt 存在
[ ] 两套 outage 阈值已扫描
[ ] 两套 submit 配置记录了各自阈值
[ ] 两套测试结果 shape 为 (500,256,4,192)
[ ] 两套测试结果 dtype 为 complex64
[ ] 两套测试结果 finite 为 True
```

### 回传和计费

```text
[ ] 权重、日志、验证、配置、测试结果已打包
[ ] 云端结果包 SHA-256 已保存
[ ] 结果包已下载到本地
[ ] 本地 SHA-256 校验一致
[ ] 本地解压和 NPY 检查通过
[ ] AutoDL 实例已经关机
```
