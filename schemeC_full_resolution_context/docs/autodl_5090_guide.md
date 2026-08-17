# Scheme C 在 AutoDL 5090 上的完整运行教程

## 1. 先说明要完成什么

本教程覆盖以下完整流程：

1. 从 GitHub 的 `0817_schemeC` 分支取得代码。
2. 把比赛数据放到正确目录。
3. 检查 5090、CUDA、PyTorch 和数据文件。
4. 先跑极小样本冒烟，确认整条链路可用。
5. 运行 Fold0 验证训练，得到可比较的验证分数。
6. 根据 Fold0 选择 epoch 和 outage 阈值。
7. 使用全部 4000 条训练样本训练最终模型。
8. 生成 500 条测试信道、保存权重、打包并下载。
9. 确认备份后在 AutoDL 控制台关机，避免继续计费。

Fold0 和最终全量训练是两次完整训练。为了严格控制每次租卡不超过约 8 小时，建议分成两个租卡时段，不要指望在同一个 8 小时窗口内完成调参、Fold0 和最终全量训练。

## 2. 目录关系必须先看懂

GitHub 仓库克隆后的推荐结构如下：

```text
/root/autodl-tmp/twotigers_digital_twins/
├── Round2_Map/
│   ├── Round2_Setup.json
│   ├── Round2_Map.ply
│   ├── Round2_Train_Pos.npy
│   ├── Round2_Train_Channel.npy
│   └── Round2_Test_Pos.npy
└── schemeC_full_resolution_context/
    ├── configs/
    ├── docs/
    ├── scheme_c/
    ├── scripts/
    └── tests/
```

配置中的数据路径是 `../Round2_Map`。它表示：从 `schemeC_full_resolution_context` 向上一级，再进入 `Round2_Map`。

不要把数据放进 `scheme_c/`。`scheme_c/` 是 Python 源代码包，不是数据目录。

## 3. 创建 AutoDL 实例

建议选择：

- GPU：RTX 5090 或 5090D，实际显存以 `nvidia-smi` 为准。
- 数据盘：至少预留约 50 GB，权重、latent、日志和打包文件都需要空间。
- 镜像：选择平台提供且能正确识别 5090 的较新 PyTorch/CUDA 镜像。

不要只根据镜像名字判断 CUDA 可用。启动后必须执行本教程第 6 节的实际检查。

项目、数据和结果应放在 `/root/autodl-tmp` 这类数据盘路径，不要长期只放系统盘。实例“关机”“释放”以及数据盘保留规则以你当前 AutoDL 页面为准；在确认平台规则前，最稳妥的做法仍是训练后立即打包并下载。

## 4. 从 GitHub 克隆指定分支

进入数据盘：

```bash
cd /root/autodl-tmp
```

克隆本项目，并直接选择 Scheme C 分支：

```bash
git clone --branch 0817_schemeC --single-branch \
  https://github.com/hututu1226/twotigers_digital_twins.git
```

各参数含义：

- `git clone`：把远程仓库下载为本地目录。
- `--branch 0817_schemeC`：检出 Scheme C 分支，不使用旧方案分支。
- `--single-branch`：只下载该分支的历史，减少传输量。
- 最后一段 URL：GitHub 仓库地址。

进入项目：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
```

确认分支和最新提交：

```bash
git branch --show-current
git log -1 --oneline
```

第一条应输出 `0817_schemeC`。

### 4.1 GitHub 443 超时怎么办

先重试一次，并强制使用 HTTP/1.1：

```bash
cd /root/autodl-tmp
git -c http.version=HTTP/1.1 clone \
  --depth 1 --branch 0817_schemeC --single-branch \
  https://github.com/hututu1226/twotigers_digital_twins.git
```

`--depth 1` 只下载最新一次提交，进一步减小传输量。

如果仍然超时，不要连续等待两小时。可在自己电脑上打开 GitHub 分支页面下载 ZIP，或运行 `scripts/package_source.sh` 生成源码包，再通过 AutoDL 网页文件管理器上传到 `/root/autodl-tmp` 并解压。

源码包解压示例：

```bash
cd /root/autodl-tmp
mkdir -p twotigers_digital_twins/schemeC_full_resolution_context
tar -xzf schemeC_source_YYYYMMDD_HHMMSS.tar.gz \
  -C twotigers_digital_twins/schemeC_full_resolution_context
```

## 5. 上传比赛数据

比赛数据通常没有提交到 Git，因为 `.npy` 信道文件很大。使用 AutoDL 文件管理器把 `Round2_Map.zip` 上传到仓库根目录：

```text
/root/autodl-tmp/twotigers_digital_twins/Round2_Map.zip
```

解压：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
unzip -q Round2_Map.zip
```

检查是否多套了一层目录：

```bash
find Round2_Map -maxdepth 2 -type f | sort | head -20
```

正确文件应直接位于 `Round2_Map/` 下。如果实际路径是 `Round2_Map/Round2_Map/Round2_Setup.json`，说明多套了一层。应移动内层目录内容，或者修改三个 JSON 配置中的 `data.root`；不要在路径错误时直接开始训练。

## 6. 检查 PyTorch、CUDA 和数据

进入 Scheme C：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
```

先看显卡：

```bash
nvidia-smi
```

再让 PyTorch 实际创建一个 CUDA 张量：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: do not start formal training")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
print("CUDA tensor mean:", x.mean().item())
PY
```

如果 `torch.cuda.is_available()` 是 `False`，不要继续正式训练。优先切换到 AutoDL 提供的兼容 5090 镜像；不要随意安装一个与驱动不匹配的旧 CUDA wheel。

安装项目时尽量保留镜像中已经能用的 PyTorch：

```bash
python -m pip install -U pip
python -m pip install "numpy>=1.26,<3"
python -m pip install -e . --no-deps
```

`-e .` 表示以可编辑方式安装当前项目；`--no-deps` 表示不让 pip 替换已经验证可用的 PyTorch。

运行自动检查：

```bash
python scripts/check_environment.py \
  --config configs/fold0_5090.json \
  --require-cuda
```

检查结果必须满足：

- `cuda_available` 为 `true`。
- `gpu` 显示租到的 5090/5090D。
- `missing_data_files` 是空列表。
- 剩余磁盘足够。

## 7. 先跑冒烟测试

冒烟只使用极少量样本和极小模型。它的目的不是看分数，而是验证代码能否完成正向、反向、保存权重、加载权重和生成 NPY。

执行：

```bash
mkdir -p logs
set -o pipefail
bash scripts/run_smoke.sh --device cuda 2>&1 | tee logs/smoke.log
```

命令解释：

- `mkdir -p logs`：创建日志目录，已经存在也不会报错。
- `set -o pipefail`：Python 失败时，即使后面接了 `tee`，整条命令仍返回失败。
- `2>&1`：把错误输出和普通输出合并。
- `tee logs/smoke.log`：终端实时显示，同时保存日志。

成功标准：

- 10 项单元测试全部 `OK`。
- 最后 JSON 中出现 `"status": "PASS"`。
- 输出 shape 是 `[2,256,4,192]`。
- dtype 是 `complex64`。
- `full_resolution_check` 是 `true`。

本地 CPU 已真实跑通该冒烟链路；云端仍必须再跑一次，因为 CUDA、驱动和 AMP 环境不同。

## 8. 正式运行 Fold0

### 8.1 建议使用 tmux

创建持久终端：

```bash
tmux new -s schemeC_fold0
```

在 tmux 内执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
mkdir -p logs
set -o pipefail
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
```

按 `Ctrl+B`，松开后按 `D`，可以退出 tmux 但不停止训练。

重新进入：

```bash
tmux attach -t schemeC_fold0
```

手机远程查看时，重新 SSH 登录后执行 attach 即可，不需要一直保持原 SSH 页面在线。

### 8.2 脚本会自动做什么

`run_fold0.sh` 按顺序执行：

1. 预处理双基站位置和点云 BEV。
2. 训练 AE v3。
3. 编码训练集，写入 30,720 维结构化 latent。
4. 训练全分辨率 Context。
5. Joint 微调 Context 和 AE decoder。
6. 分别评价 AE、Context、Joint。
7. 生成 `stage_gap.json`。
8. 扫描 outage 阈值。
9. 生成 500 条 Fold0 测试信道并检查格式。

### 8.3 为什么配置有 500 epoch 却不会盲目跑 500 次

`500` 是最大容量，不是要求必须跑满。Fold0 同时启用：

- 验证集早停。
- 验证平台自动降低学习率。
- AE 最多约 2.25 小时。
- Context 最多约 3.75 小时。
- Joint 最多约 0.5 小时。

三个训练阶段合计约 6.5 小时，剩余时间留给预处理、编码、验证、推理和打包。时间在 epoch 结束后检查，因此可能多出一个 epoch 的时间。

### 8.4 预计用时

以下是工程预算，不是对任何 5090 环境的保证：

| 阶段 | 5090/5090D 预计 |
| --- | ---: |
| 预处理 | 2 到 10 分钟 |
| AE v3 | 1.3 到 2.3 小时 |
| 全集 latent 编码 | 3 到 12 分钟 |
| Full-resolution Context | 2.5 到 3.8 小时 |
| Joint | 20 到 35 分钟 |
| 三阶段验证、阈值扫描、500 条推理 | 15 到 50 分钟 |
| 合计 | 约 5 到 8 小时 |

Scheme C 比旧 Context 慢的主要原因不是参数更多，而是 96 个 Spectrum 位置和 768 个 Detail 位置都要对同基站观测用户做学习型注意力。计算量随“latent 位置数 × 查询用户数 × 观测用户数”增长。

真实时长应在出现日志后估算：

```bash
python scripts/estimate_runtime.py \
  --config configs/fold0_5090.json \
  --recent 5
```

该脚本读取最近 5 个 epoch 的真实时间，不能在完全没有日志时凭空估计。

## 9. 训练时如何监控

另开一个 SSH 终端，查看 GPU：

```bash
watch -n 2 nvidia-smi
```

查看最新日志：

```bash
tail -n 80 -f logs/fold0.log
```

查看 AE 历史：

```bash
tail -n 5 artifacts/fold0/autoencoder/history.jsonl
```

查看 Context 历史：

```bash
tail -n 5 artifacts/fold0/context/history.jsonl
```

正常现象：

- 非验证 epoch 的 `score=nan`，因为该 epoch 没有跑验证，不代表训练损坏。
- `train` loss 有波动，因为每个 epoch 随机挖不同形状的洞。
- 学习率在验证平台期下降。

异常现象：

- 日志出现 `FloatingPointError`。
- CUDA OOM 后进程退出。
- GPU 利用率长期为 0，且日志不再更新。
- `nvidia-smi` 根本看不到 Python 进程。

## 10. 中断后续训

网络断开但 tmux 中 Python 仍在运行时，不要重复启动。先检查：

```bash
ps -ef | grep -E 'train_autoencoder|train_context|finetune_joint' | grep -v grep
```

如果进程确实已经停止，从最近 `last.pt` 续训：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
set -o pipefail
RESUME=1 bash scripts/run_fold0.sh 2>&1 | tee -a logs/fold0.log
```

`RESUME=1` 会恢复模型、优化器、学习率调度器、AMP scaler 和 epoch。即使目录中有 `final.pt`，脚本也会以 `last.pt` 为续训依据，因为上一次可能只是到达时间上限。

不要删除 `last.pt`，也不要只保留一个裸模型参数文件，否则无法完整恢复优化状态。

## 11. 显存不足如何调整

按以下顺序调整 `configs/fold0_5090.json`，每次只改一项：

1. 把 `context.attention_chunk_size` 从 `16` 改为 `8`。
2. 把 `context.inference_query_batch_size` 从 `24` 改为 `12` 或 `8`。
3. 确认 `context.gradient_checkpointing` 仍为 `true`。
4. 把 `context.maximum_targets` 从 `16` 改为 `12`。
5. 最后才考虑设置 `maximum_observations=1024`。

前四项不会删除 30,720 维 latent，只改变一次送进 GPU 的工作量。限制观测用户会改变模型看到的信息，必须记录为新的实验配置并重新比较分数。

## 12. Fold0 完成后看哪些结果

执行：

```bash
python - <<'PY'
import json
from pathlib import Path
for name in ("autoencoder", "context", "joint"):
    p = Path(f"artifacts/fold0/{name}/evaluation.json")
    d = json.loads(p.read_text())
    m = d["metrics"]
    print(name, {k: m[k] for k in ("pas", "pdp", "nmse", "score")})
print(Path("artifacts/fold0/stage_gap.json").read_text())
PY
```

判定顺序：

1. AE Score 是否接近或达到 `0.8`。
2. Context 相比 AE 丢了多少分。
3. Joint 是否稳定提升 Context。
4. PAS、PDP、NMSE 是否均衡，而不是只看总分。
5. `outage_scan.json` 选择的阈值是否产生异常多的 outage。

只有 Fold0 分数可以用于比较方案。全量最终训练没有验证集，不能从它的训练 loss 推断比赛 Score。

## 13. 先打包 Fold0，防止结果丢失

Fold0 跑完后立即执行：

```bash
bash scripts/package_fold0.sh
```

它会检查权重、评测 JSON、latent、500 条测试输出、日志和代码是否齐全，然后生成：

```text
schemeC_fold0_YYYYMMDD_HHMMSS.tar.gz
schemeC_fold0_YYYYMMDD_HHMMSS.tar.gz.sha256
```

查看大小和校验值：

```bash
ls -lh schemeC_fold0_*.tar.gz*
cat schemeC_fold0_*.sha256
```

通过 AutoDL 文件管理器下载这两个文件到本地。下载后在支持 `sha256sum` 的终端验证：

```bash
sha256sum -c schemeC_fold0_YYYYMMDD_HHMMSS.tar.gz.sha256
```

显示 `OK` 才说明下载文件没有损坏。

## 14. 由 Fold0 生成最终训练配置

确认 Fold0 结果值得继续后，执行：

```bash
python scripts/prepare_final_config.py
cat configs/final_selected.json
```

脚本会：

- 读取三个 Fold0 `best.pt` 的最佳 epoch。
- 把 epoch 数写入全量训练配置。
- 读取 Fold0 最佳 outage threshold。
- 生成 `configs/final_selected.json`。

全量训练使用全部 4000 条训练样本，不再留 Fold0。它按 Fold0 选出的 epoch 数训练，而不是在没有验证集的情况下假装早停；同时仍保留 2.25/3.75/0.5 小时的分阶段上限，避免单次租卡失控。

## 15. 运行全量最终训练

建议新开一个租卡时段或确认剩余预算充足：

```bash
tmux new -s schemeC_final
cd /root/autodl-tmp/twotigers_digital_twins/schemeC_full_resolution_context
mkdir -p logs
set -o pipefail
bash scripts/run_final.sh 2>&1 | tee logs/final.log
```

中断后续训：

```bash
RESUME=1 bash scripts/run_final.sh 2>&1 | tee -a logs/final.log
```

完成后检查 500 条输出：

```bash
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
```

必须确认：

- shape 为 `[500,256,4,192]`。
- dtype 为 `complex64`。
- 全部数值 finite。
- 文件路径是 `outputs/final/Round2_Test_Channel.npy`。

## 16. 最终权重和结果在哪里

AE：

```text
artifacts/final/autoencoder/final.pt
artifacts/final/autoencoder/last.pt
artifacts/final/autoencoder/history.jsonl
```

Context：

```text
artifacts/final/context/final.pt
artifacts/final/context/last.pt
artifacts/final/context/history.jsonl
```

Joint：

```text
artifacts/final/joint/final.pt
artifacts/final/joint/last.pt
artifacts/final/joint/history.jsonl
```

最终测试集：

```text
outputs/final/Round2_Test_Channel.npy
outputs/final/Round2_Test_Channel.json
```

结构化 latent：

```text
artifacts/final/encoded.npz
artifacts/final/encoded.json
```

## 17. 打包最终结果

执行：

```bash
bash scripts/package_results.sh
```

脚本会拒绝打包缺少关键文件的半成品。成功后生成：

```text
schemeC_results_YYYYMMDD_HHMMSS.tar.gz
schemeC_results_YYYYMMDD_HHMMSS.tar.gz.sha256
```

包内包括：

- 三阶段最终权重和训练历史。
- 预处理元数据。
- 30,720 维 encoded latent。
- 500 条最终测试信道。
- 配置、代码、测试和说明书。
- Python、PyTorch、CUDA、GPU 信息和 `pip freeze`。
- 如果存在，也会加入 Fold0 指标和日志。

把 `.tar.gz` 和 `.sha256` 都下载回本地并校验。

## 18. 什么时候可以关机

必须同时满足：

1. 训练命令已经返回 shell 提示符，确认没有 Python 训练进程。
2. `inspect_output.py` 通过。
3. 打包脚本成功完成。
4. 压缩包和 SHA256 文件已经下载到本地。
5. 本地 SHA256 校验通过。

检查进程：

```bash
ps -ef | grep -E 'train_|infer.py|run_fold0|run_final' | grep -v grep
```

AutoDL 官方支持在任务后执行 `/usr/bin/shutdown` 自动关机；对于按量实例，关机即停止 GPU 实例计费。仅关闭 SSH、手机投屏或浏览器页面不会关机。第二天仍建议到控制台确认实例状态确实为“已关机”。

如果需要在睡觉期间自动完成 Fold0、全量训练、推理、打包并关机，请直接使用 [夜间自动流水线教程](overnight_autorun.md)。该脚本会验证产物后调用 AutoDL 官方推荐的 `/usr/bin/shutdown`。

## 19. 后续从 GitHub 更新代码

如果云端目录没有自己修改过代码：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0817_schemeC
git pull origin 0817_schemeC
```

含义：

- `git switch`：切换到 Scheme C 分支。
- `git pull`：把 GitHub 上该分支的新提交拉到本机并合并。

数据、权重、latent 和输出受 `.gitignore` 保护，不应通过普通 Git 提交上传。它们应使用结果压缩包保存。

## 20. 常见问题

### `score=nan` 是否训练坏了

不一定。正式配置不是每个 epoch 都验证。未验证的 epoch 日志会显示 `score=nan`，等到 `validation_interval` 对应 epoch 才会出现真实分数。

### `final.pt` 是否一定是最佳模型

不是。Fold0 评价用 `best.pt`；`final.pt` 是本次命令结束时的模型。全量训练没有验证集，所以最终阶段使用按 Fold0 选择的训练轮数和 `final.pt`。

### 训练结束后为什么还要生成 encoded.npz

AE 权重只定义“怎么压缩”。`encoded.npz` 是 4000 条训练信道真正压缩后的 latent 和归一化统计，Context 训练和推理都需要它。

### 本地 CPU 能否生成最终测试集

理论上可以加载权重推理，但全分辨率注意力在 CPU 上会明显更慢。既然租用了 5090，建议在云端完成 500 条推理和格式检查，再把最终 NPY 下载回来。

### 训练超过预估怎么办

先用 `estimate_runtime.py` 看真实 epoch 时间。阶段小时上限会在 epoch 边界停止并保存。若总预算临近，优先确保当前 epoch 完成、运行推理和打包，不要直接强杀导致最近一个 epoch 状态丢失。
