# AutoDL 单卡 4090/4090D 全流程教程

版本：2026-08-16  
目标：从 GitHub 获取 Scheme B，在 AutoDL 完成 Fold0、全量训练、500 条测试生成、权重与日志打包、下载和关机。

平台规则可能变化，计费与存储部分以文末链接的 AutoDL 官方文档和控制台当前显示为准。

## 1. 本地先把代码推到 GitHub

本地 PowerShell 进入仓库根目录：

```powershell
cd D:\华为算法大赛复赛
git status
git branch --show-current
```

当前开发分支是 `0814_spatial_inpainting`。只暂存新方案目录，避免把其他未确认修改一起提交：

```powershell
git add schemeB_structured_context_field
git status
git commit -m "Add structured context field scheme B"
git push -u origin 0814_spatial_inpainting
```

每条命令含义：

| 命令 | 含义 |
| --- | --- |
| `git status` | 查看哪些文件已改、已暂存、未跟踪 |
| `git branch --show-current` | 查看当前分支名 |
| `git add schemeB_structured_context_field` | 只把 Scheme B 放入下一次提交 |
| 第二次 `git status` | 提交前复核范围 |
| `git commit -m ...` | 在本地记录一个版本快照 |
| `git push -u origin ...` | 首次把该分支推送到 GitHub，并建立跟踪关系 |

`origin` 是远程仓库别名，`main` 或 `0814_spatial_inpainting` 是分支，不是 `src` 里的文件夹。

数据、模型和输出已被 `.gitignore` 排除。执行 `git status` 时不应看到：

```text
Round2_Map/
Round2_Map.zip
schemeB_structured_context_field/artifacts/
schemeB_structured_context_field/outputs/
```

若最终要从 `main` clone，可在 GitHub 合并分支后执行后续 `main` 版本命令；不合并也可以直接 clone 当前分支。

## 2. 租用实例

建议选择：

- 单张 RTX 4090 或 4090D，24 GB；
- 按量计费，便于训练后立即关机；
- PyTorch 2.2+、CUDA 12.x 的官方/基础镜像；
- 至少 30 GB 空闲数据盘，建议 50 GB 或更多；
- 数据放 `/root/autodl-tmp`，不要把 5 GB 数据和 0.73 GB 输出堆在小系统盘。

AutoDL 官方说明 `/root/autodl-tmp` 是高性能数据盘，关机后数据保留，但本地盘没有冗余，重要结果仍需备份到本地或文件存储。[AutoDL 实例目录说明](https://www.autodl.com/docs/env/)

实例开机后复制控制台显示的 SSH 地址和端口，例如：

```text
ssh -p 35394 root@region-1.autodl.com
```

文中的 `35394` 和 `region-1.autodl.com` 都是示例，必须换成你自己的。

## 3. Clone GitHub 仓库

登录 AutoDL：

```bash
cd /root/autodl-tmp
git clone --branch 0814_spatial_inpainting --single-branch https://github.com/hututu1226/twotigers_digital_twins.git
cd twotigers_digital_twins/schemeB_structured_context_field
```

若代码已合并到 main：

```bash
git clone --branch main --single-branch https://github.com/hututu1226/twotigers_digital_twins.git
```

### 3.1 再次出现 GitHub 443 timeout

AutoDL 官方 Git 文档建议 clone 失败或很慢时使用学术资源加速。执行：

```bash
source /etc/network_turbo
git clone --branch 0814_spatial_inpainting --single-branch https://github.com/hututu1226/twotigers_digital_twins.git
unset http_proxy
unset https_proxy
```

加速只在 clone/pull 期间开启，用完关闭，避免影响其他网络请求。官方也声明该代理不保证一直稳定。[AutoDL Git 说明](https://www.autodl.com/docs/git/)，[学术资源加速](https://www.autodl.com/docs/network_turbo/)

若仍失败，最稳妥的备选方案是：本地把仓库源码压成小包，经 JupyterLab、FileZilla 或 `scp` 上传，不要把 5 GB 数据和源码打在同一包中。

## 4. 上传 Round2 数据

推荐把现有 `Round2_Map.zip` 单独上传到：

```text
/root/autodl-tmp/Round2_Map.zip
```

AutoDL 官方目前推荐公网网盘/AutoPanel；也支持 JupyterLab、FileZilla 和 `scp`。[上传数据官方说明](https://www.autodl.com/docs/scp/)

### 4.1 用 SCP 上传

下面命令在你的 Windows 本地 PowerShell 执行，不是在 AutoDL 里执行：

```powershell
scp -P 35394 D:\华为算法大赛复赛\Round2_Map.zip root@region-1.autodl.com:/root/autodl-tmp/
```

替换端口和主机地址。若上传文件夹才使用 `-r`；上传单个 zip 不需要 `-r`。

### 4.2 解压到仓库根目录

回到 AutoDL 终端：

```bash
cd /root/autodl-tmp
unzip -q Round2_Map.zip -d twotigers_digital_twins
```

检查：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field
ls -lh ../Round2_Map
python -c "import numpy as np; print(np.load('../Round2_Map/Round2_Train_Pos.npy').shape); print(np.load('../Round2_Map/Round2_Train_Channel.npy', mmap_mode='r').shape); print(np.load('../Round2_Map/Round2_Test_Pos.npy').shape)"
```

必须输出：

```text
(4000, 3)
(4000, 256, 4, 192)
(500, 3)
```

如果 zip 解压后多套了一层，例如 `Round2_Map/Round2_Map/...`，移动内层文件，直到配置能通过 `../Round2_Map/Round2_Setup.json` 找到数据。

## 5. 检查磁盘与 GPU

```bash
df -h /root/autodl-tmp
nvidia-smi
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

要求：

- 数据盘最好还剩 25 GB 以上；
- `torch.cuda.is_available()` 为 `True`；
- GPU 名称正确；
- `nvidia-smi` 没有其他未知训练进程占满显存。

## 6. 安装项目

AutoDL 镜像已有 PyTorch 时，避免让 pip 下载并替换它：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field
python -m pip install -e . --no-deps
python -c "import numpy, torch, structured_context_field; print(numpy.__version__); print(torch.__version__); print(structured_context_field.__version__)"
```

若缺 NumPy：

```bash
python -m pip install "numpy>=1.26,<3"
```

不要在 CUDA 正常时随意执行 `pip install torch`，否则可能装到 CPU 版或不匹配的 CUDA 版。

## 7. 云端先做测试和 GPU 冒烟

```bash
python -m compileall -q structured_context_field scripts tests
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cuda
```

最后必须出现：

```text
"status": "PASS"
```

检查 GPU 冒烟输出：

```bash
python scripts/inspect_output.py outputs/smoke/Round2_Test_Channel.npy --expected-count 2
```

若这一步不通过，不要开始几个小时的正式训练。

## 8. 使用 screen 防止断线中止训练

SSH 断线可能终止前台进程。AutoDL 官方建议长任务使用守护会话，并将日志写文件。[守护进程官方说明](https://www.autodl.com/docs/daemon/)

安装并创建 screen：

```bash
apt-get update
apt-get install -y screen
screen -S schemeb
```

进入 screen 后：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field
mkdir -p logs
set -o pipefail
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
```

离开但不终止训练：按 `Ctrl+A`，松开后按 `D`。

重新进入：

```bash
screen -ls
screen -r schemeb
```

不进入 screen 也可查看日志：

```bash
tail -f /root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field/logs/fold0.log
```

## 9. Fold0 训练期间监控

另开一个 SSH/Jupyter 终端：

```bash
watch -n 2 nvidia-smi
```

查看当前阶段：

```bash
tail -n 30 logs/fold0.log
```

查看真实剩余时间：

```bash
python scripts/estimate_runtime.py --config configs/fold0_4090.json --recent 5
```

若显存不足，先停止当前进程，按 `docs/runbook.md` 的 OOM 顺序降低 batch/decode batch，再使用：

```bash
RESUME=1 bash scripts/run_fold0.sh 2>&1 | tee -a logs/fold0.log
```

## 10. Fold0 完成后的检查

```bash
cat artifacts/fold0/autoencoder/evaluation.json
cat artifacts/fold0/context/evaluation.json
cat artifacts/fold0/joint/evaluation.json
cat artifacts/fold0/stage_gap.json
cat artifacts/fold0/joint/outage_scan.json
python scripts/inspect_output.py outputs/fold0/Round2_Test_Channel.npy
```

先看 AE ceiling，再看 Context 和 Joint。若 AE ceiling 仍很低，暂停全量训练，优先改 AE；全量重训不会自动修复表示层上限。

## 11. 生成最终配置

```bash
python scripts/prepare_final_config.py
cat configs/final_selected.json
```

检查输出中的：

- `selected_epochs.autoencoder`；
- `selected_epochs.context`；
- `selected_epochs.joint`；
- `outage_threshold`；
- `split.validation_fold = null`。

## 12. 全量训练并生成 500 条测试结果

在 screen 中执行：

```bash
set -o pipefail
bash scripts/run_final.sh 2>&1 | tee logs/final.log
```

断点恢复：

```bash
RESUME=1 bash scripts/run_final.sh 2>&1 | tee -a logs/final.log
```

完成后检查：

```bash
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
ls -lh outputs/final/Round2_Test_Channel.npy
```

必须看到：

```text
shape = [500, 256, 4, 192]
dtype = complex64
finite = true
valid = true
```

## 13. 打包和 SHA256 校验

```bash
bash scripts/package_results.sh
ls -lh schemeB_results_*.tar.gz*
sha256sum -c schemeB_results_*.tar.gz.sha256
```

只有 `sha256sum ...: OK` 才说明压缩包完整。

找最新压缩包：

```bash
LATEST="$(ls -t schemeB_results_*.tar.gz | head -n 1)"
echo "$LATEST"
```

## 14. 备份到 AutoDL 文件存储

若实例已挂载 `/root/autodl-fs`：

```bash
LATEST="$(ls -t schemeB_results_*.tar.gz | head -n 1)"
cp "$LATEST" "$LATEST.sha256" /root/autodl-fs/
sync
ls -lh /root/autodl-fs/"$LATEST"*
```

文件存储是网络盘，可跨同地区实例使用；官方说明前 20 GB 免费，超过部分按当前规则计费。小文件应先打包再复制。[AutoDL 文件存储说明](https://www.autodl.com/docs/fs/)

本地数据盘没有冗余，所以仍建议至少下载一份到自己的电脑。

## 15. 下载回 Windows 本地

在 Windows 本地 PowerShell 执行，不是在 AutoDL 终端：

```powershell
cd C:\Users\13171\Downloads
scp -P 35394 root@region-1.autodl.com:/root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field/schemeB_results_20260816_123456.tar.gz .
scp -P 35394 root@region-1.autodl.com:/root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field/schemeB_results_20260816_123456.tar.gz.sha256 .
```

替换端口、主机和实际时间戳。AutoDL 官方下载文档也支持 JupyterLab、FileZilla、公网网盘和 `scp`。[下载数据官方说明](https://www.autodl.com/docs/down/)

Windows 校验：

```powershell
Get-FileHash .\schemeB_results_20260816_123456.tar.gz -Algorithm SHA256
Get-Content .\schemeB_results_20260816_123456.tar.gz.sha256
```

两个哈希字符串必须完全相同。

也可以只下载正式输出：

```powershell
scp -P 35394 root@region-1.autodl.com:/root/autodl-tmp/twotigers_digital_twins/schemeB_structured_context_field/outputs/final/Round2_Test_Channel.npy .
```

但只下载 NPY 不足以复现实验，至少还应保留最终三个 checkpoint、配置和日志；最推荐下载完整打包文件。

## 16. 如何避免训练结束后的额外 GPU 计费

AutoDL 按量实例目前以“实例开机时间”计费，不以 GPU 是否忙为准；控制台关机后实例 GPU 计费停止。关机后 GPU 不再预留。[AutoDL 官方计费说明](https://www.autodl.com/docs/price/)

### 16.1 手动方式，最稳妥

依次确认：

1. `run_final.sh` 成功结束；
2. 正式 NPY 格式检查通过；
3. `package_results.sh` 成功；
4. SHA256 校验通过；
5. 压缩包已复制到文件存储或下载到本地；
6. 在 AutoDL 控制台点击关机。

### 16.2 自动关机方式

AutoDL 官方支持任务后调用 `/usr/bin/shutdown`。使用 `&&` 时，只有前一步成功才关机；失败时实例保持开机，便于查看错误，但也会继续计费。[AutoDL 自动关机说明](https://www.autodl.com/docs/save_money/)

在已经完成 Fold0、生成 `final_selected.json` 后，可执行：

```bash
mkdir -p logs
bash -lc 'set -euo pipefail; bash scripts/run_final.sh > logs/final.log 2>&1; bash scripts/package_results.sh >> logs/final.log 2>&1; LATEST="$(ls -t schemeB_results_*.tar.gz | head -n 1)"; sha256sum -c "$LATEST.sha256"; sync' && /usr/bin/shutdown
```

如果已经挂载文件存储，可在 `sync` 前增加：

```bash
cp "$LATEST" "$LATEST.sha256" /root/autodl-fs/
```

不要使用 `; /usr/bin/shutdown`，因为即使训练失败也会关机，可能来不及看错误和保存中间状态。

自动关机后标准输出不再可见，所以日志必须写入 `logs/final.log`。

## 17. 关机后数据会不会丢

官方当前规则：正常关机不会清空系统盘或 `/root/autodl-tmp`；再次开机仍可看到数据。但连续关机 15 天实例会被释放，释放后数据不可恢复；本地磁盘也没有冗余保障。[AutoDL 实例数据保留说明](https://www.autodl.com/docs/instance_data/)

因此：

- 短期关机后重新开机，不需要重训；
- 不要把“关机保留”理解为永久、安全备份；
- 权重和结果至少保存在本地或文件存储一份；
- 如果扩容了付费数据盘，官方目前规定即使关机仍可能按日计数据盘费用，直到缩容或释放，具体看控制台费用明细。[AutoDL 数据盘计费说明](https://www.autodl.com/docs/local_disk/)

## 18. 无卡模式的用途

若只需检查日志、压缩或下载文件，可以先关机，再使用无卡模式开机。AutoDL 官方当前标价为 `0.1 元/小时`，配置为低 CPU/内存且无 GPU；无卡模式会释放原 GPU，之后正常开机可能遇到该 GPU 暂无空闲。[AutoDL 省钱说明](https://www.autodl.com/docs/save_money/)

大约 1 GB 的打包/校验在低配无卡模式可能较慢，因此优先在 GPU 训练结束时顺手打包，再关机；无卡模式更适合补下载和查看小文件。

## 19. 一份可直接照做的命令清单

AutoDL 终端：

```bash
cd /root/autodl-tmp
source /etc/network_turbo
git clone --branch 0814_spatial_inpainting --single-branch https://github.com/hututu1226/twotigers_digital_twins.git
unset http_proxy
unset https_proxy
unzip -q Round2_Map.zip -d twotigers_digital_twins
cd twotigers_digital_twins/schemeB_structured_context_field
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cuda
screen -S schemeb
mkdir -p logs
set -o pipefail
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
python scripts/prepare_final_config.py
bash scripts/run_final.sh 2>&1 | tee logs/final.log
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
bash scripts/package_results.sh
sha256sum -c schemeB_results_*.tar.gz.sha256
```

确认下载/备份后，在 AutoDL 控制台关机。

