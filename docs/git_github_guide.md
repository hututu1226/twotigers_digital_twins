# Git 与 GitHub 项目管理和远程训练指南

## 1. 管理原则

Git 负责版本化：

- Python 源代码；
- JSON 配置；
- 测试；
- Markdown 文档；
- 小型元数据模板；
- 训练逻辑修改记录。

Git 不负责版本化：

- `Round2_Train_Channel.npy` 等比赛原始数据；
- `Round2_Map.zip`；
- 预处理缓存；
- checkpoint；
- 500 条测试信道输出；
- Python 虚拟环境。

本项目根目录的 `.gitignore` 已排除上述大文件。原因：

- 原始信道约 6.3GB，普通 Git/GitHub 无法合理管理；
- checkpoint 经常变化，会快速膨胀仓库历史；
- 比赛数据和输出不应进入公开仓库；
- 代码与数据解耦后，远端机器只需配置 `data.root`。

## 2. 建议使用私有仓库

比赛期间应创建 GitHub Private Repository，原因：

- 避免参赛代码被检索或抄袭；
- 避免违反比赛数据和代码查重要求；
- 可以只邀请提供 GPU 的协作者；
- 比赛结束后再决定是否公开。

不要把 GitHub Personal Access Token、账号密码或服务器 SSH 私钥写进仓库。

## 3. 本机首次初始化 Git

在项目根目录执行：

```bash
git init -b main
git config user.name "你的名字或团队名"
git config user.email "你的GitHub邮箱"
```

如果本机已经全局配置身份，可以省略后两行；查看：

```bash
git config --get user.name
git config --get user.email
```

## 4. 提交前检查忽略规则

检查大文件是否被忽略：

```bash
git check-ignore -v Round2_Map/Round2_Train_Channel.npy
git check-ignore -v Round2_Map.zip
git check-ignore -v artifacts/runs/scheme1_smoke/final.pt
```

每条都应显示来自 `.gitignore` 的匹配规则。

查看候选文件：

```bash
git status --short
```

首次添加：

```bash
git add .
git status --short
```

仔细确认暂存区中没有：

```text
*.npy
*.pt
Round2_Map/
artifacts/
outputs/
.venv/
```

题面 `Problem-Round2.pdf` 体积较小，可以进入私有仓库；若比赛规则不允许分发题面，可在 `.gitignore` 中额外排除。

## 5. 第一次提交

建议提交信息：

```bash
git commit -m "feat: add two Physical AI channel generation pipelines"
```

检查提交：

```bash
git log --oneline --decorate -5
git show --stat --oneline HEAD
```

## 6. 在 GitHub 创建仓库

在 GitHub 网页：

1. 点击 `New repository`；
2. 填写仓库名，例如 `huawei-round2-channel-ai`；
3. 选择 `Private`；
4. 不勾选 README、`.gitignore` 或 License，因为本地已有内容；
5. 创建仓库。

GitHub 会显示仓库 URL，例如：

```text
https://github.com/<owner>/huawei-round2-channel-ai.git
```

绑定远端并上传：

```bash
git remote add origin https://github.com/<owner>/huawei-round2-channel-ai.git
git remote -v
git push -u origin main
```

如果使用 SSH：

```bash
git remote add origin git@github.com:<owner>/huawei-round2-channel-ai.git
git push -u origin main
```

不要同时添加两个名为 `origin` 的远端。

## 7. 邀请提供 GPU 的协作者

GitHub 仓库页面：

```text
Settings -> Collaborators -> Add people
```

输入对方 GitHub 用户名并发送邀请。对方接受后可以克隆私有仓库。

不建议直接共享你的 GitHub 密码或 Token。

## 8. 远端 4070 机器克隆代码

对方执行：

```bash
git clone https://github.com/<owner>/huawei-round2-channel-ai.git
cd huawei-round2-channel-ai
git status
```

此时不会包含比赛数据，这是正确行为。

## 9. 单独传输数据

将整个 `Round2_Map` 目录通过以下任一方式传到远端项目根目录：

- 移动硬盘；
- 私有网盘；
- `scp`；
- `rsync`；
- 局域网文件共享。

Linux 示例：

```bash
rsync -av --progress Round2_Map/ user@gpu-host:/path/huawei-round2-channel-ai/Round2_Map/
```

传输后在远端检查文件大小和数组 shape。不要只看文件名。

```bash
python -c "from pathlib import Path; p=Path('Round2_Map/Round2_Train_Channel.npy'); print(p.stat().st_size)"
python -c "import numpy as np; print(np.load('Round2_Map/Round2_Train_Channel.npy',mmap_mode='r').shape)"
```

预期字节数接近：

```text
6291456128
```

预期 shape：

```text
(4000, 256, 4, 192)
```

## 10. 远端安装和冒烟

对方先安装 CUDA PyTorch 和项目，再执行：

```bash
python scripts/preprocess.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cuda --samples 2
```

只有双方案冒烟通过后才进行长训练。

## 11. 推荐分支工作流

不要让多人直接同时修改 `main`。每个任务建立分支：

```bash
git switch main
git pull --ff-only
git switch -c experiment/scheme1-latent-512
```

修改配置或代码后：

```bash
git status --short
git diff
git add configs/scheme1_4070.json src/
git commit -m "exp: test larger scheme1 latent"
git push -u origin experiment/scheme1-latent-512
```

通过 GitHub Pull Request 审查后合并。即使只有两个人，这也能避免远端训练时不知道某个 checkpoint 对应哪份代码。

## 12. 训练配置必须提交

每次正式实验前：

1. 新建或修改独立配置文件；
2. 提交配置；
3. 记录 commit hash；
4. 再启动训练。

查看当前 hash：

```bash
git rev-parse HEAD
```

建议实验命名：

```text
configs/experiments/s1_latent256_seed2026.json
configs/experiments/s1_latent512_seed2026.json
configs/experiments/s2_k32_seed2026.json
configs/experiments/s2_k48_seed2026.json
```

不要反复覆盖同一个配置后只保留最终版本，否则无法复现中间结果。

## 13. 训练过程中同步代码

远端开始训练前：

```bash
git status --short
git pull --ff-only
git rev-parse HEAD
```

如果 `git status` 显示未提交修改，不要直接 `git pull`。先确认这些修改是谁做的，提交到实验分支或妥善保留。

训练期间不建议切换包含模型代码修改的分支，因为正在运行的 Python 进程与磁盘代码可能不一致。

## 14. checkpoint 和结果如何传回

checkpoint 不进入普通 Git。可使用：

- 私有网盘；
- `scp/rsync`；
- GitHub Release 的私有附件；
- 对象存储。

至少传回：

```text
best.pt
resolved_config.json
history.jsonl
manifest.json
Git commit hash
环境信息
```

环境信息可保存为：

```bash
python -c "import torch,platform; print(platform.platform()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))" > environment.txt
```

测试输出约 750 MiB，也不要进入 Git。

## 15. Git LFS 是否需要

Git LFS 可以管理较大的 checkpoint，但本项目默认不启用，原因：

- GitHub LFS 有存储和流量额度；
- checkpoint 更新频繁；
- 原始 6.3GB 数据仍不适合放在团队代码仓库；
- 私有网盘或对象存储更直接。

只有在团队明确了解额度并希望版本化少量最终权重时再使用：

```bash
git lfs install
git lfs track "release_checkpoints/*.pt"
git add .gitattributes
```

不要对 `artifacts/runs/**/*.pt` 全量开启 LFS，否则每个 epoch 的权重都会消耗额度。

## 16. 发布可复现版本

确定最终模型后：

```bash
git switch main
git pull --ff-only
git tag -a round2-submit-v1 -m "Round 2 submission v1"
git push origin round2-submit-v1
```

在实验记录中保存：

```text
tag: round2-submit-v1
checkpoint: scheme1 best.pt
config: configs/scheme1_4070.json
output SHA256: ...
submission score: ...
```

PowerShell 计算输出哈希：

```powershell
Get-FileHash -Algorithm SHA256 outputs\Round2_Test_Channel.npy
```

Linux：

```bash
sha256sum outputs/Round2_Test_Channel.npy
```

## 17. 日常最小工作流

本机修改：

```bash
git switch main
git pull --ff-only
git switch -c feature/my-change
# 修改和测试
git add <明确文件>
git commit -m "feat: describe the change"
git push -u origin feature/my-change
```

远端训练机更新：

```bash
git switch main
git pull --ff-only
python -m unittest discover -s tests -v
```

不要使用 `git add .` 代替长期的精确提交习惯；首次提交之后，优先显式指定文件，以减少误提交风险。

## 18. 上传前检查清单

- 仓库是 Private；
- `.gitignore` 生效；
- 没有 `.npy/.pt/zip` 大文件进入暂存区；
- 没有 Token、密码、SSH 私钥；
- 单元测试通过；
- 两方案冒烟通过；
- README 和四份方案文档存在；
- 训练配置已提交；
- `git status` 干净；
- 远端 GPU 协作者只获得必要权限。

