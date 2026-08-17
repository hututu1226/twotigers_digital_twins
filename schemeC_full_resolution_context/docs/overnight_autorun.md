# AutoDL 夜间全自动训练与关机教程

## 1. 直接结论

可以让 Scheme C 从 Fold0 开始，一直执行到 4000 条全量训练、生成 500 条 `Round2_Test_Channel.npy`、校验、打包，最后自动关闭 AutoDL 实例。

可以把它理解成洗衣机的预约程序：不是每隔一分钟猜衣服是否洗完，而是“洗涤程序正常结束并检查排水”后才进入漂洗。这里对应关系是：

- 洗涤程序结束：训练命令返回成功状态。
- 检查排水：核验 checkpoint、历史、指标和 NPY 文件。
- 下一步漂洗：进入全量训练、推理或打包。
- 全部结束断电：执行 `/usr/bin/shutdown`。

## 2. 计费规则必须先确认

AutoDL 官方说明：

- 按量计费实例以实例开关机时间计费，关机后结束 GPU 实例计费。
- 是否正在调用 GPU 不是计费依据，Python 训练结束但实例仍开机，仍然计费。
- 包年包月实例在租期内无论是否关机都会计时，自动关机不能退回已经购买的租期。
- 付费扩容数据盘可能有独立费用，不能理解成关机后所有费用都必然归零。

官方资料：

- [AutoDL 省钱绝招和自动 shutdown](https://www.autodl.com/docs/save_money/)
- [AutoDL 容器实例计费规则](https://api.autodl.com/docs/price/)
- [AutoDL 实例数据保留规则](https://api.autodl.com/docs/instance_data/)

下面脚本解决的是“按量 GPU 实例空转计费”。

## 3. 脚本会做什么

`scripts/run_overnight_pipeline.sh` 按顺序执行：

1. 获取独占锁，防止同一项目启动两个夜间脚本。
2. 检查 Fold0 是否已经完整结束。
3. Fold0 不完整时从 `last.pt` 续训；完整时直接跳过。
4. 验证 Fold0 三阶段 checkpoint、指标、30,720 维 latent 和 500 条输出。
5. 根据 Fold0 最佳 epoch 和 outage threshold 生成 `final_selected.json`。
6. 使用全部 4000 条训练样本执行 AE、Context 和 Joint 最终训练。
7. 生成最终 500 条测试信道。
8. 验证 shape、dtype、NaN/Inf、checkpoint 时间和 encoded latent。
9. 打包模型、输出、日志、配置和说明书，并验证 SHA256。
10. 如果已挂载 `/root/autodl-fs`，额外复制一份结果包。
11. 写入成功状态，执行 `/usr/bin/shutdown`。

任一步失败时，脚本会写入失败状态、打包诊断日志，并默认关机，避免你睡觉时实例空转。

## 4. 从头执行需要多久

若当前只完成了冒烟测试，从 Fold0 到全量最终训练是两套训练：

| 部分 | 5090/5090D 工程预估 |
| --- | ---: |
| Fold0 训练、评测、推理 | 约 5 到 8 小时 |
| 4000 条全量最终训练和推理 | 约 5 到 8 小时 |
| 总计 | 约 10 到 16 小时 |

所以今晚启动后，明早不一定已经结束；但脚本会在真正结束的时刻自行关机。如果 Fold0 已完整跑完，脚本会验证后跳过，剩余时间约为全量阶段用时。

这是预估，不是保证。实际时间受 5090/5090D 型号、PyTorch SDPA、地图尺寸和验证次数影响。

## 5. 今晚直接执行的命令

先进入项目并更新分支：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0817_schemeC
git pull origin 0817_schemeC
cd schemeC_full_resolution_context
```

再次确认冒烟产物有效：

```bash
python scripts/verify_completion.py --stage smoke
```

清除“禁止自动关机”标志并建立日志目录：

```bash
rm -f NO_AUTO_SHUTDOWN
mkdir -p logs
```

后台启动完整流水线：

```bash
nohup env \
  CONFIRM_AUTODL_SHUTDOWN=YES \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_overnight_pipeline.sh \
  > logs/overnight_launcher.log 2>&1 &
echo $! > logs/overnight.pid
```

参数含义：

- `nohup`：SSH 或手机网络断开后，脚本仍继续运行。
- `CONFIRM_AUTODL_SHUTDOWN=YES`：明确授权脚本关闭实例，防止误关本机。
- `SHUTDOWN_ON_SUCCESS=1`：全部成功后关机。
- `SHUTDOWN_ON_FAILURE=1`：中途失败也关机，避免无人监控时持续计费。
- `&`：放到后台执行。
- `$!`：刚启动的后台进程 PID，写入文件便于查询。

启动后立即检查一次：

```bash
cat logs/overnight.pid
ps -fp "$(cat logs/overnight.pid)"
tail -n 30 logs/overnight_launcher.log
```

确认能看到 `Scheme C overnight pipeline started` 后，就可以断开 SSH。

## 6. 睡觉前建议再确认四件事

```bash
nvidia-smi
df -h /root/autodl-tmp
test -x /usr/bin/shutdown && echo shutdown-command-ok
tail -n 30 logs/overnight_pipeline.log
```

必须确认：

- 训练 Python 进程正在占用 GPU。
- 数据盘空间足够。
- `/usr/bin/shutdown` 存在。
- 日志正在增长，没有立即报错。

## 7. 如何查看当前进行到哪一步

查看简短状态：

```bash
cat logs/overnight_status.txt
```

可能看到：

```text
state=RUNNING
stage=4000-sample final training and inference
```

查看完整日志：

```bash
tail -n 100 -f logs/overnight_pipeline.log
```

如果实例已经自动关机，需要从 AutoDL 控制台以无卡模式或正常模式重新开机，再查看日志和下载结果。

## 8. 最终结果在哪里

成功后核心文件为：

```text
outputs/final/Round2_Test_Channel.npy
artifacts/final/autoencoder/final.pt
artifacts/final/context/final.pt
artifacts/final/joint/final.pt
artifacts/final/completion_report.json
logs/overnight_status.txt
schemeC_results_YYYYMMDD_HHMMSS.tar.gz
schemeC_results_YYYYMMDD_HHMMSS.tar.gz.sha256
```

`logs/overnight_status.txt` 应显示：

```text
state=SUCCESS
stage=complete
```

如果 AutoDL 文件存储已经挂载且可写，压缩包还会复制到：

```text
/root/autodl-fs/
```

若没有挂载文件存储，压缩包保留在 `/root/autodl-tmp` 的项目目录。AutoDL 官方说明关机后实例数据仍保留，但本地盘不是冗余存储，重新开机后仍应尽快下载结果包。

## 9. 失败时会发生什么

默认命令设置了 `SHUTDOWN_ON_FAILURE=1`。任一步失败后会：

1. 把 `state=FAILED` 和失败阶段写入 `logs/overnight_status.txt`。
2. 保存 `logs/`、配置和已存在的 summary。
3. 生成 `schemeC_failure_时间.tar.gz`。
4. 尝试复制到 `/root/autodl-fs`。
5. 执行关机。

第二天重新开机后读取：

```bash
cat logs/overnight_status.txt
tail -n 200 logs/overnight_pipeline.log
ls -lh schemeC_failure_*.tar.gz*
```

如果你宁愿失败后保持实例开机方便立即排错，可以把启动参数改为 `SHUTDOWN_ON_FAILURE=0`。这会增加无人监控时继续计费的风险，不推荐晚上睡觉时使用。

## 10. 临时取消自动关机

脚本运行期间创建下面的标志文件：

```bash
touch NO_AUTO_SHUTDOWN
```

流水线仍会继续训练、推理和打包，但最后发现该文件后不会关机。

恢复自动关机：

```bash
rm -f NO_AUTO_SHUTDOWN
```

## 11. 人工停止整条流水线

先禁止自动关机，再终止后台脚本：

```bash
touch NO_AUTO_SHUTDOWN
kill "$(cat logs/overnight.pid)"
```

随后检查是否仍有训练子进程：

```bash
ps -ef | grep -E 'train_autoencoder|train_context|finetune_joint|run_overnight' | grep -v grep
```

如果仍有训练进程，需要单独结束。仅杀掉外层脚本不一定会自动结束已经启动的 Python 子进程。

## 12. 最重要的限制

自动关机只能控制费用和流程，不能保证最终 Score 达到 0.7。脚本能证明的是：训练按配置完成、输出格式正确、结果已保存和校验。算法准确率仍必须查看 Fold0 的真实 PAS、PDP、NMSE 和 Score。
