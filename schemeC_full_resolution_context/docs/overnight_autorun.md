# AutoDL 夜间全自动训练与关机教程

## 1. 直接结论

可以让 Scheme C 无人值守训练、校验、打包并自动关闭 AutoDL 实例。当前排查 AE 时优先使用独立的 `ae` 模式；确认 AE 合格以后，再分别运行完整 Fold0 和 4000 条全量训练。

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

`scripts/run_overnight_pipeline.sh` 支持四种模式：

- `PIPELINE_MODE=ae`：验证已有容量结果，只训练正式 Fold0 AE，评估、打包并关机，不启动 Context。
- `PIPELINE_MODE=fold0`：默认模式，容量测试、Fold0、打包、关机。
- `PIPELINE_MODE=final`：要求已有合格 Fold0，全量训练、推理、打包、关机。
- `PIPELINE_MODE=all`：连续做前两项，可能超过 8 小时，不推荐受限时使用。

默认 Fold0 模式按顺序执行：

1. 获取独占锁，防止同一项目启动两个夜间脚本。
2. 运行或验证 1/32 样本 AE 容量门槛。
3. Fold0 不完整时从兼容的 `last.pt` 续训；完整时直接跳过。
4. AE 训练后检查 Score、去 Detail 增益和打乱 Detail 下降。
5. 只有 AE 门槛通过才运行 Context 与 Joint。
6. 验证 Fold0 checkpoint、指标、30,720 维 latent 和 500 条输出。
7. 打包 Fold0 并验证 SHA256。
8. 如果已挂载 `/root/autodl-fs`，额外复制一份结果包。
9. 写入成功状态，执行 `/usr/bin/shutdown`。

Final 模式会先重新验证 Fold0 和 AE 门槛，再生成 `final_selected.json`，然后运行 4000 条全量训练、最终推理、格式验证和结果打包。

任一步失败时，脚本会写入失败状态、打包诊断日志，并默认关机，避免你睡觉时实例空转。

`ae` 模式与完整 Fold0 的关键区别是：AE 质量门槛不通过不算自动化故障。脚本仍会把 `best.pt`、`last.pt`、完整验证指标、Detail 消融和日志打成分析包，然后正常关机；它绝不会继续启动 Context。

## 4. 从头执行需要多久

若当前只完成了冒烟测试，总共需要两套训练：

| 部分 | 5090/5090D 工程预估 |
| --- | ---: |
| 已通过容量门槛后，只跑正式 AE、评估和打包 | 约 2.2 到 3.6 小时 |
| 容量门槛 + Fold0 训练、评测、推理 | 约 5.5 到 8 小时 |
| 4000 条全量最终训练和推理 | 约 5 到 8 小时 |
| 总计 | 约 10 到 16 小时 |

因此不要在 8 小时预算下使用 `PIPELINE_MODE=all`。第一晚默认 Fold0 模式完成并关机；确认 Fold0 AE 门槛和分数值得继续后，第二晚使用 Final 模式。如果 Fold0 已完整跑完，Final 模式只做全量阶段。

这是预估，不是保证。实际时间受 5090/5090D 型号、PyTorch SDPA、地图尺寸和验证次数影响。

## 5. 当前只训练 AE 并自动关机

适用于已经完成容量测试、需要离开电脑的情况。先更新代码：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch 0817_schemeC
git pull origin 0817_schemeC
cd schemeC_full_resolution_context
```

确认容量结果仍然有效：

```bash
python scripts/verify_completion.py --stage capacity
```

必须看到最后的 `"status": "PASS"`。然后一键启动：

```bash
bash scripts/launch_ae_autoshutdown.sh
```

启动器会完成以下操作：

1. 再次拒绝无效容量结果，并确认 `/usr/bin/shutdown` 存在。
2. 使用 `nohup` 在后台启动 `PIPELINE_MODE=ae`。
3. 训练或从兼容的 `last.pt` 续训正式 Fold0 AE。
4. 使用 `best.pt` 计算完整验证分数和 Detail 消融。
5. 无论质量门槛是 `PASS` 还是 `FAIL`，都生成 AE 分析包并校验 SHA256。
6. 如果 `/root/autodl-fs` 已挂载且可写，再复制一份持久化备份。
7. 成功、模型质量不达标或程序异常时均自动关机。

启动成功时会打印 `AE automation started successfully` 和后台 PID。此后可以关闭 SSH，不需要保持网页或手机在线。

离开前建议等待约 30 秒，再执行一次：

```bash
cat logs/overnight_status.txt
tail -n 30 logs/overnight_pipeline.log
ps -fp "$(cat logs/overnight.pid)"
```

应看到 `state=RUNNING`，并且日志已经进入环境检查、预处理或 AE 训练。只要不再执行其他训练命令，就可以离开。

完成后的文件名为：

```text
schemeC_ae_analysis_PASS_YYYYMMDD_HHMMSS.tar.gz
schemeC_ae_analysis_PASS_YYYYMMDD_HHMMSS.tar.gz.sha256
```

如果 AE 没通过质量门槛，文件名中的 `PASS` 会变成 `FAIL`。这里表示模型质量门槛，不表示脚本或压缩包损坏。

## 6. 完整 Fold0 或最终训练的命令

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

后台启动第一阶段 Fold0 流水线：

```bash
nohup env \
  PIPELINE_MODE=fold0 \
  CONFIRM_AUTODL_SHUTDOWN=YES \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_overnight_pipeline.sh \
  > logs/overnight_launcher.log 2>&1 &
echo $! > logs/overnight.pid
```

参数含义：

- `nohup`：SSH 或手机网络断开后，脚本仍继续运行。
- `PIPELINE_MODE=fold0`：本次只跑容量门槛和 Fold0，不跨入第二套全量训练。
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

Fold0 下载并确认值得继续后，第二个租卡时段使用：

```bash
nohup env \
  PIPELINE_MODE=final \
  CONFIRM_AUTODL_SHUTDOWN=YES \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_overnight_pipeline.sh \
  > logs/overnight_launcher.log 2>&1 &
echo $! > logs/overnight.pid
```

Final 模式会拒绝使用未通过 AE 质量门槛的 Fold0，不需要手工判断文件是否齐全。

## 7. 睡觉前建议再确认四件事

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

## 8. 如何查看当前进行到哪一步

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

## 9. 各阶段的结果在哪里

Fold0 模式成功后核心文件为：

```text
artifacts/capacity/one_sample.json
artifacts/capacity/thirty_two_samples.json
artifacts/fold0/autoencoder/quality_gate.json
artifacts/fold0/completion_report.json
schemeC_fold0_YYYYMMDD_HHMMSS.tar.gz
schemeC_fold0_YYYYMMDD_HHMMSS.tar.gz.sha256
```

AE 模式完成后核心文件为：

```text
artifacts/fold0/autoencoder/best.pt
artifacts/fold0/autoencoder/evaluation.json
artifacts/fold0/autoencoder/ablation.json
artifacts/fold0/autoencoder/quality_gate.json
artifacts/fold0/autoencoder/completion_report.json
schemeC_ae_analysis_PASS或FAIL_YYYYMMDD_HHMMSS.tar.gz
schemeC_ae_analysis_PASS或FAIL_YYYYMMDD_HHMMSS.tar.gz.sha256
```

Final 模式成功后核心文件为：

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

## 10. 失败时会发生什么

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

## 11. 临时取消自动关机

脚本运行期间创建下面的标志文件：

```bash
touch NO_AUTO_SHUTDOWN
```

流水线仍会继续训练、推理和打包，但最后发现该文件后不会关机。

恢复自动关机：

```bash
rm -f NO_AUTO_SHUTDOWN
```

## 12. 人工停止整条流水线

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

## 13. 最重要的限制

自动关机只能控制费用和流程，不能保证最终 Score 达到 0.7。脚本能证明的是：训练按配置完成、输出格式正确、结果已保存和校验。算法准确率仍必须查看 Fold0 的真实 PAS、PDP、NMSE 和 Score。
