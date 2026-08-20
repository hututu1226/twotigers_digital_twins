# Scheme F 实验与实现计划

## 1. 实现原则

Scheme F 的实现应从 Scheme C 的 AE 和 Scheme E 的几何/频谱预处理复用稳定代码，但新建独立包，不直接修改 Scheme C/D/E。计划目录：

```text
schemeF_query_adaptive_operator_fusion/
  configs/
    smoke.json
    fold0_5090.json
    three_fold_5090.json
  scheme_f/
    data.py
    radio_chart.py
    geometry_operator.py
    anchor_retrieval.py
    latent_transport.py
    token_router.py
    neural_operator.py
    power_cnp.py
    outage.py
    model.py
    training.py
    inference.py
    metrics.py
  scripts/
    check_environment.py
    diagnose_existing_models.py
    preprocess.py
    train_fold.py
    evaluate.py
    prepare_final_config.py
    train_final.py
    infer.py
    verify_completion.py
    package_results.sh
    run_smoke.sh
    run_all_5090.sh
    run_unattended.sh
  tests/
  docs/
```

所有手工选择必须转成配置或 OOF 自动选择，正式无人值守脚本中间不得要求用户输入。

## 2. 第一部分：现有模型融合诊断

在训练 Scheme F 前，由脚本从 Scheme C/D/E Fold0 checkpoint 重新生成逐样本缓存。必须测试：

| 编号 | 复数形状 | 总功率 | 频谱约束 | 目的 |
| --- | --- | --- | --- | --- |
| X0 | Scheme C | Scheme C | 无 | 复现 0.6027 |
| X1 | Scheme D | Scheme D | 无 | 复现 0.5960 |
| X2 | Scheme E | Scheme D | 单位能量 soft prior | 测频谱与稳定功率组合 |
| X3 | Scheme D | 新 PowerCNP oracle-free | 无 | 测功率改进贡献 |
| X4 | Scheme D | Scheme D | E PAS only | 测 PAS prior 独立贡献 |
| X5 | Scheme D | Scheme D | E PDP only | 测 PDP prior 独立贡献 |
| X6 | per-sample D/E oracle | 各自 | 各自 | 仅估计互补上限，不用于提交 |

这里的 oracle 只用于诊断：使用真值为每条验证样本挑较好的旧模型，不能进入正式推理。若 X6 仍低于 `0.65`，说明仅融合旧输出没有足够上限，必须依赖新的 transport/operator。

预估时间：5090 `10--25` 分钟；CPU 约 `40--120` 分钟。

## 3. 第二部分：代码级和小样本冒烟

冒烟测试使用少量样本但覆盖完整链路：

1. 点云 Gaussian token 和 71 维旁路；
2. 两个基站候选严格隔离；
3. chart OOF 不泄漏；
4. Top8/每 token Top2；
5. Spectrum/Detail 位移场尺寸正确；
6. phasor 有限且单位模；
7. 30,720 latent 无全连接瓶颈；
8. PowerCNP 分位数有序；
9. outage 关闭/开启均可推理；
10. AE 解码输出 `[B,256,4,192]` complex；
11. 16 条训练、验证、推理、报告和打包全链路；
12. 断点续跑和失败状态码真实反映结果。

必须增加的回归测试：

- `validation_fold=null` 的 final 配置；
- 报告同时兼容顶层和 `metrics` 嵌套字段；
- `SHUTDOWN_ON_SUCCESS=0` 时成功脚本返回 0；
- LFS pointer 不能被误当权重；
- 任一 BS NMSE 非有限或大于安全阈值时 final gate 失败；
- 输出归一化后频谱约束不得改变总功率。

预估时间：5090 `5--10` 分钟。

## 4. 第三部分：一次上卡自动消融

用户不需要分阶段守在电脑前。`run_all_5090.sh` 自动串行执行以下短实验，并把结果写入统一矩阵：

| 实验 | 改动 | 最大时间 | 继续门槛 |
| --- | --- | ---: | --- |
| A0 | Top1 anchor，无 transport | 20 min | 诊断基线 |
| A1 | chart Top8 + token Top2，无 transport | 30 min | PAS/PDP 高于 A0 |
| A2 | A1 + 低秩位移 | 35 min | 搬运单锚点 loss 明显下降 |
| A3 | A2 + complex phasor | 40 min | NMSE 改善且不伤 PAS |
| A4 | A3 + soft PAS/PDP prior | 30 min | Score 至少再升 0.01 |
| A5 | A4 + PowerCNP/outage | 30 min | 两个 BS NMSE 均有限且 < 3 |

短实验使用相同固定验证集合和相同训练预算，只用于判断模块方向。最高分完整配置随后从头做正式 Fold，不把多个短实验 checkpoint 拼接成 final。

若 A2/A3 完全无收益，正式配置自动退回 token Top2 operator，不强行保留无效 Warp。若 soft prior 在 BS1 降分，门控可以对 BS1 自动关闭，而不是全局放弃频谱教师。

## 5. 第四部分：正式三折训练

### 5.1 划分

- 至少 3 个空间折；
- 每折都按 BS 和最近支持距离分层；
- validation 洞形分布匹配测试连通分量；
- 每折报告全体、BS0、BS1、`0--6 m`、`6--12 m`、`12+ m`；
- 训练过程中所有 teacher 输入使用 OOF 预测。

### 5.2 收敛控制

- `max_epochs`: 1500--2000；
- 每 5 epoch 解码完整 validation；
- ReduceLROnPlateau 以 Score 为准；
- early stopping 至少观察 40 次 validation；
- 最小改进阈值不大于 `5e-5`；
- 单折最大墙钟 2 小时；
- 保存 best、last 和最近 3 个周期性 checkpoint；
- 发生 NaN、BS NMSE > 10 或 power P99 爆炸立即停止该折并标记失败。

### 5.3 正式门槛

| 等级 | 条件 | 决策 |
| --- | --- | --- |
| FAIL | 任一折 < 0.61 或任一 BS NMSE > 3 | 不跑 final |
| WEAK | 三折均值 0.61--0.63 | 仅保留分析包 |
| PROMISING | 三折均值 0.63--0.65 | 可继续一次定向修正 |
| STRONG | 三折均值 0.65--0.68，最差折 >= 0.63 | 运行 final |
| TARGET | 三折均值 >= 0.68，至少一折 >= 0.70 | 作为 0.70 主攻候选 |

不能因为训练 loss 下降就绕过这些门槛。

## 6. 第五部分：4000 条 final 与测试生成

三折通过 STRONG 门槛后自动执行：

1. 由三折最佳 epoch 的中位数和收敛曲线确定 final epoch；
2. 用全部 4000 条重新拟合 chart、teacher、PowerCNP 和 Context；
3. 保持三折确定的超参数，不在测试点上再调；
4. 对 500 个测试点推理；
5. 分别保存 no-outage 和 conservative-outage 两个候选输出；
6. 由 OOF 规则指定主输出，不人工查看测试结果后选择；
7. 校验 shape、dtype、NaN/Inf、exact-zero 数量、功率分位数和 BS 数量；
8. 生成模型卡、实验报告、SHA256 和压缩包。

主输出必须为：

```text
outputs/final/Round2_Test_Channel.npy
shape = [500, 256, 4, 192]
dtype = complex64
```

## 7. 5090 实测基础上的时间预算

已测参考：

- Scheme D Fold0 约 49.6 分钟，final 约 36.0 分钟；
- Scheme E Fold0 Hybrid 约 10.4 分钟，final Hybrid 约 4.9 分钟；
- 两套完整无人值守任务总墙钟约 1 小时 57 分钟。

Scheme F 的 per-token Top2、低秩场和神经算子比 Scheme D 更重，预算为：

| 阶段 | 5090 预计 |
| --- | ---: |
| 现有模型融合诊断 | 10--25 min |
| 预处理/OOF/chart cache | 10--25 min |
| 冒烟与代码检查 | 5--10 min |
| 自动短消融 | 1.5--2.5 h |
| 3 个正式空间折 | 3--5 h |
| 4000 条 final | 45--90 min |
| 测试、报告、打包、校验 | 10--25 min |
| 合计 | 5.5--9 h |

如果只跑 Fold0 进行首轮可行性验证，预计 `1.5--2.5` 小时。完整三折加 final 可能略超过 8 小时，因此无人值守脚本应提供 `FAST=1` 和 `FULL=1`：FAST 在单折不过 `0.63` 时直接停止并关机；FULL 仅在门槛通过后自动继续。

## 8. 无人值守流程契约

未来实现的入口应只有一条：

```bash
bash scripts/run_unattended.sh --mode full --shutdown-on-success
```

脚本必须：

- 启动前检查 GPU、磁盘、数据、AE 权重和 LFS pointer；
- 每阶段写状态文件和独立日志；
- 可从最后完成阶段继续；
- 失败时不删除中间权重；
- 只有 NPY、报告、归档和 SHA256 全部验证通过才视为成功；
- 成功自动关机；
- 失败默认保留实例 15 分钟后关机，并在状态文件写明失败阶段；
- shutdown 开关关闭时仍返回正确的成功退出码。

## 9. 结果包内容

结果包至少包含：

```text
configs/
artifacts/folds/*/best.pt
artifacts/final/best.pt
artifacts/*/summary.json
outputs/final/Round2_Test_Channel.npy
outputs/final/Round2_Test_Channel.json
reports/generated/ablation_matrix.json
reports/generated/fold_breakdown.json
reports/generated/final_completion.json
reports/generated/SCHEME_F_EXPERIMENT_REPORT.md
logs/unattended.log
manifest.sha256
```

权重和 NPY 使用 Git LFS 或 AutoDL 文件存储传输。普通源代码提交不包含训练产物；下载归档保存在根目录已忽略的 `autodl_results/`。

## 10. 实现完成定义

Scheme F 只有同时满足以下条件才算“实现完成”：

1. 单元测试和 CPU shape test 通过；
2. GPU 冒烟完整跑通；
3. 现有 C/D/E 诊断可复现主要指标；
4. 自动消融有统一报告；
5. 至少一个完整空间折训练和评估成功；
6. 全量训练、500 条测试生成、打包可一条命令串行完成；
7. 关机脚本状态码经过回归测试；
8. 算法说明书、AutoDL 运行说明书和实验报告模板齐全。

未达到 Fold 分数门槛不等于代码没完成，但必须在报告中明确写为“算法验证未通过”，不得用输出格式通过代替准确率结论。
