# Scheme F 实验、自动化与验收计划

## 1. 已确认执行边界

- 使用 RTX 5090；参考价格上限 `3 元/小时`。
- GPU 计费预算最多约 12 小时。
- 正式 Scheme F Fold0 最多两次：一次主实验和一次根据指标定向修改的修复实验。
- Fold0 `<0.63` 但结果有限、稳定时，仍完成 4000 条 full-data final 和 500 条测试推理；报告必须标记“不建议提交”。
- NaN/Inf、权重损坏、输出格式错误、单基站 NMSE `>10` 或持续 OOM 时停止，保留并打包已有证据。
- D/E 现有源码、权重和结果只读，不覆盖。

## 2. 实际目录

```text
schemeF_query_adaptive_operator_fusion/
  configs/
    smoke.json
    fold0_5090.json
    final_5090.json
    fold0_selected.json       # D/E 扫描后生成
    fold0_attempt2.json       # 必要时生成
    fold0_best.json           # 两次 Fold 自动选优后生成
    final_selected.json       # 根据最佳 epoch/推理策略生成
  scheme_f/
    preprocessing.py          # 双基站、BEV、RF Gaussian 71维特征
    context_data.py           # latent/OOF prior/同基站集合
    context_model.py          # chart、Top8、regional warp、token Top2、operator、PowerCNP
    context_training.py       # 训练、验证、checkpoint、推理策略
    projection.py             # 保功率 PAS/PDP soft projection
    inference.py              # 500 条流式输出
  scripts/
    prepare_shared_assets.sh
    run_legacy_scans.sh
    run_fold_attempt.sh
    run_all_5090.sh
    run_unattended.sh
    package_results.sh
  docs/
  tests/
```

## 3. 自动流水线阶段

`run_unattended.sh` 调用 `run_all_5090.sh`，串行执行：

1. 从 `/root/autodl-fs` 复用 Scheme C AE 与 Scheme E priors；缺失时自动重建 E OOF/test priors。
2. 校验 CUDA、数据、磁盘和 LFS 权重不是 100 多字节 pointer。
3. Scheme D Router 只读扫描。
4. Scheme E outage/power 只读扫描。
5. 生成 `configs/fold0_selected.json`。
6. 运行单元测试和 GPU 小样本端到端冒烟。
7. 运行 Fold0 attempt 1，自动扫描 outage 和 spectral prior 策略。
8. 根据 attempt 1 指标决定是否运行唯一一次 attempt 2。
9. 选出最高 Score，建立 canonical `artifacts/fold0/context`。
10. 计算全体、BS0、BS1 breakdown。
11. 根据最佳 checkpoint epoch 生成 final config。
12. 用全部 4000 条训练，无 validation 数据被保留。
13. 生成 500 条 `Round2_Test_Channel.npy`。
14. 校验 shape、dtype、finite、checkpoint 和报告。
15. 打包、SHA256、复制到持久盘。
16. 成功或失败均按启动参数自动关机。

中间没有人工输入点；重复运行会复用已完成阶段或从 `last.pt` 恢复。

## 4. D/E 诊断的用途

### 4.1 Scheme D

固定同一份权重和验证子集，扫描：

- 重载错误温度；
- 推断的 checkpoint 温度；
- Top16；
- Top8；
- Top8 + no-warp。

这些是 inference-only counterfactual，不能冒充重新训练后的分数。用途是决定 F 是否收紧 warp 范围，并确认温度恢复问题的实际影响。

### 4.2 Scheme E

固定同一 Hybrid 权重，比较：

- baseline；
- oracle outage；
- oracle power；
- oracle outage + power；
- oracle-free、按基站真值训练分布截断的 GP power；
- bounded GP power + oracle outage。

Oracle 只定位上限，不能进入 final。若 power oracle 增益大，F 自动提高 PowerCNP/quantile 权重并收紧范围；若 outage oracle 增益大，自动加强 BCE 和 positive weight。

## 5. Fold0 attempt 1

默认正式配置：

```text
Context parameters          约 19.19M
Router candidates           Top8
Per-token anchors           Top2
Steps per epoch             48
Maximum epochs              1500
Validation interval         5 epochs
Early stopping patience     100 epochs
Minimum score delta         5e-5
Maximum wall time           3.25 h
AE decoder                  fixed
```

最大 epoch 是保护上限，不是预估收敛 epoch。真正停止条件是验证早停或墙钟上限。

## 6. 唯一一次 attempt 2

只有 attempt 1 `<0.66` 才运行。修复规则由指标触发：

- NMSE `>0.9`：加大 power/quantile loss，收紧 PowerCNP z 范围。
- PAS 明显落后 PDP：提高 chart、Spectrum latent loss。
- token effective neighbors `<1.2`：提高温度和 token diversity，避免过早 Top1。
- Detail warp 几乎为零：缩小 warp 上限并降低 saturation 正则。

attempt 2 使用不同 seed、较低学习率、至少 56 steps/epoch、最多 3 小时。不会继续产生第三次实验。

## 7. 自动推理策略扫描

模型权重固定后，只推理 Context 一次，再复用 latent 预测扫描：

```text
hard outage threshold: 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99, 0.999
soft outage strength:   0, 0.5, 0.75, 1.0
spectral prior alpha:   0, 0.15, 0.30
```

同分时优先更高 outage threshold，减少误杀。最佳三元组写入 `outage_scan.json` 并复制到 final config。

## 8. 安全门槛

### 8.1 硬失败

- 任一关键 loss、PAS、PDP、NMSE、Score 为 NaN/Inf；
- AE/Context 权重是 LFS pointer 或小于合理尺寸；
- BS0 或 BS1 NMSE `>10`；
- 最终数组不是 `[500,256,4,192] complex64`；
- 数组包含 NaN/Inf；
- 打包或 SHA256 校验失败。

### 8.2 结果等级

```text
Score < 0.63   完成 final，但标记“不建议提交”
0.63--0.65     有改进，仍需谨慎
0.65--0.68     有竞争力，建议做多折复核
>= 0.68        接近 0.70 目标，优先提交候选
```

输出格式通过只代表工程链路通过，不代表竞赛分数通过。

## 9. 5090 时间预算

时间取决于旧缓存是否保留、实际每 epoch 秒数和早停位置：

| 阶段 | 有缓存 | 无缓存 |
|---|---:|---:|
| E prior 复用/重建 | 1--3 min | 15--40 min |
| D/E 诊断 | 8--25 min | 缺权重时跳过并记录 |
| GPU 冒烟 | 2--8 min | 2--8 min |
| Fold0 attempt 1 | 0.8--3.25 h | 0.8--3.25 h |
| attempt 2（按需） | 0--3.0 h | 0--3.0 h |
| full-data final | 0.6--3.0 h | 0.6--3.0 h |
| 推理、报告、压缩 | 10--35 min | 10--35 min |
| 总计常见范围 | 2.5--7.5 h | 3--8.5 h |

最坏保护上限接近 10 小时，加上下载/异常余量仍控制在已确认的 12 小时 GPU 预算内。时间是工程估计，不是保证。

## 10. 最终产物

```text
artifacts/fold0/context/best.pt
artifacts/fold0/context/evaluation.json
artifacts/fold0/context/outage_scan.json
artifacts/final/context/final.pt
outputs/final/Round2_Test_Channel.npy
outputs/final/Round2_Test_Channel.json
reports/generated/scheme_d_scan.json
reports/generated/scheme_e_scan.json
reports/generated/fold_attempt_selection.json
reports/generated/fold0_breakdown.json
reports/generated/schemeF_fold0_EXPERIMENT_REPORT.md
reports/generated/schemeF_final_EXPERIMENT_REPORT.md
schemeF_results_YYYYMMDD_HHMMSS.tar.gz
schemeF_results_YYYYMMDD_HHMMSS.tar.gz.sha256
```

持久盘默认目录：`/root/autodl-fs/schemeF_latest`。即使实例关机，该目录仍用于重新下载；实际保留策略以 AutoDL 当前产品规则为准。

## 11. 当前验证状态

本地已完成：

- 26 个单元测试；
- CPU 小样本预处理、AE、encoding、Context、inference；
- 小样本输出 `[2,256,4,192] complex64`；
- architecture full-resolution check；
- spectral projection 保功率测试。

尚未完成并必须由本次 5090 任务给出的证据：

- D/E 云端扫描报告；
- Scheme F GPU smoke；
- 正式 Fold0 分数；
- 4000 条 final 权重；
- 500 条最终测试 NPY。
