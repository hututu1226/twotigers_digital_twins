# Scheme E-v5：严格外折的局部频谱教师融合

## 结论

V4 的主要限制不是 AE，也不是 PAS/PDP 的 PCA 维数，而是测试位置的粗粒度
PAS/PDP 教师预测不够准。V5 保留现有 GP 教师，并新增两个只预测粗粒度频谱的
局部专家，再用严格外折数据为两个基站分别学习非负融合权重。

这不是把最近邻信道直接作为最终答案。局部专家只提供 PAS/PDP、UE 能量和功率先验，
后续仍由全分辨率 AE、Hybrid 网络、参考信道和交替投影生成复信道。

## 诊断依据

Fold0 的 530 个非零验证样本得到：

- 现有严格 GP：PAS 0.6695，PDP 0.8448。
- GP 与局部专家凸融合：PAS 0.6825，PDP 0.8595。
- 每个样本在所有专家中选最优的上限：PAS 0.7543，PDP 0.9023。
- PAS 128 维 PCA 的重建准确率为 0.9860～0.9871；PDP 64 维超过 0.9998。

因此继续扩大 PCA 维数收益很小；GP 与局部频谱估计的互补性是更直接的改进证据。
BS1 的 GP 指标明显低于 BS0，所以融合权重按基站独立学习。

## 实现

两个局部专家都只使用同一基站的非零训练样本：

- `idw8_p1`：8 个支持点，距离权重为 `1 / d`。
- `idw8_p2`：8 个支持点，距离权重为 `1 / d^2`。

PAS/PDP 先还原到物理功率域再加权，不能直接平均 log 值。UE 能量和 log 功率使用
同一组空间权重。GP 和局部专家的最终权重由外折预测学习，Fold0 验证点不会参与
权重拟合。

## AutoDL Fold0 运行

在项目目录执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins/schemeE_spectral_gaussian_hybrid
set -o pipefail
nohup bash scripts/run_v5_fold0.sh > logs/v5_fold0.log 2>&1 &
echo $! > logs/v5_fold0.pid
tail -f logs/v5_fold0.log
```

脚本会串行完成配置生成、19 项代码测试、无泄漏严格先验重建、V4 最佳权重热启动、
Fold0 训练、策略扫描和输出投影扫描。`tail -f` 使用 `Ctrl+C` 退出不会停止训练。

5090 上预计严格教师重建约 20～50 分钟，Hybrid 训练约 1～2 小时；早停可能提前结束。
最终以 `reports/generated/v5_fold0_output_projection.json` 的 Fold0 离线分数为准，
该分数不是官方测试分数。
