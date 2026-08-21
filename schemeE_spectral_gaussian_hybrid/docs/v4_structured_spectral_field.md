# Scheme E-v4：结构化频谱场优化说明

## 1. 结论先行

E-v3 的最佳严格 Fold0 分数为 `0.626215`，三种容量设置只相差约 `0.0033`。这说明继续增加 epoch 或扩大普通残差网络不是主要突破口。

代码审查发现，E-v3 虽然输入了 PAS/PDP，但条件编码器最后使用全局池化。模型能够知道“频谱中存在强能量”，却很难保留“强能量位于哪个角度格和哪个时延格”。E-v4 不删除 E-v3，也不改变 30,720 维 AE latent；它增加两条可关闭的新路径：

1. 绝对频谱位置编码：将完整 PAS/PDP 展平后映射为条件向量，保留每个 bin 的身份。
2. 结构化频谱场：把 PAS/PDP 自适应池化到 Spectrum 和 Detail latent 的三维网格，再用 3D 卷积逐位置注入。

## 2. 为什么这个改动有针对性

以一条主径为例，旧编码器可能识别出“有一个很强的峰”，但全局平均后不容易区分峰在左侧角度还是右侧角度。比赛 PAS 指标会逐子载波、逐 UE 比较完整阵列角度能量，因此峰的位置错误会直接失分。

E-v4 的对应关系如下：

- PAS 角度轴 -> AE latent 的垂直/水平空间轴；
- PAS 频率代理、PDP 时延轴 -> AE latent 的最后一维；
- 两个基站 ID、71 维几何、参考点信息 -> 继续进入全局条件分支；
- 30,720 维 latent -> 始终保持网格形态，不经过几百维线性瓶颈。

新分支最后一层使用零初始化。训练开始时网络等价于 E-v3 的稳定起点，随后由验证指标决定是否使用结构化频谱场。

## 3. 两个 Fold0 实验

### Attempt 1：冻结 AE decoder

- 只训练频谱位置编码、结构化频谱场、latent adapter、功率头和双种子门控；
- 约 `2.05M` 个可训练参数；
- 用于判断新信息通路本身是否有效。

### Attempt 2：小学习率微调 decoder

- 从 Attempt 1 最佳 checkpoint 初始化；
- decoder 学习率为主学习率的 `0.03`；
- 用于修复“latent 已改善但固定 decoder 无法充分表达”的剩余误差。

### Attempt 3：V3 最佳权重热启动

- 从 V3 Attempt 3 的 `0.626215` checkpoint 加载所有形状兼容的权重；
- 保持旧条件编码器结构，只新增结构化频谱场和对应门控；
- 冻结 decoder，以较小学习率训练；
- 用于避免新分支从头训练时先丢掉已经学会的能力。

三次训练均使用验证集早停和单次时长保护。epoch 是上限，不是必须跑满的经验值；Attempt 3 的上限为 `700` epoch、连续 `100` epoch 无提升早停。

## 4. 输出后投影扫描

E-v4 同时保留 E-v3 的输出后 PAS/PDP 校正扫描。校正完成后会把总功率精确恢复到模型预测值，防止为了改善 PAS/PDP 而让 NMSE 幅度失控。

扫描会为 BS0、BS1 独立选择强度；这是因为已测相位搬运拟合质量差异很大，不能默认两个基站使用同一强度。

## 5. AutoDL 运行

在仓库根目录更新分支后执行：

```bash
cd /root/autodl-tmp/twotigers_digital_twins
git switch codex/0821_schemeE_v3
git pull --ff-only origin codex/0821_schemeE_v3

cd schemeE_spectral_gaussian_hybrid
set -o pipefail
nohup bash scripts/run_v4_fold0.sh \
  > logs/v4_fold0.log 2>&1 < /dev/null &
echo $! > logs/v4_fold0.pid
```

实时查看：

```bash
tail -f logs/v4_fold0.log
```

按 `Ctrl+C` 只退出日志查看，不会停止后台训练。

关键结果：

```text
reports/generated/v4_attempt1_policy.json
reports/generated/v4_attempt1_output_projection.json
reports/generated/v4_attempt2_policy.json
reports/generated/v4_attempt2_output_projection.json
reports/generated/v4_attempt_selection.json
configs/v4_fold_best.json
artifacts/v4/fold0_attempt1/hybrid/best.pt
artifacts/v4/fold0_attempt2/hybrid/best.pt
```

## 6. 5090 时间估计

- 代码检查和模型构建：2--5 分钟；
- Attempt 1：通常 35--80 分钟；
- Attempt 2：通常 45--100 分钟；
- Attempt 3：通常 35--80 分钟；
- 三次策略与输出投影扫描：约 20--50 分钟；
- 总计通常 2--5 小时，极端情况由每次训练的保护上限约束。

## 7. 判定标准

- `<0.63`：新通路没有有效利用频谱位置，应停止继续堆 epoch；
- `0.63--0.65`：方向有效但仍不足，需要升级频谱教师的目标分辨率；
- `>=0.65`：进入 4,000 条全量训练与测试集生成；
- `>=0.70`：达到当前研究目标。

Fold0 是模型选择依据，不等于官方测试集分数。任何离线分数都不能被描述为官方得分。
