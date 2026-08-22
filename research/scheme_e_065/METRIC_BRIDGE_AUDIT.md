# Metric Bridge Audit

## 直接结论

Teacher 中间特征变好，并不等于最终复数信道变好。逐阶段复评没有发现样本顺序、BS 顺序、复数维度、保存精度或重复反归一化错误；损失发生在“粗 PAS/PDP 特征被投影成复数信道，再被 Hybrid 修正”的建模过程。

一个简单例子：草图的轮廓更像，并不保证按这张草图恢复出的每个像素也更像。这里的 coarse Teacher PAS/PDP 是“草图”，最终 `[256,4,192]` 复数信道才是评分对象。

## Canonical Stage Metrics

| Stage | PAS | PDP | NMSE | Score |
|---|---:|---:|---:|---:|
| Base Teacher projected seed | 0.559987 | 0.761072 | 1.269673 | 0.616542 |
| V4 Hybrid raw | - | - | - | 0.624391 |
| V4 post projection | - | - | - | 0.625941 |
| V4 final policy output | 0.567081 | 0.758360 | 1.063711 | 0.627089 |
| V4 saved/reloaded NPY | 0.567081 | 0.758360 | 1.063711 | 0.627089 |
| Adaptive final output | 0.568953 | 0.753717 | 1.089747 | 0.624773 |

The base coarse-feature Teacher metrics were PAS `0.66953`, PDP `0.84482`; the adaptive variant improved them to PAS `0.68143`, PDP `0.85934`. Those numbers live before phase initialization, alternating projection, AE encoding/decoding and Hybrid correction, so they are not interchangeable with final-channel PAS/PDP.

## Checks

- Sample ordering: consistent with the 565 Fold0 indices.
- Cell ordering: consistent with `train_cells`; no BS concatenation swap found.
- Complex layout: consistent with `[sample, 256, 4, 192]`.
- Angle-delay inverse transform: same implementation at audit and inference.
- Power restoration: exactly once.
- Outage policy: same selected per-cell thresholds and strengths.
- Saved dtype: `complex64`; no measurable score loss after reload.
- Canonical row-level and streaming aggregation: agree within numerical tolerance.

## Decision

Metric bridge is considered healthy. The adaptive Teacher path is DROP because its improved intermediate targets reduced final Score by `0.002316`.
