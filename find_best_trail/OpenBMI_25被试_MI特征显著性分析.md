# OpenBMI · 1 被试 MI 特征显著性分析

- 生成时间：`2026-08-07T20:35:25`
- 数据：`D:\cyy\MI\DATA\openbmi\openbmi\openbmi` · 仅 `EEG_MI_train` · sess01+sess02 同人合并
- 通道：`['Cz', 'C3', 'C4', 'CP3', 'FC4', 'FC3', 'CP4', 'CPz']`（与 hop100 训练一致）
- 参考文档：`脑电特征提取指标与模板量化分析.docx`
- 分析标准版本：`OpenBMI_MI_feature_v2_hop100`

## 0. 切段与重定标准

1. **切段（对齐 hop100）**：Rest=cue前4s(可缩短); MI=cue后0–4s; slide Tw=2s hop=100ms → 窗@250Hz; CAR+notch50+bp8-30; no z-score
2. **不用**窗内 z-score；从 mat 独立重切。
3. **合格阈值**：

| 维度 | OpenBMI 合格线 | 原文档优秀线（对照） |
|------|----------------|----------------------|
| 对侧 Mu ERD | ≤ -15.0%（优秀≤-35.0%） | −50%~−65% |
| 偏侧性 laterality | ≥ 8.0 百分点 | 对侧远强于同侧 |
| Mu vs 低频 Beta | Mu ERD 不弱于 BetaL（+5pp） | Mu 降幅最大 |
| 静息 Mu 占比 | ≥ 0.4 | ≥0.60 |
| 时间形态 | hop 曲线：早段→0.4–0.9s 降≥8%，谷底∈[0.7,2.0]s | 0.5s 起降等 |

计分：左右各 5 项；≥0.8 明显，≥0.5 中等，否则弱。被试总评=左右平均。
左手核心 **C4/CP4**；右手 **C3/CP3**。`nL/nR/nRest` 为 **窗数**（每 MI 试次约 21 窗）。

## 1. 总表（1 被试）

| 被试 | 总评 | 通过率 | 左Mu对侧ERD% | 左偏侧pp | 左评级 | 右Mu对侧ERD% | 右偏侧pp | 右评级 | nL/nR/nRest窗 |
|------|------|--------|--------------|----------|--------|--------------|----------|--------|---------------|
| openbmi:subj01 | **中等** | 0.50 | -10.7 | 3.7 | 弱/不明显 | -6.5 | 18.6 | 中等 | 1050/1050/1050 |

### 总评分布

- **明显**：0 / 1
- **中等**：1 / 1
- **弱/不明显**：0 / 1

## 2. 分被试明细

### openbmi:subj01 · **中等**（通过率 0.50）

- 窗数 Left=1050 Right=1050 Rest=1050 · 试次 L/R/Rest=50/50/50 · mats=1
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-7.0% C4=-10.7% | 对侧=-10.7% 同侧=-7.0% | 偏侧pp=3.7
  - CP3/CP4：-9.8% / -6.1% | βL/βH对侧：37.7% / 107.2%
  - 静息Mu占比=0.686 | hop时间降幅=0.041 谷底=2.00s | C3–C4相关 Rest→MI：-0.83→-0.81
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-6.5% C4=12.1% | 对侧=-6.5% 同侧=12.1% | 偏侧pp=18.6
  - CP3/CP4：-14.4% / 24.6% | βL/βH对侧：-1.5% / 94.5%
  - 静息Mu占比=0.686 | hop时间降幅=-0.039 谷底=2.00s | C3–C4相关 Rest→MI：-0.83→-0.82
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

## 3. 结论摘要

OpenBMI 1 名被试、`OpenBMI_MI_feature_v2_hop100`：明显 0、中等 1、弱 0。

- JSON：`D:\cyy\MI\find_best_trail\OpenBMI_25被试_MI特征显著性分析.json`

