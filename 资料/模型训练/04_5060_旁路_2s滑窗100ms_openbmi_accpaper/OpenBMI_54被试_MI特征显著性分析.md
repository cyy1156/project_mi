# OpenBMI · 54 被试 MI 特征显著性分析

- 生成时间：`2026-08-08T00:45:12`
- 数据：`D:\cyy\MI\DATA\openbmi\openbmi\openbmi` · **仅 `EEG_MI_train`**（不含 `EEG_MI_test`）· sess01+sess02 同人合并
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

## 1. 总表（54 被试）

| 被试 | 总评 | 通过率 | 左Mu对侧ERD% | 左偏侧pp | 左评级 | 右Mu对侧ERD% | 右偏侧pp | 右评级 | nL/nR/nRest窗 |
|------|------|--------|--------------|----------|--------|--------------|----------|--------|---------------|
| openbmi:subj01 | **中等** | 0.70 | -15.8 | 6.5 | 中等 | -19.6 | 20.1 | 明显 | 2100/2100/2100 |
| openbmi:subj02 | **中等** | 0.50 | -10.0 | -2.8 | 弱/不明显 | -22.1 | 12.8 | 明显 | 2100/2100/2100 |
| openbmi:subj03 | **明显** | 0.80 | -51.2 | 30.6 | 明显 | -51.2 | 20.9 | 明显 | 2100/2100/2100 |
| openbmi:subj04 | **明显** | 0.80 | -35.4 | -3.5 | 明显 | -37.5 | 8.8 | 明显 | 2100/2100/2100 |
| openbmi:subj05 | **中等** | 0.60 | -53.4 | 17.6 | 中等 | -45.6 | 5.8 | 中等 | 2100/2100/2100 |
| openbmi:subj06 | **明显** | 0.80 | -22.8 | 10.6 | 明显 | -31.1 | 10.8 | 明显 | 2100/2100/2100 |
| openbmi:subj07 | **中等** | 0.60 | -23.2 | 7.0 | 中等 | -26.3 | 3.6 | 中等 | 2100/2100/2100 |
| openbmi:subj08 | **中等** | 0.50 | -24.9 | 14.8 | 明显 | -12.7 | -1.4 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj09 | **弱/不明显** | 0.40 | -7.0 | 19.3 | 弱/不明显 | -4.3 | 16.2 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj10 | **弱/不明显** | 0.20 | 33.2 | 6.2 | 弱/不明显 | 51.5 | -14.6 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj11 | **中等** | 0.70 | -29.5 | -7.6 | 中等 | -37.3 | 10.2 | 明显 | 2100/2100/2100 |
| openbmi:subj12 | **中等** | 0.70 | -32.9 | 0.9 | 明显 | -27.0 | -5.2 | 中等 | 2100/2100/2100 |
| openbmi:subj13 | **弱/不明显** | 0.30 | 48.8 | 11.7 | 弱/不明显 | 59.0 | -13.8 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj14 | **弱/不明显** | 0.40 | -5.3 | 3.5 | 弱/不明显 | -12.7 | 9.7 | 中等 | 2100/2100/2100 |
| openbmi:subj15 | **弱/不明显** | 0.30 | -1.3 | 9.5 | 弱/不明显 | 7.1 | -6.0 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj16 | **中等** | 0.70 | -15.9 | -2.3 | 中等 | -25.1 | 8.4 | 明显 | 2100/2100/2100 |
| openbmi:subj17 | **明显** | 0.80 | -69.0 | 11.2 | 明显 | -61.5 | -8.9 | 中等 | 2100/2100/2100 |
| openbmi:subj18 | **明显** | 1.00 | -47.0 | 27.5 | 明显 | -49.2 | 31.3 | 明显 | 2100/2100/2100 |
| openbmi:subj19 | **明显** | 1.00 | -57.7 | 8.1 | 明显 | -61.2 | 10.0 | 明显 | 2100/2100/2100 |
| openbmi:subj20 | **弱/不明显** | 0.30 | 3.8 | -7.5 | 弱/不明显 | 5.9 | 8.2 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj21 | **中等** | 0.70 | -76.4 | 12.6 | 明显 | -68.5 | -2.0 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj22 | **中等** | 0.70 | -18.4 | 6.0 | 中等 | -18.2 | 10.9 | 明显 | 2100/2100/2100 |
| openbmi:subj23 | **中等** | 0.70 | -56.0 | 6.1 | 明显 | -54.0 | 6.9 | 中等 | 2100/2100/2100 |
| openbmi:subj24 | **中等** | 0.50 | -36.0 | 11.6 | 中等 | -15.4 | -1.8 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj25 | **弱/不明显** | 0.40 | -22.6 | 8.4 | 中等 | -10.0 | -8.1 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj26 | **中等** | 0.60 | -29.2 | -1.2 | 中等 | -25.4 | -0.6 | 中等 | 2100/2100/2100 |
| openbmi:subj27 | **中等** | 0.60 | -45.5 | -3.2 | 中等 | -40.9 | 4.0 | 中等 | 2100/2100/2100 |
| openbmi:subj28 | **中等** | 0.70 | -37.4 | -12.8 | 中等 | -44.2 | 29.0 | 明显 | 2100/2100/2100 |
| openbmi:subj29 | **中等** | 0.70 | -19.9 | 10.6 | 明显 | -42.0 | 6.4 | 中等 | 2100/2100/2100 |
| openbmi:subj30 | **中等** | 0.60 | -39.9 | 9.6 | 明显 | -33.6 | 9.1 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj31 | **弱/不明显** | 0.20 | 9.1 | 5.0 | 弱/不明显 | 22.4 | -0.4 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj32 | **明显** | 1.00 | -77.9 | 10.7 | 明显 | -71.8 | 26.4 | 明显 | 2100/2100/2100 |
| openbmi:subj33 | **明显** | 0.80 | -73.0 | -1.0 | 明显 | -71.5 | 7.7 | 明显 | 2100/2100/2100 |
| openbmi:subj34 | **弱/不明显** | 0.30 | 0.7 | -22.4 | 弱/不明显 | -11.7 | 34.3 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj35 | **中等** | 0.60 | -23.5 | 6.9 | 中等 | -14.7 | -7.4 | 中等 | 2100/2100/2100 |
| openbmi:subj36 | **明显** | 0.90 | -70.7 | 39.5 | 明显 | -58.9 | 78.6 | 明显 | 2100/2100/2100 |
| openbmi:subj37 | **中等** | 0.50 | -34.7 | 12.1 | 中等 | -10.4 | 9.9 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj38 | **明显** | 0.80 | -58.0 | 22.3 | 明显 | -32.0 | -17.8 | 中等 | 2100/2100/2100 |
| openbmi:subj39 | **中等** | 0.70 | -45.9 | 4.4 | 明显 | -38.9 | 5.7 | 中等 | 2100/2100/2100 |
| openbmi:subj40 | **弱/不明显** | 0.30 | -15.2 | 4.6 | 弱/不明显 | -4.9 | 2.1 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj41 | **中等** | 0.60 | -20.3 | -2.3 | 弱/不明显 | -28.9 | 8.4 | 明显 | 2100/2100/2100 |
| openbmi:subj42 | **中等** | 0.60 | -32.8 | 0.6 | 中等 | -46.5 | 4.8 | 中等 | 2100/2100/2100 |
| openbmi:subj43 | **中等** | 0.60 | -83.5 | 2.0 | 中等 | -80.1 | 1.4 | 中等 | 2100/2100/2100 |
| openbmi:subj44 | **明显** | 0.90 | -89.2 | 10.6 | 明显 | -84.8 | 8.8 | 明显 | 2100/2100/2100 |
| openbmi:subj45 | **明显** | 0.80 | -70.2 | 15.1 | 明显 | -52.7 | 15.6 | 中等 | 2100/2100/2100 |
| openbmi:subj46 | **弱/不明显** | 0.40 | -14.1 | 18.3 | 中等 | 29.8 | -26.2 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj47 | **中等** | 0.60 | -31.2 | 0.6 | 中等 | -26.1 | 3.7 | 中等 | 2100/2100/2100 |
| openbmi:subj48 | **弱/不明显** | 0.40 | -16.6 | -6.5 | 弱/不明显 | 0.8 | 8.9 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj49 | **弱/不明显** | 0.30 | -2.0 | 19.6 | 弱/不明显 | 5.1 | -16.3 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj50 | **弱/不明显** | 0.40 | -7.1 | 6.4 | 弱/不明显 | -13.1 | 2.5 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj51 | **弱/不明显** | 0.30 | -42.7 | 8.9 | 弱/不明显 | -40.4 | -3.0 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj52 | **弱/不明显** | 0.40 | -54.9 | -0.1 | 弱/不明显 | -48.8 | -1.8 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj53 | **弱/不明显** | 0.20 | -2.0 | 5.4 | 弱/不明显 | 5.2 | 5.4 | 弱/不明显 | 2100/2100/2100 |
| openbmi:subj54 | **弱/不明显** | 0.20 | -6.0 | -3.4 | 弱/不明显 | -19.2 | 11.9 | 弱/不明显 | 2100/2100/2100 |

### 总评分布

- **明显**：12 / 54
- **中等**：24 / 54
- **弱/不明显**：18 / 54

## 2. 分被试明细

### openbmi:subj01 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-9.3% C4=-15.8% | 对侧=-15.8% 同侧=-9.3% | 偏侧pp=6.5
  - CP3/CP4：-21.6% / -16.6% | βL/βH对侧：2.7% / 37.5%
  - 静息Mu占比=0.641 | hop时间降幅=0.011 谷底=2.00s | C3–C4相关 Rest→MI：-0.81→-0.80
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-19.6% C4=0.5% | 对侧=-19.6% 同侧=0.5% | 偏侧pp=20.1
  - CP3/CP4：-28.7% / 9.0% | βL/βH对侧：-11.6% / 61.1%
  - 静息Mu占比=0.641 | hop时间降幅=-0.048 谷底=2.00s | C3–C4相关 Rest→MI：-0.81→-0.80
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj02 · **中等**（通过率 0.50）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（1/5）
  - Mu ERD：C3=-12.8% C4=-10.0% | 对侧=-10.0% 同侧=-12.8% | 偏侧pp=-2.8
  - CP3/CP4：-14.1% / -16.7% | βL/βH对侧：-42.4% / -50.7%
  - 静息Mu占比=0.419 | hop时间降幅=0.016 谷底=1.80s | C3–C4相关 Rest→MI：-0.48→-0.50
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-22.1% C4=-9.3% | 对侧=-22.1% 同侧=-9.3% | 偏侧pp=12.8
  - CP3/CP4：-23.6% / -11.7% | βL/βH对侧：-21.5% / -35.3%
  - 静息Mu占比=0.419 | hop时间降幅=0.079 谷底=1.40s | C3–C4相关 Rest→MI：-0.48→-0.45
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj03 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-20.7% C4=-51.2% | 对侧=-51.2% 同侧=-20.7% | 偏侧pp=30.6
  - CP3/CP4：-20.5% / -36.4% | βL/βH对侧：-27.7% / -5.9%
  - 静息Mu占比=0.631 | hop时间降幅=0.064 谷底=0.40s | C3–C4相关 Rest→MI：-0.77→-0.79
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-51.2% C4=-30.3% | 对侧=-51.2% 同侧=-30.3% | 偏侧pp=20.9
  - CP3/CP4：-35.3% / -11.8% | βL/βH对侧：-24.8% / -0.3%
  - 静息Mu占比=0.631 | hop时间降幅=0.086 谷底=0.40s | C3–C4相关 Rest→MI：-0.77→-0.79
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj04 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-38.9% C4=-35.4% | 对侧=-35.4% 同侧=-38.9% | 偏侧pp=-3.5
  - CP3/CP4：-65.4% / -61.1% | βL/βH对侧：-34.3% / -29.5%
  - 静息Mu占比=0.589 | hop时间降幅=0.161 谷底=2.00s | C3–C4相关 Rest→MI：-0.69→-0.64
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-37.5% C4=-28.7% | 对侧=-37.5% 同侧=-28.7% | 偏侧pp=8.8
  - CP3/CP4：-61.8% / -58.1% | βL/βH对侧：-11.5% / 1.5%
  - 静息Mu占比=0.589 | hop时间降幅=0.056 谷底=1.10s | C3–C4相关 Rest→MI：-0.69→-0.63
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj05 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-35.8% C4=-53.4% | 对侧=-53.4% 同侧=-35.8% | 偏侧pp=17.6
  - CP3/CP4：-21.7% / -50.7% | βL/βH对侧：-58.8% / -65.7%
  - 静息Mu占比=0.458 | hop时间降幅=0.077 谷底=0.40s | C3–C4相关 Rest→MI：-0.75→-0.74
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-45.6% C4=-39.7% | 对侧=-45.6% 同侧=-39.7% | 偏侧pp=5.8
  - CP3/CP4：-28.0% / -35.3% | βL/βH对侧：-45.9% / -52.3%
  - 静息Mu占比=0.458 | hop时间降幅=0.046 谷底=0.30s | C3–C4相关 Rest→MI：-0.75→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj06 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-12.3% C4=-22.8% | 对侧=-22.8% 同侧=-12.3% | 偏侧pp=10.6
  - CP3/CP4：-16.6% / -18.7% | βL/βH对侧：-27.0% / -25.5%
  - 静息Mu占比=0.567 | hop时间降幅=0.001 谷底=0.40s | C3–C4相关 Rest→MI：-0.79→-0.80
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-31.1% C4=-20.2% | 对侧=-31.1% 同侧=-20.2% | 偏侧pp=10.8
  - CP3/CP4：-34.5% / -18.2% | βL/βH对侧：-20.0% / -20.9%
  - 静息Mu占比=0.567 | hop时间降幅=0.030 谷底=0.40s | C3–C4相关 Rest→MI：-0.79→-0.80
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj07 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-16.2% C4=-23.2% | 对侧=-23.2% 同侧=-16.2% | 偏侧pp=7.0
  - CP3/CP4：-8.7% / -13.2% | βL/βH对侧：13.2% / -4.6%
  - 静息Mu占比=0.575 | hop时间降幅=-0.117 谷底=0.20s | C3–C4相关 Rest→MI：-0.71→-0.70
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-26.3% C4=-22.8% | 对侧=-26.3% 同侧=-22.8% | 偏侧pp=3.6
  - CP3/CP4：-28.3% / -11.7% | βL/βH对侧：5.0% / 13.1%
  - 静息Mu占比=0.575 | hop时间降幅=-0.053 谷底=0.30s | C3–C4相关 Rest→MI：-0.71→-0.69
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj08 · **中等**（通过率 0.50）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-10.0% C4=-24.9% | 对侧=-24.9% 同侧=-10.0% | 偏侧pp=14.8
  - CP3/CP4：-29.7% / -33.3% | βL/βH对侧：-12.4% / -25.6%
  - 静息Mu占比=0.793 | hop时间降幅=-0.159 谷底=0.10s | C3–C4相关 Rest→MI：-0.81→-0.83
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=-12.7% C4=-14.1% | 对侧=-12.7% 同侧=-14.1% | 偏侧pp=-1.4
  - CP3/CP4：-31.0% / -21.0% | βL/βH对侧：-22.5% / -10.0%
  - 静息Mu占比=0.793 | hop时间降幅=-0.103 谷底=0.10s | C3–C4相关 Rest→MI：-0.81→-0.83
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj09 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=12.4% C4=-7.0% | 对侧=-7.0% 同侧=12.4% | 偏侧pp=19.3
  - CP3/CP4：12.0% / -4.3% | βL/βH对侧：-15.9% / -31.5%
  - 静息Mu占比=0.664 | hop时间降幅=-0.007 谷底=0.20s | C3–C4相关 Rest→MI：-0.80→-0.81
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-4.3% C4=11.9% | 对侧=-4.3% 同侧=11.9% | 偏侧pp=16.2
  - CP3/CP4：-5.6% / 22.6% | βL/βH对侧：-11.2% / -21.8%
  - 静息Mu占比=0.664 | hop时间降幅=0.017 谷底=0.60s | C3–C4相关 Rest→MI：-0.80→-0.80
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj10 · **弱/不明显**（通过率 0.20）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（1/5）
  - Mu ERD：C3=39.4% C4=33.2% | 对侧=33.2% 同侧=39.4% | 偏侧pp=6.2
  - CP3/CP4：69.0% / 37.3% | βL/βH对侧：-13.1% / -30.7%
  - 静息Mu占比=0.775 | hop时间降幅=-0.155 谷底=0.10s | C3–C4相关 Rest→MI：-0.79→-0.78
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=51.5% C4=36.9% | 对侧=51.5% 同侧=36.9% | 偏侧pp=-14.6
  - CP3/CP4：48.6% / 96.2% | βL/βH对侧：2.8% / 3.9%
  - 静息Mu占比=0.775 | hop时间降幅=-0.109 谷底=0.10s | C3–C4相关 Rest→MI：-0.79→-0.76
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj11 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-37.1% C4=-29.5% | 对侧=-29.5% 同侧=-37.1% | 偏侧pp=-7.6
  - CP3/CP4：-53.2% / -70.1% | βL/βH对侧：-17.5% / -28.7%
  - 静息Mu占比=0.653 | hop时间降幅=0.004 谷底=2.00s | C3–C4相关 Rest→MI：-0.77→-0.79
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-37.3% C4=-27.2% | 对侧=-37.3% 同侧=-27.2% | 偏侧pp=10.2
  - CP3/CP4：-51.6% / -65.9% | βL/βH对侧：-6.1% / -11.2%
  - 静息Mu占比=0.653 | hop时间降幅=0.002 谷底=0.60s | C3–C4相关 Rest→MI：-0.77→-0.79
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj12 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-32.0% C4=-32.9% | 对侧=-32.9% 同侧=-32.0% | 偏侧pp=0.9
  - CP3/CP4：8.8% / -12.1% | βL/βH对侧：-31.9% / -30.4%
  - 静息Mu占比=0.477 | hop时间降幅=0.164 谷底=0.90s | C3–C4相关 Rest→MI：-0.78→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-27.0% C4=-32.2% | 对侧=-27.0% 同侧=-32.2% | 偏侧pp=-5.2
  - CP3/CP4：-4.5% / 14.0% | βL/βH对侧：-21.0% / -28.7%
  - 静息Mu占比=0.477 | hop时间降幅=0.060 谷底=2.00s | C3–C4相关 Rest→MI：-0.78→-0.78
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj13 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=60.5% C4=48.8% | 对侧=48.8% 同侧=60.5% | 偏侧pp=11.7
  - CP3/CP4：68.9% / 47.3% | βL/βH对侧：0.4% / -8.0%
  - 静息Mu占比=0.820 | hop时间降幅=-0.163 谷底=0.00s | C3–C4相关 Rest→MI：-0.65→-0.66
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=59.0% C4=45.2% | 对侧=59.0% 同侧=45.2% | 偏侧pp=-13.8
  - CP3/CP4：53.7% / 61.4% | βL/βH对侧：6.0% / -11.2%
  - 静息Mu占比=0.820 | hop时间降幅=-0.175 谷底=0.00s | C3–C4相关 Rest→MI：-0.65→-0.67
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj14 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（1/5）
  - Mu ERD：C3=-1.8% C4=-5.3% | 对侧=-5.3% 同侧=-1.8% | 偏侧pp=3.5
  - CP3/CP4：-2.5% / -10.4% | βL/βH对侧：-18.5% / -13.5%
  - 静息Mu占比=0.679 | hop时间降幅=-0.016 谷底=0.10s | C3–C4相关 Rest→MI：-0.84→-0.84
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-12.7% C4=-3.0% | 对侧=-12.7% 同侧=-3.0% | 偏侧pp=9.7
  - CP3/CP4：-21.8% / -1.2% | βL/βH对侧：-8.5% / 3.2%
  - 静息Mu占比=0.679 | hop时间降幅=-0.038 谷底=1.20s | C3–C4相关 Rest→MI：-0.84→-0.83
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj15 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=8.2% C4=-1.3% | 对侧=-1.3% 同侧=8.2% | 偏侧pp=9.5
  - CP3/CP4：16.6% / 12.3% | βL/βH对侧：-44.1% / -37.3%
  - 静息Mu占比=0.647 | hop时间降幅=-0.004 谷底=0.60s | C3–C4相关 Rest→MI：-0.61→-0.64
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=7.1% C4=1.1% | 对侧=7.1% 同侧=1.1% | 偏侧pp=-6.0
  - CP3/CP4：-4.9% / 12.2% | βL/βH对侧：-36.8% / -31.0%
  - 静息Mu占比=0.647 | hop时间降幅=-0.056 谷底=0.10s | C3–C4相关 Rest→MI：-0.61→-0.64
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj16 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-18.1% C4=-15.9% | 对侧=-15.9% 同侧=-18.1% | 偏侧pp=-2.3
  - CP3/CP4：-29.9% / -20.6% | βL/βH对侧：4.1% / 38.6%
  - 静息Mu占比=0.581 | hop时间降幅=-0.009 谷底=1.70s | C3–C4相关 Rest→MI：-0.79→-0.80
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-25.1% C4=-16.7% | 对侧=-25.1% 同侧=-16.7% | 偏侧pp=8.4
  - CP3/CP4：-30.2% / -8.9% | βL/βH对侧：-17.5% / -18.1%
  - 静息Mu占比=0.581 | hop时间降幅=0.076 谷底=1.90s | C3–C4相关 Rest→MI：-0.79→-0.79
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj17 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-57.8% C4=-69.0% | 对侧=-69.0% 同侧=-57.8% | 偏侧pp=11.2
  - CP3/CP4：-31.0% / -50.6% | βL/βH对侧：-51.6% / -30.3%
  - 静息Mu占比=0.592 | hop时间降幅=0.149 谷底=1.70s | C3–C4相关 Rest→MI：-0.77→-0.77
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-61.5% C4=-70.4% | 对侧=-61.5% 同侧=-70.4% | 偏侧pp=-8.9
  - CP3/CP4：-43.3% / -44.5% | βL/βH对侧：-45.7% / -21.4%
  - 静息Mu占比=0.592 | hop时间降幅=0.077 谷底=0.30s | C3–C4相关 Rest→MI：-0.77→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj18 · **明显**（通过率 1.00）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-19.5% C4=-47.0% | 对侧=-47.0% 同侧=-19.5% | 偏侧pp=27.5
  - CP3/CP4：-30.2% / -31.4% | βL/βH对侧：-34.6% / -48.0%
  - 静息Mu占比=0.624 | hop时间降幅=0.137 谷底=0.90s | C3–C4相关 Rest→MI：-0.79→-0.78
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 明显（5/5）
  - Mu ERD：C3=-49.2% C4=-18.0% | 对侧=-49.2% 同侧=-18.0% | 偏侧pp=31.3
  - CP3/CP4：-41.8% / -22.3% | βL/βH对侧：-41.8% / -48.3%
  - 静息Mu占比=0.624 | hop时间降幅=0.100 谷底=0.80s | C3–C4相关 Rest→MI：-0.79→-0.78
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y

### openbmi:subj19 · **明显**（通过率 1.00）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-49.5% C4=-57.7% | 对侧=-57.7% 同侧=-49.5% | 偏侧pp=8.1
  - CP3/CP4：-58.4% / -55.7% | βL/βH对侧：-32.5% / -49.3%
  - 静息Mu占比=0.789 | hop时间降幅=0.161 谷底=1.00s | C3–C4相关 Rest→MI：-0.83→-0.83
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 明显（5/5）
  - Mu ERD：C3=-61.2% C4=-51.3% | 对侧=-61.2% 同侧=-51.3% | 偏侧pp=10.0
  - CP3/CP4：-66.3% / -35.5% | βL/βH对侧：-34.3% / -46.2%
  - 静息Mu占比=0.789 | hop时间降幅=0.147 谷底=1.50s | C3–C4相关 Rest→MI：-0.83→-0.82
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y

### openbmi:subj20 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（1/5）
  - Mu ERD：C3=-3.7% C4=3.8% | 对侧=3.8% 同侧=-3.7% | 偏侧pp=-7.5
  - CP3/CP4：3.4% / 16.3% | βL/βH对侧：-42.1% / -70.7%
  - 静息Mu占比=0.606 | hop时间降幅=-0.068 谷底=0.00s | C3–C4相关 Rest→MI：-0.82→-0.80
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=5.9% C4=14.2% | 对侧=5.9% 同侧=14.2% | 偏侧pp=8.2
  - CP3/CP4：-1.1% / 33.8% | βL/βH对侧：-38.4% / -72.4%
  - 静息Mu占比=0.606 | hop时间降幅=-0.048 谷底=2.00s | C3–C4相关 Rest→MI：-0.82→-0.81
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj21 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-63.9% C4=-76.4% | 对侧=-76.4% 同侧=-63.9% | 偏侧pp=12.6
  - CP3/CP4：-63.9% / -70.8% | βL/βH对侧：-78.9% / -53.0%
  - 静息Mu占比=0.510 | hop时间降幅=0.262 谷底=0.70s | C3–C4相关 Rest→MI：-0.70→-0.81
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-68.5% C4=-70.6% | 对侧=-68.5% 同侧=-70.6% | 偏侧pp=-2.0
  - CP3/CP4：-70.5% / -60.2% | βL/βH对侧：-75.5% / -46.7%
  - 静息Mu占比=0.510 | hop时间降幅=0.227 谷底=0.40s | C3–C4相关 Rest→MI：-0.70→-0.78
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj22 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-12.5% C4=-18.4% | 对侧=-18.4% 同侧=-12.5% | 偏侧pp=6.0
  - CP3/CP4：-4.4% / -25.3% | βL/βH对侧：-8.8% / -9.5%
  - 静息Mu占比=0.702 | hop时间降幅=-0.056 谷底=0.20s | C3–C4相关 Rest→MI：-0.79→-0.78
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-18.2% C4=-7.3% | 对侧=-18.2% 同侧=-7.3% | 偏侧pp=10.9
  - CP3/CP4：-10.4% / -3.8% | βL/βH对侧：-4.3% / 30.1%
  - 静息Mu占比=0.702 | hop时间降幅=-0.106 谷底=0.20s | C3–C4相关 Rest→MI：-0.79→-0.76
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj23 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-49.9% C4=-56.0% | 对侧=-56.0% 同侧=-49.9% | 偏侧pp=6.1
  - CP3/CP4：-58.2% / -66.1% | βL/βH对侧：-36.8% / -62.8%
  - 静息Mu占比=0.627 | hop时间降幅=0.136 谷底=0.70s | C3–C4相关 Rest→MI：-0.81→-0.81
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-54.0% C4=-47.2% | 对侧=-54.0% 同侧=-47.2% | 偏侧pp=6.9
  - CP3/CP4：-59.2% / -59.9% | βL/βH对侧：-38.7% / -77.1%
  - 静息Mu占比=0.627 | hop时间降幅=0.072 谷底=0.40s | C3–C4相关 Rest→MI：-0.81→-0.82
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj24 · **中等**（通过率 0.50）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-24.3% C4=-36.0% | 对侧=-36.0% 同侧=-24.3% | 偏侧pp=11.6
  - CP3/CP4：-19.9% / -37.5% | βL/βH对侧：-44.4% / -59.0%
  - 静息Mu占比=0.426 | hop时间降幅=-0.001 谷底=1.00s | C3–C4相关 Rest→MI：-0.69→-0.69
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-15.4% C4=-17.2% | 对侧=-15.4% 同侧=-17.2% | 偏侧pp=-1.8
  - CP3/CP4：-9.1% / -18.7% | βL/βH对侧：-34.1% / -52.6%
  - 静息Mu占比=0.426 | hop时间降幅=-0.032 谷底=0.00s | C3–C4相关 Rest→MI：-0.69→-0.69
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj25 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-14.2% C4=-22.6% | 对侧=-22.6% 同侧=-14.2% | 偏侧pp=8.4
  - CP3/CP4：-14.6% / -20.9% | βL/βH对侧：-46.0% / -59.5%
  - 静息Mu占比=0.545 | hop时间降幅=-0.013 谷底=0.00s | C3–C4相关 Rest→MI：-0.80→-0.81
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=-10.0% C4=-18.1% | 对侧=-10.0% 同侧=-18.1% | 偏侧pp=-8.1
  - CP3/CP4：-8.2% / -11.4% | βL/βH对侧：-39.6% / -41.1%
  - 静息Mu占比=0.545 | hop时间降幅=0.004 谷底=0.50s | C3–C4相关 Rest→MI：-0.80→-0.81
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj26 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-30.4% C4=-29.2% | 对侧=-29.2% 同侧=-30.4% | 偏侧pp=-1.2
  - CP3/CP4：-39.9% / -34.7% | βL/βH对侧：-30.8% / -12.0%
  - 静息Mu占比=0.707 | hop时间降幅=0.010 谷底=2.00s | C3–C4相关 Rest→MI：-0.82→-0.81
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-25.4% C4=-26.0% | 对侧=-25.4% 同侧=-26.0% | 偏侧pp=-0.6
  - CP3/CP4：-40.6% / -32.0% | βL/βH对侧：-20.4% / 31.5%
  - 静息Mu占比=0.707 | hop时间降幅=-0.039 谷底=1.90s | C3–C4相关 Rest→MI：-0.82→-0.81
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj27 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-48.7% C4=-45.5% | 对侧=-45.5% 同侧=-48.7% | 偏侧pp=-3.2
  - CP3/CP4：-52.3% / -61.2% | βL/βH对侧：-49.2% / -50.7%
  - 静息Mu占比=0.532 | hop时间降幅=0.002 谷底=0.50s | C3–C4相关 Rest→MI：-0.75→-0.72
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-40.9% C4=-36.9% | 对侧=-40.9% 同侧=-36.9% | 偏侧pp=4.0
  - CP3/CP4：-46.4% / -54.0% | βL/βH对侧：-21.8% / -32.9%
  - 静息Mu占比=0.532 | hop时间降幅=-0.034 谷底=0.20s | C3–C4相关 Rest→MI：-0.75→-0.74
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj28 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-50.2% C4=-37.4% | 对侧=-37.4% 同侧=-50.2% | 偏侧pp=-12.8
  - CP3/CP4：-33.3% / -48.4% | βL/βH对侧：-14.1% / -34.0%
  - 静息Mu占比=0.748 | hop时间降幅=-0.072 谷底=0.20s | C3–C4相关 Rest→MI：-0.81→-0.81
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-44.2% C4=-15.2% | 对侧=-44.2% 同侧=-15.2% | 偏侧pp=29.0
  - CP3/CP4：-27.9% / -11.7% | βL/βH对侧：-30.4% / -30.8%
  - 静息Mu占比=0.748 | hop时间降幅=-0.059 谷底=0.20s | C3–C4相关 Rest→MI：-0.81→-0.83
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj29 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-9.4% C4=-19.9% | 对侧=-19.9% 同侧=-9.4% | 偏侧pp=10.6
  - CP3/CP4：-48.3% / -57.0% | βL/βH对侧：75.5% / 432.1%
  - 静息Mu占比=0.714 | hop时间降幅=-0.029 谷底=0.20s | C3–C4相关 Rest→MI：-0.70→-0.80
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-42.0% C4=-35.7% | 对侧=-42.0% 同侧=-35.7% | 偏侧pp=6.4
  - CP3/CP4：-54.9% / -54.0% | βL/βH对侧：-29.9% / -38.7%
  - 静息Mu占比=0.714 | hop时间降幅=0.042 谷底=1.90s | C3–C4相关 Rest→MI：-0.70→-0.70
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj30 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-30.4% C4=-39.9% | 对侧=-39.9% 同侧=-30.4% | 偏侧pp=9.6
  - CP3/CP4：-29.3% / -46.4% | βL/βH对侧：-42.2% / -40.3%
  - 静息Mu占比=0.364 | hop时间降幅=0.081 谷底=1.10s | C3–C4相关 Rest→MI：-0.82→-0.81
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=N, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-33.6% C4=-24.4% | 对侧=-33.6% 同侧=-24.4% | 偏侧pp=9.1
  - CP3/CP4：-34.6% / -34.3% | βL/βH对侧：-47.0% / -43.0%
  - 静息Mu占比=0.364 | hop时间降幅=0.066 谷底=1.50s | C3–C4相关 Rest→MI：-0.82→-0.82
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=N, time_pattern=N | excellent=N

### openbmi:subj31 · **弱/不明显**（通过率 0.20）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（1/5）
  - Mu ERD：C3=14.1% C4=9.1% | 对侧=9.1% 同侧=14.1% | 偏侧pp=5.0
  - CP3/CP4：9.4% / -17.0% | βL/βH对侧：-35.2% / -22.2%
  - 静息Mu占比=0.595 | hop时间降幅=-0.068 谷底=0.20s | C3–C4相关 Rest→MI：-0.76→-0.74
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=22.4% C4=22.1% | 对侧=22.4% 同侧=22.1% | 偏侧pp=-0.4
  - CP3/CP4：0.9% / 10.4% | βL/βH对侧：-39.4% / -17.6%
  - 静息Mu占比=0.595 | hop时间降幅=-0.064 谷底=0.20s | C3–C4相关 Rest→MI：-0.76→-0.76
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj32 · **明显**（通过率 1.00）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-67.2% C4=-77.9% | 对侧=-77.9% 同侧=-67.2% | 偏侧pp=10.7
  - CP3/CP4：-68.0% / -82.1% | βL/βH对侧：-54.2% / -40.4%
  - 静息Mu占比=0.598 | hop时间降幅=0.434 谷底=1.20s | C3–C4相关 Rest→MI：-0.78→-0.78
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 明显（5/5）
  - Mu ERD：C3=-71.8% C4=-45.4% | 对侧=-71.8% 同侧=-45.4% | 偏侧pp=26.4
  - CP3/CP4：-76.7% / -42.5% | βL/βH对侧：-53.1% / -48.7%
  - 静息Mu占比=0.598 | hop时间降幅=0.334 谷底=0.80s | C3–C4相关 Rest→MI：-0.78→-0.77
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y

### openbmi:subj33 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-74.0% C4=-73.0% | 对侧=-73.0% 同侧=-74.0% | 偏侧pp=-1.0
  - CP3/CP4：-61.2% / -63.3% | βL/βH对侧：-67.5% / -46.8%
  - 静息Mu占比=0.838 | hop时间降幅=0.116 谷底=1.90s | C3–C4相关 Rest→MI：-0.85→-0.83
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-71.5% C4=-63.8% | 对侧=-71.5% 同侧=-63.8% | 偏侧pp=7.7
  - CP3/CP4：-63.3% / -45.5% | βL/βH对侧：-73.0% / -65.2%
  - 静息Mu占比=0.838 | hop时间降幅=0.135 谷底=2.00s | C3–C4相关 Rest→MI：-0.85→-0.84
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y

### openbmi:subj34 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-21.6% C4=0.7% | 对侧=0.7% 同侧=-21.6% | 偏侧pp=-22.4
  - CP3/CP4：-11.8% / -12.2% | βL/βH对侧：11.3% / -27.5%
  - 静息Mu占比=0.235 | hop时间降幅=0.174 谷底=2.00s | C3–C4相关 Rest→MI：-0.69→-0.71
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=Y, rest_mu_frac=N, time_pattern=Y | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=-11.7% C4=22.6% | 对侧=-11.7% 同侧=22.6% | 偏侧pp=34.3
  - CP3/CP4：-17.8% / 4.9% | βL/βH对侧：-30.6% / -32.0%
  - 静息Mu占比=0.235 | hop时间降幅=0.024 谷底=2.00s | C3–C4相关 Rest→MI：-0.69→-0.69
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=N, time_pattern=N | excellent=N

### openbmi:subj35 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-16.6% C4=-23.5% | 对侧=-23.5% 同侧=-16.6% | 偏侧pp=6.9
  - CP3/CP4：-16.4% / -35.9% | βL/βH对侧：-7.9% / -17.8%
  - 静息Mu占比=0.407 | hop时间降幅=0.037 谷底=0.90s | C3–C4相关 Rest→MI：-0.77→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-14.7% C4=-22.1% | 对侧=-14.7% 同侧=-22.1% | 偏侧pp=-7.4
  - CP3/CP4：-20.4% / -29.2% | βL/βH对侧：-5.4% / -11.8%
  - 静息Mu占比=0.407 | hop时间降幅=0.086 谷底=1.70s | C3–C4相关 Rest→MI：-0.77→-0.76
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=N

### openbmi:subj36 · **明显**（通过率 0.90）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-31.2% C4=-70.7% | 对侧=-70.7% 同侧=-31.2% | 偏侧pp=39.5
  - CP3/CP4：-31.8% / -61.5% | βL/βH对侧：-26.1% / -40.8%
  - 静息Mu占比=0.720 | hop时间降幅=0.278 谷底=0.70s | C3–C4相关 Rest→MI：-0.80→-0.81
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-58.9% C4=19.6% | 对侧=-58.9% 同侧=19.6% | 偏侧pp=78.6
  - CP3/CP4：-55.9% / 20.2% | βL/βH对侧：-15.4% / -48.7%
  - 静息Mu占比=0.720 | hop时间降幅=0.143 谷底=0.50s | C3–C4相关 Rest→MI：-0.80→-0.87
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj37 · **中等**（通过率 0.50）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-22.6% C4=-34.7% | 对侧=-34.7% 同侧=-22.6% | 偏侧pp=12.1
  - CP3/CP4：4.1% / -28.7% | βL/βH对侧：-53.1% / -29.2%
  - 静息Mu占比=0.626 | hop时间降幅=-0.087 谷底=0.20s | C3–C4相关 Rest→MI：-0.80→-0.77
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-10.4% C4=-0.5% | 对侧=-10.4% 同侧=-0.5% | 偏侧pp=9.9
  - CP3/CP4：-9.0% / 22.0% | βL/βH对侧：-42.7% / -33.2%
  - 静息Mu占比=0.626 | hop时间降幅=-0.103 谷底=0.00s | C3–C4相关 Rest→MI：-0.80→-0.81
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj38 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-35.7% C4=-58.0% | 对侧=-58.0% 同侧=-35.7% | 偏侧pp=22.3
  - CP3/CP4：-40.3% / -37.9% | βL/βH对侧：-45.3% / -61.7%
  - 静息Mu占比=0.487 | hop时间降幅=0.128 谷底=1.90s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-32.0% C4=-49.9% | 对侧=-32.0% 同侧=-49.9% | 偏侧pp=-17.8
  - CP3/CP4：-37.1% / -28.0% | βL/βH对侧：-26.6% / -36.4%
  - 静息Mu占比=0.487 | hop时间降幅=0.045 谷底=2.00s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj39 · **中等**（通过率 0.70）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-41.5% C4=-45.9% | 对侧=-45.9% 同侧=-41.5% | 偏侧pp=4.4
  - CP3/CP4：-30.7% / -31.4% | βL/βH对侧：-38.2% / -53.1%
  - 静息Mu占比=0.798 | hop时间降幅=0.095 谷底=0.80s | C3–C4相关 Rest→MI：-0.80→-0.79
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-38.9% C4=-33.3% | 对侧=-38.9% 同侧=-33.3% | 偏侧pp=5.7
  - CP3/CP4：-35.3% / -29.1% | βL/βH对侧：-32.4% / -41.4%
  - 静息Mu占比=0.798 | hop时间降幅=0.027 谷底=0.80s | C3–C4相关 Rest→MI：-0.80→-0.79
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj40 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-10.6% C4=-15.2% | 对侧=-15.2% 同侧=-10.6% | 偏侧pp=4.6
  - CP3/CP4：-10.9% / -28.9% | βL/βH对侧：-46.0% / -48.2%
  - 静息Mu占比=0.529 | hop时间降幅=-0.018 谷底=0.10s | C3–C4相关 Rest→MI：-0.75→-0.74
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=-4.9% C4=-2.8% | 对侧=-4.9% 同侧=-2.8% | 偏侧pp=2.1
  - CP3/CP4：-14.5% / -17.1% | βL/βH对侧：-40.1% / -33.3%
  - 静息Mu占比=0.529 | hop时间降幅=-0.014 谷底=1.60s | C3–C4相关 Rest→MI：-0.75→-0.75
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj41 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-22.5% C4=-20.3% | 对侧=-20.3% 同侧=-22.5% | 偏侧pp=-2.3
  - CP3/CP4：-28.3% / -36.5% | βL/βH对侧：-33.2% / -31.5%
  - 静息Mu占比=0.421 | hop时间降幅=0.022 谷底=0.70s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 明显（4/5）
  - Mu ERD：C3=-28.9% C4=-20.5% | 对侧=-28.9% 同侧=-20.5% | 偏侧pp=8.4
  - CP3/CP4：-36.5% / -45.1% | βL/βH对侧：-28.0% / -30.4%
  - 静息Mu占比=0.421 | hop时间降幅=0.010 谷底=2.00s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj42 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-32.2% C4=-32.8% | 对侧=-32.8% 同侧=-32.2% | 偏侧pp=0.6
  - CP3/CP4：-21.3% / -42.0% | βL/βH对侧：-30.7% / -24.1%
  - 静息Mu占比=0.707 | hop时间降幅=0.004 谷底=2.00s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-46.5% C4=-41.7% | 对侧=-46.5% 同侧=-41.7% | 偏侧pp=4.8
  - CP3/CP4：-49.7% / -46.6% | βL/βH对侧：-35.5% / -40.6%
  - 静息Mu占比=0.707 | hop时间降幅=0.072 谷底=0.40s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj43 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-81.5% C4=-83.5% | 对侧=-83.5% 同侧=-81.5% | 偏侧pp=2.0
  - CP3/CP4：-68.3% / -80.9% | βL/βH对侧：-79.8% / -66.3%
  - 静息Mu占比=0.705 | hop时间降幅=0.312 谷底=0.60s | C3–C4相关 Rest→MI：-0.77→-0.73
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-80.1% C4=-78.7% | 对侧=-80.1% 同侧=-78.7% | 偏侧pp=1.4
  - CP3/CP4：-66.8% / -72.3% | βL/βH对侧：-81.8% / -64.9%
  - 静息Mu占比=0.705 | hop时间降幅=0.222 谷底=0.40s | C3–C4相关 Rest→MI：-0.77→-0.74
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj44 · **明显**（通过率 0.90）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（4/5）
  - Mu ERD：C3=-78.5% C4=-89.2% | 对侧=-89.2% 同侧=-78.5% | 偏侧pp=10.6
  - CP3/CP4：-83.6% / -79.9% | βL/βH对侧：-38.2% / -70.6%
  - 静息Mu占比=0.887 | hop时间降幅=0.124 谷底=0.50s | C3–C4相关 Rest→MI：-0.84→-0.83
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 明显（5/5）
  - Mu ERD：C3=-84.8% C4=-75.9% | 对侧=-84.8% 同侧=-75.9% | 偏侧pp=8.8
  - CP3/CP4：-80.5% / -75.4% | βL/βH对侧：-39.3% / -71.8%
  - 静息Mu占比=0.887 | hop时间降幅=0.097 谷底=0.70s | C3–C4相关 Rest→MI：-0.84→-0.84
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y

### openbmi:subj45 · **明显**（通过率 0.80）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 明显（5/5）
  - Mu ERD：C3=-55.1% C4=-70.2% | 对侧=-70.2% 同侧=-55.1% | 偏侧pp=15.1
  - CP3/CP4：-49.0% / -72.1% | βL/βH对侧：-41.8% / -50.4%
  - 静息Mu占比=0.478 | hop时间降幅=0.152 谷底=1.00s | C3–C4相关 Rest→MI：-0.79→-0.75
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=Y | excellent=Y
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-52.7% C4=-37.1% | 对侧=-52.7% 同侧=-37.1% | 偏侧pp=15.6
  - CP3/CP4：-49.0% / -42.8% | βL/βH对侧：-57.8% / -72.9%
  - 静息Mu占比=0.478 | hop时间降幅=0.071 谷底=0.80s | C3–C4相关 Rest→MI：-0.79→-0.77
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj46 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=4.1% C4=-14.1% | 对侧=-14.1% 同侧=4.1% | 偏侧pp=18.3
  - CP3/CP4：-21.1% / -3.6% | βL/βH对侧：-1.8% / 18.0%
  - 静息Mu占比=0.782 | hop时间降幅=-0.085 谷底=0.20s | C3–C4相关 Rest→MI：-0.67→-0.70
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=29.8% C4=3.6% | 对侧=29.8% 同侧=3.6% | 偏侧pp=-26.2
  - CP3/CP4：-19.3% / 33.8% | βL/βH对侧：7.0% / 15.3%
  - 静息Mu占比=0.782 | hop时间降幅=-0.240 谷底=0.00s | C3–C4相关 Rest→MI：-0.67→-0.68
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj47 · **中等**（通过率 0.60）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 中等（3/5）
  - Mu ERD：C3=-30.6% C4=-31.2% | 对侧=-31.2% 同侧=-30.6% | 偏侧pp=0.6
  - CP3/CP4：-26.4% / -30.4% | βL/βH对侧：-19.1% / -30.7%
  - 静息Mu占比=0.882 | hop时间降幅=-0.111 谷底=0.20s | C3–C4相关 Rest→MI：-0.86→-0.83
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 中等（3/5）
  - Mu ERD：C3=-26.1% C4=-22.3% | 对侧=-26.1% 同侧=-22.3% | 偏侧pp=3.7
  - CP3/CP4：-27.8% / -15.2% | βL/βH对侧：-17.9% / -29.3%
  - 静息Mu占比=0.882 | hop时间降幅=-0.114 谷底=0.10s | C3–C4相关 Rest→MI：-0.86→-0.85
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj48 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-23.1% C4=-16.6% | 对侧=-16.6% 同侧=-23.1% | 偏侧pp=-6.5
  - CP3/CP4：-39.1% / -25.4% | βL/βH对侧：-39.1% / -46.7%
  - 静息Mu占比=0.807 | hop时间降幅=-0.040 谷底=0.00s | C3–C4相关 Rest→MI：-0.63→-0.59
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=0.8% C4=9.7% | 对侧=0.8% 同侧=9.7% | 偏侧pp=8.9
  - CP3/CP4：-32.1% / -1.1% | βL/βH对侧：-23.7% / -33.2%
  - 静息Mu占比=0.807 | hop时间降幅=-0.111 谷底=0.00s | C3–C4相关 Rest→MI：-0.63→-0.64
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj49 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=17.6% C4=-2.0% | 对侧=-2.0% 同侧=17.6% | 偏侧pp=19.6
  - CP3/CP4：-6.3% / -14.2% | βL/βH对侧：-33.9% / -56.4%
  - 静息Mu占比=0.522 | hop时间降幅=-0.072 谷底=0.20s | C3–C4相关 Rest→MI：-0.76→-0.77
  - 检查：mu_erd_contra=N, laterality=Y, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=5.1% C4=-11.2% | 对侧=5.1% 同侧=-11.2% | 偏侧pp=-16.3
  - CP3/CP4：-35.4% / 13.6% | βL/βH对侧：-33.2% / -62.6%
  - 静息Mu占比=0.522 | hop时间降幅=-0.042 谷底=0.20s | C3–C4相关 Rest→MI：-0.76→-0.76
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj50 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-0.8% C4=-7.1% | 对侧=-7.1% 同侧=-0.8% | 偏侧pp=6.4
  - CP3/CP4：-3.8% / -14.8% | βL/βH对侧：10.3% / 17.7%
  - 静息Mu占比=0.548 | hop时间降幅=-0.158 谷底=0.10s | C3–C4相关 Rest→MI：-0.75→-0.77
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-13.1% C4=-10.6% | 对侧=-13.1% 同侧=-10.6% | 偏侧pp=2.5
  - CP3/CP4：-18.5% / -22.4% | βL/βH对侧：11.4% / 40.6%
  - 静息Mu占比=0.548 | hop时间降幅=-0.047 谷底=0.00s | C3–C4相关 Rest→MI：-0.75→-0.78
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=Y, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj51 · **弱/不明显**（通过率 0.30）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-33.8% C4=-42.7% | 对侧=-42.7% 同侧=-33.8% | 偏侧pp=8.9
  - CP3/CP4：-41.1% / -44.5% | βL/βH对侧：-79.4% / -70.2%
  - 静息Mu占比=0.325 | hop时间降幅=-0.007 谷底=2.00s | C3–C4相关 Rest→MI：-0.81→-0.82
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=N, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=-40.4% C4=-43.5% | 对侧=-40.4% 同侧=-43.5% | 偏侧pp=-3.0
  - CP3/CP4：-48.3% / -44.3% | βL/βH对侧：-82.8% / -73.6%
  - 静息Mu占比=0.325 | hop时间降幅=-0.054 谷底=0.20s | C3–C4相关 Rest→MI：-0.81→-0.81
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=N, time_pattern=N | excellent=Y

### openbmi:subj52 · **弱/不明显**（通过率 0.40）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（2/5）
  - Mu ERD：C3=-55.0% C4=-54.9% | 对侧=-54.9% 同侧=-55.0% | 偏侧pp=-0.1
  - CP3/CP4：-36.4% / -51.4% | βL/βH对侧：-61.2% / -43.8%
  - 静息Mu占比=0.660 | hop时间降幅=0.013 谷底=2.00s | C3–C4相关 Rest→MI：-0.81→-0.77
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=Y
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-48.8% C4=-50.6% | 对侧=-48.8% 同侧=-50.6% | 偏侧pp=-1.8
  - CP3/CP4：-39.1% / -43.8% | βL/βH对侧：-58.5% / -44.9%
  - 静息Mu占比=0.660 | hop时间降幅=-0.017 谷底=0.20s | C3–C4相关 Rest→MI：-0.81→-0.78
  - 检查：mu_erd_contra=Y, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=Y

### openbmi:subj53 · **弱/不明显**（通过率 0.20）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（1/5）
  - Mu ERD：C3=3.5% C4=-2.0% | 对侧=-2.0% 同侧=3.5% | 偏侧pp=5.4
  - CP3/CP4：3.6% / -6.5% | βL/βH对侧：-19.5% / -24.9%
  - 静息Mu占比=0.806 | hop时间降幅=0.001 谷底=0.10s | C3–C4相关 Rest→MI：-0.81→-0.82
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（1/5）
  - Mu ERD：C3=5.2% C4=10.6% | 对侧=5.2% 同侧=10.6% | 偏侧pp=5.4
  - CP3/CP4：-11.4% / 6.0% | βL/βH对侧：-24.8% / -31.8%
  - 静息Mu占比=0.806 | hop时间降幅=-0.001 谷底=0.90s | C3–C4相关 Rest→MI：-0.81→-0.83
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=Y, time_pattern=N | excellent=N

### openbmi:subj54 · **弱/不明显**（通过率 0.20）

- 窗数 Left=2100 Right=2100 Rest=2100 · 试次 L/R/Rest=100/100/100 · mats=2
- **左手 MI（期望 C4）** → 弱/不明显（0/5）
  - Mu ERD：C3=-9.4% C4=-6.0% | 对侧=-6.0% 同侧=-9.4% | 偏侧pp=-3.4
  - CP3/CP4：-21.6% / -7.3% | βL/βH对侧：-27.1% / -18.8%
  - 静息Mu占比=0.308 | hop时间降幅=0.055 谷底=2.00s | C3–C4相关 Rest→MI：-0.79→-0.80
  - 检查：mu_erd_contra=N, laterality=N, mu_vs_betal=N, rest_mu_frac=N, time_pattern=N | excellent=N
- **右手 MI（期望 C3）** → 弱/不明显（2/5）
  - Mu ERD：C3=-19.2% C4=-7.3% | 对侧=-19.2% 同侧=-7.3% | 偏侧pp=11.9
  - CP3/CP4：-25.8% / -8.1% | βL/βH对侧：-30.9% / -53.7%
  - 静息Mu占比=0.308 | hop时间降幅=0.048 谷底=1.90s | C3–C4相关 Rest→MI：-0.79→-0.81
  - 检查：mu_erd_contra=Y, laterality=Y, mu_vs_betal=N, rest_mu_frac=N, time_pattern=N | excellent=N

## 3. 结论摘要

OpenBMI 54 名被试、`OpenBMI_MI_feature_v2_hop100`：明显 12、中等 24、弱 18。

- JSON：`D:\cyy\MI\资料\模型训练\04_5060_旁路_2s滑窗100ms_openbmi_accpaper\OpenBMI_54被试_MI特征显著性分析.json`

