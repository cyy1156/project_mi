# 统一 EEG 预处理 —— 分步代码示例与自检

> 依据：`统一EEG预处理流水线_代码学习指南.md`、项目计划 §3.1.2 双标签设计  
> 用法：按顺序阅读各 `Step*.md`，**自己写代码**；写完后把代码贴给 AI **只做检查、不直接改你的文件**。

---

## 约定（重要）

1. **你写代码，AI 只审阅**：指出问题、给修改建议与对照点；不直接改你仓库里的实现文件。
2. **先单受试者**：用 `A01T.mat` 跑通再扩展。
3. **中间结构统一**：所有 loader 返回 `ContinuousEEG`（见 `00_中间结构.md`）。
4. **最终输出固定**（双标签，与项目计划 §3.1.2 一致）：
   - `X.shape == (N, 1, 8, 1000)`（样本只存一份，不按头复制）
   - `y_task ∈ {0, 1}`：`0`=静息，`1`=任务
   - `y_three ∈ {0, 1, 2}`：`0`=空闲，`1`=左手，`2`=右手
   - 映射：静息→`(0,0)`；左手→`(1,1)`；右手→`(1,2)`
   - 通道顺序：`Cz, C3, C4, CP3, FC4, FC3, CP4, CPz`
   - 采样率：250 Hz，4 s → 1000 点

---

## 推荐文件落地位置（自学工程）

你可以自己建（名称可微调，但步骤对应关系建议保持）：

```text
preprocess_lab/
  src/
    types.py                 # ContinuousEEG
    io/
      load_bci2a_mat.py      # Step1
      load_gdf.py
      load_openbci_csv.py
    steps/
      harmonize_labels.py    # Step2
      select_channels.py     # Step3
      filter_car.py          # Step4–5
      epoch_baseline.py      # Step6–8
      resample_zscore.py     # Step9–11
      split_subjects.py      # Step12（全体混合 8:2 + 跨被试）
    pipeline.py              # 串联
  scripts/
    run_one_subject.py
  out/
```

---

## 学习顺序 ↔ 文档对照

| 顺序 | 文档 | 你要实现的能力 |
|------|------|----------------|
| 0 | `00_中间结构.md` | `ContinuousEEG` + `sanity_check` |
| 1 | `01_Step1_多格式读取.md` | 先写通 `load_bci2a_mat` |
| 2 | `02_Step2_标签同质化.md` | 双标签 `task/three`；脚/舌丢；伪迹丢；静息起点 |
| 3 | `03_Step3_通道筛选.md` | 固定 8 通道，缺通道报错 |
| 4 | `04_Step4_5_CAR与滤波.md` | CAR → Notch50 → Bandpass 8–30 |
| 5 | `05_Step6_8_Epoch基线分类窗.md` | 切窗 / 基线 / 分类窗 |
| 6 | `06_Step9_重采样.md` | → `(1000, 8)` @ 250 Hz |
| 7 | `07_Step10_11_Zscore与张量.md` | trial-wise Z-score + `(N,1,8,1000)` |
| 8 | `08_Step12_按被试划分.md` | 全体试次混合 8:2；后续再跨被试 |
| 9 | `09_串流水线与验收.md` | 拼完整 `preprocess_run` |

---

## 每次提交检查时请附上

把下面模板填好发给 AI（复制即可）：

```text
【步骤】例如：Step3 通道筛选
【文件路径】preprocess_lab/src/steps/select_channels.py
【代码】（完整函数或完整文件）
【自测结果】例如：A01 run3 → x.shape=(..., 8)，通道顺序打印为 ...
【我不确定的点】（可选）
```

AI 会按该步文档的「验收清单」逐项对照，只给审查意见，不直接改代码。

---

## 依赖

```text
numpy
scipy
mne
pandas   # OpenBCI CSV 时需要
scikit-learn  # Step12 全体试次划分
```
