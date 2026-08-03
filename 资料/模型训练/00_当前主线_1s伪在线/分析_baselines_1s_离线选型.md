# 分析：`baselines_single` → `baselines_1s`（原结构、不做 RAP）

> 状态：**代码已落地**（P0/P1/P2）；旧 [`baselines_single/`](../../../code/train_lab/src/step/baselines_single/) **未改**。  
> 依据：[`方案_滑窗输入与RAP改造.md`](./方案_滑窗输入与RAP改造.md)。  
> 操作：[`实验操作手册.md`](./实验操作手册.md)。  
> **策略冻结**：  
> 1. 新目录 [`baselines_1s/`](../../../code/train_lab/src/step/baselines_1s/)；  
> 2. 八基线 **原结构 / 默认池化，不做 RAP**；  
> 3. **单库** `bci2a_1s` / `stieger_1s` 分别训练（**不 merged**）；  
> 4. **Val BalAcc 早停** + **train batch balance**（普通 CE）；  
> 5. Task-only；伪在线 / RAP / OTTA → 阶段 B；  
> 6. Stieger 预处理覆盖 **DATA/stieger 全部会话**（`--subjects all`）。

---

## 0. 结论摘要

| 问题 | 结论 |
|------|------|
| 旧目录 | 冻结只读 |
| 新目录 | `code/train_lab/src/step/baselines_1s/` |
| 数据 | `preprocess_lab/out/bci2a_1s`、`out/stieger_1s`，`(N,1,8,250)` |
| Stieger 范围 | **全库**（非仅旧 15 人） |
| 结构 | 原样；**无 RAP** |
| 早停 / 采样 | Val **BalAcc** / **WeightedRandomSampler** 1:1 |
| 一键脚本 | `baselines_1s/run_all.py --data bci2a_1s\|stieger_1s\|both` |
| 旧实验 | 已迁入 [`../归档/`](../归档/)；**不夺冠** |

---

## 1. 已实现布局

```text
code/preprocess_lab/
  src/common/steps/slide_1s.py
  src/common/steps/filter_car.py        # 短试次自适应 FIR（消 notch 警告）
  src/datasets/bci2a/… + preprocess_run_1s
  src/datasets/stieger/pipeline_1s.py / batch_1s.py
  config/bci2a_1s.yaml

code/train_lab/src/step/baselines_1s/
  task_runner.py + 八个 baseline_*.py + run_all.py
```

权重：`train_lab/out/baseline_1s/<model>_1s_balbatch_balacc/<data>/run_<stamp>/`  
MD：`资料/模型训练/runs/<stamp>_<model>_1s_balbatch_balacc/`

---

## 2. 相对旧主线的替换关系

| 旧（归档） | 现（主线） |
|------------|------------|
| 整段 2s/4s / merged 训练 | 1 s / 40 ms 单窗离线 |
| Val F1 或特异度套件夺冠 | Val BalAcc + batch balance 选型 |
| Stieger down = Rest | Cue 前 4 s Rest |
| 直接伪在线 / 窗数组 for | 阶段 B 正式伪在线环 |
| 八基线硬上 RAP | 选型无 RAP；冠军再议 |

---

## 3. 命令

```bash
cd code/preprocess_lab
python -m src.common.steps.test_slide_1s
python -m src.datasets.bci2a.batch --cfg config/bci2a_1s.yaml
python -m src.datasets.stieger.batch_1s --out out/stieger_1s --no-merge --subjects all
python -m src.datasets.stieger.batch_1s --out out/stieger_1s --merge-only

cd ../train_lab/src/step/baselines_1s
python smoke_models.py
python run_all.py --data bci2a_1s
python run_all.py --data stieger_1s --continue-on-error
```

---

## 4. 分模型

- 时域五模型：`(B,8,250)`，默认池化；Deep4@250 可能自动缩核。
- 特征三模型：1 s 窗 μ/β bandpower → `(B,8,2)`。
- **raw 三模型**（`dbn_raw` / `gcbnet_raw` / `dgcnn_raw`）：TemporalEncoder + 图/DBN，输入原始 `(B,8,250)`（来自 `Self_development_model`，`n_times=250`）。
- **统一不做 RAP**。

---

## 5. 阶段 B

冠军确定后伪在线同构上线；仅 EEGNet 可选 RAP 臂 **重训**。见主方案 §5 / §9.3。

---

## 6. 决策冻结

| 项 | 决定 |
|----|------|
| 旧目录 / 旧实验 | 不改代码；结果进归档 |
| 新目录 | `baselines_1s` |
| RAP | 选型阶段不做 |
| 数据 | 单库 `*_1s`；Stieger 全库 |
| 训练 | BalAcc + batch balance + 普通 CE |
| 范围 | 离线 Task 选型 → 再伪在线 |
