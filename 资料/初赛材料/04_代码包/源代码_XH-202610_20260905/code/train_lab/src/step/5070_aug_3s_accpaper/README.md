# 5070 · 方案25 代码包

旁路域增广训练 + 增量 FT 配套。方案见  
[`资料/模型训练/25_旁路_域增广训练_增量FT配套_openbmi_accpaper/方案.md`](../../../../../../资料/模型训练/25_旁路_域增广训练_增量FT配套_openbmi_accpaper/方案.md)

## 目录

| 文件 | 作用 |
|------|------|
| `baseline_shallow_aug.py` | G1：OpenBMI 3s 五折 + `--aug g1` |
| `incremental_ft.py` | A0/G2/G3：Stieger 累积增量 FT 爬坡曲线 |
| `eval_stieger.py` | G1：Stieger eval_half 零样本（09-A0 口径） |
| `eval_openbmi_guard.py` | G1：OpenBMI 域内 Test Acc_paper 护栏 |
| `domain_aug.py` | 冻结增广集（z-score 后） |
| `patch_baseline.py` | 注入增广到 baseline DataLoader |
| `s25_config.py` / `s25_weights.py` | 路径与权重解析（避免与 07 包 `config`/`weights` 冲突） |
| `stieger_eval.py` | eval_half noz_unified 评测工具 |
| `smoke_aug_test.py` | 阶段 0 口径单测 |
| `verify_imports.py` | 导入与锚点权重快速校验 |
| `run_all_25_5090.ps1` | **5090 一键链**（G 优先 · A0 末段） |
| `run_25_watch_progress.ps1` | 5090 进度监控 |

权重输出：`train_lab/out/5090_aug_3s_accpaper/`（5090）· `5070_aug_3s_accpaper/`（5070）  
结果登记：`资料/模型训练/25_.../results/`

## 前置

- conda 环境 `cyy`
- `stieger_3s_hop100` 预处理（含 `stieger_X_noz.npy`）
- **5090 A0 锚点 S3**：`5090_alg_incr_3s_hop100_accpaper` · `run_20260823_095327`（three 头）

## 5090 一键运行

```powershell
cd F:\Cyy\MI\code\train_lab\src\step\5070_aug_3s_accpaper
powershell -File .\run_all_25_5090.ps1
powershell -File .\run_25_watch_progress.ps1 -PollSec 60
```

## 5070 一键运行

```powershell
cd D:\MI\code\train_lab\src\step\5070_aug_3s_accpaper
.\run_all_25.ps1
```

## 分步运行

```powershell
cd D:\MI\code\train_lab\src\step\5070_aug_3s_accpaper

python verify_imports.py
python smoke_aug_test.py

# A0 爬坡（S3 锚点）
python incremental_ft.py --arm A0

# G1 五折训练
python baseline_shallow_aug.py --aug g1

# G1 评测（--run-stamp 填 G1 产出目录名）
python eval_openbmi_guard.py --run-stamp run_YYYYMMDD_HHMMSS
python eval_stieger.py --arm G1 --run-stamp run_YYYYMMDD_HHMMSS
python incremental_ft.py --arm G1 --run-stamp run_YYYYMMDD_HHMMSS

# G2 / G3
python incremental_ft.py --arm G2 --run-stamp run_... --replay-ratio 0.15
python incremental_ft.py --arm G3 --run-stamp run_... --aug g3
```

## 增广规格（G1 冻结）

z-score **之后**、训练期、p=0.5、每窗一种：

1. 高斯噪声 σ∈{0.1,0.2,0.3}
2. 循环时移 ±25 点
3. 通道 dropout p=0.1

G3 FT 轻增广：仅 #1+#2，σ/时移减半、p=0.3。

## 冒烟

```powershell
python incremental_ft.py --arm A0 --smoke --subjects S1
python eval_stieger.py --arm A0 --smoke --subjects S1 --tasks three
python baseline_shallow_aug.py --aug g1 --max-folds 1 --skip-task
```

## 臂说明

| 臂 | 权重来源 | 增量 FT 差异 |
|----|----------|--------------|
| A0 | S3 锚点 | 基线爬坡曲线 |
| G1 | G1 五折产出 | 同 A0 流程 |
| G2 | G1 权重 | batch 混 10–20% OpenBMI replay |
| G3 | G1 权重 | FT 期轻增广（`--aug g3`） |

增量 FT 为**累积式**：k 增大时在上一 checkpoint 权重上对前 k 个 cue 全量 FT，评测口径为 eval_half + noz_unified（对齐 09-A0）。
