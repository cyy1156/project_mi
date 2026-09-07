# Task 特异度提升（二分类：静息 vs 任务）

本目录专门存放 **提高 Task Specificity** 相关的实验代码与结论，与常规八模型基线脚本分开，避免和 `baseline_*.py` 主入口混淆。

规划文档（仓库资料区）：  
[`资料/模型训练/方案_Task特异度提升_指标与改码计划.md`](../../../../../../资料/模型训练/方案_Task特异度提升_指标与改码计划.md)

**当日文档**  
- 实验步骤：[`步骤_20260801_下午特异度实验.md`](./步骤_20260801_下午特异度实验.md)  
- 数字结论：[`结论_20260801_Task特异度.md`](./结论_20260801_Task特异度.md)

---

## 目录文件

| 文件 | 作用 |
|------|------|
| `baseline_shallow_wce2_balacc.py` | **主特异度臂**：加权 CE + BalAcc 早停；Task+Three（从原 `baseline_shallow` 改出） |
| `task_objective.py` | 加权 CE（`fixed` / `invfreq`）+ **Focal Loss**（本目录专用；上级旧副本已移除） |
| `baseline_eegnet_focal_balacc.py` | 实验 **F1**：EEGNet + Focal(γ=2, α_rest=0.75) + BalAcc；仅 Task |
| `task_sampler.py` | train 类逆频 `WeightedRandomSampler`（batch balance） |
| `task_smote.py` | train 折 SMOTE（展平 8×T） |
| `baseline_shallow_balbatch.py` | 臂 B1/B2：balance ± 加权，仅 Task |
| `baseline_shallow_smote.py` | 臂 S1/S2：SMOTE ± 加权，仅 Task |
| `eval_threshold_sweep.py` | 对已有 `best_task.pt` 做 τ 扫描（不训练） |
| `步骤_20260801_下午特异度实验.md` | 2026-08-01 下午实验流程（按步复现） |
| `结论_20260801_Task特异度.md` | 2026-08-01 下午实验数字与结论 |

**说明**：上级 `../baseline_shallow.py` 已恢复为**旧基线**（普通 CE + Val F1 早停，`MODEL_NAME=shallow`）。特异度相关请只用本目录脚本。

---

## 验收口径（五折均值）

- Spec ≥ **0.40**
- Rec ≥ **0.75**
- BalAcc ≥ **0.65**
- F1 仅附报

---

## 常用命令

在本目录下执行：

```bash
cd code/train_lab/src/step/baselines_single/task_specificity

# 主特异度臂（加权CE + BalAcc）
python baseline_shallow_wce2_balacc.py --data merged_2s

# 旧基线对照（在上级目录）
python ../baseline_shallow.py --data merged_2s

# batch balance
python baseline_shallow_balbatch.py --data merged_2s --arm B1
python baseline_shallow_balbatch.py --data merged_2s --arm B2

# SMOTE
python baseline_shallow_smote.py --data merged_2s --arm S1
python baseline_shallow_smote.py --data merged_2s --arm S2

# 阈值扫描（示例：w0=2.0 / 2.2 已有 run）
python eval_threshold_sweep.py --data merged_2s --run_dir ../../../../out/baseline/shallow_wce2_balacc/merged_2s/run_20260801_153617
python eval_threshold_sweep.py --data merged_2s --run_dir ../../../../out/baseline/shallow_wce2_balacc/merged_2s/run_20260801_161301

# Focal Loss 单模型（EEGNet，不改上级 baseline_eegnet.py）
python baseline_eegnet_focal_balacc.py --data merged_2s
```

依赖：`imbalanced-learn`（SMOTE 臂需要；已在项目 `.venv` 安装过则可跳过）。

---

## 七模型套件（其余模型一键跑步骤 A/B/C）

```bash
# 训练：eegnet/deep/eegtcnet/conformer/dbn/gcbnet/dgcnn × A22,A20,B1,B2,S1
python suite_runner.py --models all --arms A22,A20,B1,B2,S1 --data merged_2s

# 训练结束后：阈值扫描(A20/A22) + 写八模型总汇总 MD
# （默认合并 shallow_progress.json；已有 threshold_sweep.json 会复用）
python suite_report.py --progress suite_progress.json
```

进度：`suite_progress.json`（七模型）+ `shallow_progress.json`（Shallow）→ `suite_progress_all8.json`。  
汇总：`汇总_八模型特异度_最新.md`（及带时间戳副本）；资料区同步一份。  
Shallow 叙事细节仍见 `结论_20260801_Task特异度.md`。

---

## 输出位置（与其它基线相同约定）

- 权重：`code/train_lab/out/baseline/<model_name>/<data>/run_<stamp>/`
- MD：`资料/模型训练/runs/<stamp>_<model_name>/`
