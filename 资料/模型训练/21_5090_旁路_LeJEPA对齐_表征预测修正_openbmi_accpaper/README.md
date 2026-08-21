# 21 · 5090 旁路 · LeJEPA 对齐 · 表征预测修正 · OpenBMI Acc_paper

> **方案正文**：[`21系列_LeJEPA对齐_表征预测修正.md`](../../Lejepa_shallow模型方案/掩码未来表征预测_双专家门控_在线MI/实验方案/21系列_LeJEPA对齐_表征预测修正.md)  
> **代码包**：[`5090_mask_future_dual_expert_accpaper/`](../../../code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/)  
> **前置**：方案十七（[`17_5090_.../结果登记表.md`](../17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/结果登记表.md)）

## 设备

NVIDIA RTX 5090 · conda `cyy` · `F:\Cyy\MI`

## 系列目标

推理 **统一仅 `p_cur`**；验证 LeJEPA 对齐修正能否超过 A1（0.5754）。

| 臂 | 改什么 |
|----|--------|
| **F_mi_a** | t0≤1.0 s 重滤 · Predictor 同 A2 |
| **F_mi_080** | 运行时裁切 pf1000→800pt · t0≤1.2 |
| **A2_pt** | 阶段1 L_pred+SIGReg → 阶段2 仅 CE |
| **J1_tok** | past+cur 内块掩码 JEPA · token L_pred |

## 只读基线

| 臂 | Test Acc_paper |
|----|----------------|
| **A1** | **0.5754±0.021** |
| A2 | 0.5667±0.020 |

## 数据

| 臂 | 数据 |
|----|------|
| 全部 21 臂 | `openbmi_2s_hop100_pf1000` v3 |
| F_mi_a / F_mi_080 | 须 **`openbmi_t0_sec.npy`** |
| F_mi_080 | **无需新预处理**（训练时裁切 800pt） |

## 一键启动

```powershell
conda activate cyy
cd F:\Cyy\MI\code\train_lab\src\step\5090_mask_future_dual_expert_accpaper

python _smoke_local.py
python run_arm.py --arm F_mi_a --dry-run

# fold0 探测
python chain_21_all.py --max-folds 1

# 正式五折
$env:Path = "$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;" + $env:Path
powershell -File .\run_21_chain_guarded.ps1 -MaxFolds 0 -NoConsole

# 断点
run_21_chain_resume.bat A2_pt
```

**不算力**：`python chain_21_all.py --max-folds 1 --skip-fmi080 --skip-j1`

## 代码文件

| 文件 | 作用 |
|------|------|
| `train_21_kfold.py` | 21 系列五折训练（含 A2_pt 两阶段） |
| `scheme21_data.py` | t0 过滤 · pf800 裁切 |
| `inwin_jepa.py` | J1 块掩码采样与 target |
| `chain_21_all.py` | 顺序链 |
| `run_21_chain_guarded.ps1` | 长跑 + mem guard |

## 结果登记

[`结果登记表.md`](结果登记表.md)（**F_mi_a 五折完成 · 链待续**）

## 结案

- 任一臂五折 ≥ A1 → 候选在线主系统（仍仅 `p_cur`）  
- 四臂 fold0 均不过线 → 阴性结案 · 主路径 A1 在线
