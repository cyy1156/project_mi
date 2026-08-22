# 5090 · 方案 23 机制验证 · OpenBMI Acc_paper

> 方案：[`资料/模型训练/23_旁路_机制验证_未来信息上界与基线复核_openbmi_accpaper/`](../../../../资料/模型训练/23_旁路_机制验证_未来信息上界与基线复核_openbmi_accpaper/)  
> 5070 姊妹包：[`5070_mech_verify_accpaper/`](../5070_mech_verify_accpaper/)

## 设备

**NVIDIA RTX 5090** · batch **256/512** · AMP · 链脚本带外部 mem_guard

## 一键

```powershell
conda activate cyy
cd F:\Cyy\MI\code\train_lab\src\step\5090_mech_verify_accpaper

python _smoke_local.py
python run_arm.py --arm O2s_m --dry-run

# fold0 校准（阶段 0）
python run_arm.py --arm O2s_m --max-folds 1 --num-workers 0

# Tier1 五折链（推荐）
powershell -File .\run_23_chain_guarded.ps1 -MaxFolds 0 -NoConsole

# 断点
python chain_23_all.py --from L025 --max-folds 0
run_23_chain_resume.bat L025

# 配对统计
python stats.py path/to/O2s_m_run path/to/O2s_f_run
```

## 输出

`code/train_lab/out/5090_mech_verify_accpaper/{stamp}_{arm}/`

oracle 臂目录名带 `_oracle`（如 `20260822_120000_O2s_f_oracle`）。

## 与 5070 差异

| 项 | 5070 | **5090（本包）** |
|----|------|------------------|
| batch | 128/256 | **256/512** |
| num_workers | 0 | **4**（链脚本传 0 保稳） |
| mem_guard | 进程内默认开 | **链脚本外部 guard** |
| 校准参考 A1 | 0.5717 / 0.5754 | **0.5754±0.021** |

校准门 O2s_m 五折仍 ∈ **[0.562, 0.585]**。
