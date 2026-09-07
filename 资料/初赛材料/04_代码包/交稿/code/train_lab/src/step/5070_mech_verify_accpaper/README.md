# 23 · 5070 机制验证 · OpenBMI Acc_paper

> 方案：[`资料/模型训练/23_旁路_机制验证_未来信息上界与基线复核_openbmi_accpaper/`](../../../../资料/模型训练/23_旁路_机制验证_未来信息上界与基线复核_openbmi_accpaper/)

## 设备

RTX 5070 Laptop · batch **128/256** · AMP · mem_guard

## 一键

```powershell
conda activate cyy
cd F:\Cyy\MI\code\train_lab\src\step\5070_mech_verify_accpaper

python _smoke_local.py
python run_arm.py --arm O2s_m --dry-run

# fold0 探测
python run_arm.py --arm O2s_m --max-folds 1

# Tier1 五折链
python chain_23_all.py --max-folds 0

# 断点
python chain_23_all.py --from L025 --max-folds 0

# 配对统计
python stats.py path/to/O2s_m_run path/to/O2s_f_run
```

## 输出

`code/train_lab/out/5070_mech_verify_accpaper/{stamp}_{arm}/`

oracle 臂目录名带 `_oracle` 后缀（如 `O2s_f_oracle`）。

## Tier1 臂

O2s_m · O2s_f · O1s_m · O1s_f · O600 · L025 · L050 · A1_all

## 校准门

O2s_m 五折 Test Acc_paper ∈ **[0.562, 0.585]** 后再跑后续臂。
