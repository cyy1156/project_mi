# S09 流水线 · v1.2 严格对齐

**v1.0 / v1.1 结果作废**。仅 `protocol_version=v1.2` + `input_pipeline=noz_unified` 可入主表。

## 重跑

```powershell
cd D:\MI\code\train_lab\src\step\5070_stieger_otta_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py
.\run_all_09.ps1
```

顺序：A0→A3 → B0→B4 → C1（C1 依赖同跑 A0/B3 summary 作崩塌参考）。

## 归档

将 `results/S09-*` 下无 `protocol_version: v1.2` 的 run 移入 `results/archive_v1.0_v1.1/`。
