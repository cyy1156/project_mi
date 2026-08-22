# S09 流水线 · v1.2 严格对齐

**v1.0 / v1.1 结果作废**。仅 `protocol_version=v1.2` + `input_pipeline=noz_unified` + 24 被试全量可入主表。

**2026-08-23**：已手动停止旧链 `A3,B3,B4,C1`（C1 在 S12 中断）。无效 run 请归档后重跑。

## 重跑

```powershell
cd D:\MI\code\train_lab\src\step\5070_stieger_otta_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe archive_invalid_runs.py   # 归档无效/中断 run
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py             # 可选：S1 冒烟
.\run_all_09.ps1                                                   # 全链（内含归档 + 预检）
```

顺序：归档 → A0→A3 → B0→B4 → **C1 预检** → C1。

C1 要求同目录下已有 **v1.2 全量** A0/B3 summary（24 被试）；否则会报错，不会再用 smoke 锚点 silently 跑全量。

## 归档

`archive_invalid_runs.py` 将：

- 无 `protocol_version: v1.2` → `results/archive_v1.0_v1.1/`
- v1.2 但被试少于 24（smoke）→ `archive_v1.0_v1.1/v1.2_smoke/`
- 中断 C1（少于 24 被试）→ `results/archive_stopped/`
