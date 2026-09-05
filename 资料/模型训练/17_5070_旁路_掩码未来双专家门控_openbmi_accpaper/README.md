# 17 · 5070 旁路 · 掩码未来双专家门控 · OpenBMI Acc_paper

> **本目录方案**：[`方案.md`](./方案.md)  
> **设备**：NVIDIA RTX 5070 Laptop（本机正式机位）  
> **代码包**：`code/train_lab/src/step/5070_mask_future_dual_expert_accpaper/`  
> **out**：`code/train_lab/out/5070_mask_future_dual_expert_accpaper/`  
> **姊妹**：5090 大 batch 对照 · 5060 历史锚点（只读）

## 状态

| 系列 | 状态 | 备注 |
|------|------|------|
| 主线 A→P2 / B / C | 待跑 | 先 fold0 门控 |
| U 系列 | 待跑 | `run_u_chain_guarded.ps1` |
| **T 系列 v2** | **代码就绪** | `T1` / `T1_aux` / `T1_128` · `_smoke_local.py` 已通过 |

## T 系列启动

```powershell
cd code/train_lab/src/step/5070_mask_future_dual_expert_accpaper
python _smoke_local.py
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 1
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole
```

## 结果登记表

（run 完成后在此补 fold 表与 mean±std）
