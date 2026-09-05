# 5070_mask_future_dual_expert_accpaper

OpenBMI · Acc_paper · **掩码未来表征预测 + 双专家门控**（定稿方案 · **本机正式机位**）。  
**训练设备：NVIDIA RTX 5070 Laptop（~16GB RAM · 8GB VRAM）**

| 项 | 路径 |
|----|------|
| 方案文档 | `资料/模型方案/掩码未来表征预测_双专家门控_在线MI/` |
| 本机计划 | `资料/模型训练/5070_本机配置与实验计划.md` |
| 本包 | `code/train_lab/src/step/5070_mask_future_dual_expert_accpaper/` |
| **5090 大内存姊妹包** | `../5090_mask_future_dual_expert_accpaper/` |
| **5060 历史锚点（只读）** | `../5060_mask_future_dual_expert_accpaper/` |
| 权重 out | `code/train_lab/out/5070_mask_future_dual_expert_accpaper/` |
| A0 数据 | `preprocess_lab/out/openbmi_2s_hop100/` |
| A1+ 数据 | `preprocess_lab/out/openbmi_2s_hop100_pf1000/`（须含 `openbmi_t0_sec.npy`） |

## 与 5090 / 5060 差异

| | **5070（本包）** | 5090 | 5060（历史） |
|--|------------------|------|--------------|
| batch | **128 / 256** | 256 / 512 | 128 / 256 |
| workers / pin | **0～2 / 关** | 4 / 开 | 0 / 关 |
| 默认 folds | **fold0 门控** | 五折 | fold0 |
| 角色 | **本机正式出数** | 大 batch 对照 | 历史锚点（勿覆盖 out） |

臂开关与定稿方案一致；含 **T 系列 v2**（`T1` / `T1_aux` / `T1_128`）。

## 推荐启动（5070）

```powershell
cd code/train_lab/src/step/5070_mask_future_dual_expert_accpaper

python _smoke_local.py
python run_arm.py --arm T1 --dry-run

# T 系列 v2（T1→T1_aux→T1_128）
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 1          # fold0 冒烟
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole # 正式五折

# 主线门控链（fold0）
powershell -File .\run_gate_chain_guarded.ps1

# U 系列五折
powershell -File .\run_u_chain_guarded.ps1 -FromArm U1 -MaxFolds 0 -NoConsole
```

## 结果登记

`资料/模型训练/17_5070_旁路_掩码未来双专家门控_openbmi_accpaper/`（待建 run 后补表）
