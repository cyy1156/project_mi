# 5060_three_hier_loss_accpaper · 方案 16（本机 · 低内存）

OpenBMI · 2s/hop100 · Acc_paper · **Shallow + Three 复合损失**。  
**训练设备：NVIDIA RTX 5060 Laptop（~16GB RAM）** · 旁路试探 / fold0 门控。

| 项 | 路径 |
|----|------|
| 姊妹包（5090 · **全量推荐**） | `../5090_three_hier_loss_accpaper/` |
| 方案文档 | `资料/模型训练/16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/` |
| 权重 out | `code/train_lab/out/5060_three_hier_loss_accpaper/` |

## 与 5090 包差异

| | **5060（本包）** | 5090 |
|--|------------------|------|
| 默认 | H* → fold0、`workers=0`、关 pin、`stream_windows=False`（折内 pack） | 五折、`workers=4`、pin |
| 角色 | 本机冒烟 / OOM 规避 | **大内存全量对照** |

## 内存保护（必读）

本机 `commit_limit≈30.7G`。对照实测（同机同监控）：

| | 方案 14 fold0·1ep | 方案 16 S0 fold0·1ep |
|--|--|--|
| 进程 PageFileUsage 峰值 | ≈ **4.1GB** | ≈ **18–19GB** |
| 系统 commit 峰值 | ≈ **15.8GB** | ≈ **30GB+**（顶满） |
| 结果 | 能跑完 | 易在首个 epoch 内触顶 / 被熔断 |

因此 5060 上跑方案 16 请用看门狗，并优先把 **Windows 虚拟内存固定 ≥32768MB（建议 48GB）**，或改到 5090 包全量跑。

```powershell
cd code/train_lab/src/step/5060_three_hier_loss_accpaper
# 推荐：外部看门狗（防整机卡死）
powershell -File .\run_with_mem_guard.ps1 -Arm S0 -ExtraArgs "--three-only --max-folds 1 --num-workers 0"

# 若 fold0 pack 已在某 run_* 下生成，可复用避免再 pack：
# --resume-dir D:\cyy\MI\code\train_lab\out\5060_three_hier_loss_accpaper\...\run_YYYYMMDD_HHMMSS
```

`run_arm.py` 进程内也会启 `mem_guard`（commit / 工作集 / 空闲内存熔断）。

```powershell
python run_arm.py --arm S0 --max-folds 1
python run_arm.py --arm H1 --max-folds 1
```

全量请到 `5090_three_hier_loss_accpaper` 跑 `chain_all.py`。
