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
| 默认 | H* → fold0、`workers=0`、关 pin | 五折、`workers=4`、pin |
| 角色 | 本机冒烟 / OOM 规避 | **128GB 内存全量对照** |

```powershell
cd code/train_lab/src/step/5060_three_hier_loss_accpaper
python run_arm.py --arm S0 --max-folds 1
python run_arm.py --arm H1 --max-folds 1
```

全量请到 `5090_three_hier_loss_accpaper` 跑 `chain_all.py`。
