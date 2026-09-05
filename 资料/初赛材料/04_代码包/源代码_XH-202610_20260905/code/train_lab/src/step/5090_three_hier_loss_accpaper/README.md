# 5090_three_hier_loss_accpaper · 方案 16（对照 · 全量）

OpenBMI · 2s/hop100 · Acc_paper · **Shallow + Three 复合损失**。  
**训练设备：NVIDIA RTX 5090（内存 128GB · 显存 32GB）** · **本包结果为对照，非正式**。

| 项 | 路径 |
|----|------|
| 姊妹包（本机 5060 · 低内存试探） | `../5060_three_hier_loss_accpaper/` |
| 方案文档 | `资料/模型训练/16_5090_旁路_shallow_Three复合损失_openbmi_accpaper/` |
| 权重 out | `code/train_lab/out/5090_three_hier_loss_accpaper/` |
| 共享数据 | `preprocess_lab/out/openbmi_2s_hop100/` |

## 与 5060 包差异

| | 5060 | **5090（本包）** |
|--|------|------------------|
| out | `5060_three_hier_loss_accpaper` | `5090_three_hier_loss_accpaper` |
| 默认 | fold0 试探、`num_workers=0`、关 pin | **五折全量**、`workers=4`、pin |
| 角色 | 本机冒烟 / 门控 | **正式旁路全量跑数（对照表）** |

损失与臂（S0/H1/H2/H3/T0）与 5060 **完全一致**。

## 5090 机推荐：全链五折

```powershell
cd code/train_lab/src/step/5090_three_hier_loss_accpaper
# 或双击 run_chain_detached.bat
python chain_all.py
```

单臂：

```powershell
python run_arm.py --arm S0 --max-folds 0
python run_arm.py --arm H1 --max-folds 0
python run_arm.py --arm H2 --max-folds 0
python run_arm.py --arm H3 --max-folds 0
python run_arm.py --arm S0 --with-task --skip-three --max-folds 0
```

## 同步

1. `git pull --rebase`
2. 只改本包与 `16_5090_*` 文档；勿改 5060 登记数字
3. `git push`
