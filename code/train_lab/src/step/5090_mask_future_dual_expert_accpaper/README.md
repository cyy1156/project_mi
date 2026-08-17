# 5090_mask_future_dual_expert_accpaper

OpenBMI · Acc_paper · **掩码未来表征预测 + 双专家门控**（定稿方案）。  
**训练设备：NVIDIA RTX 5090（RAM 128GB · VRAM 32GB）**

| 项 | 路径 |
|----|------|
| 方案文档 | `资料/模型方案/掩码未来表征预测_双专家门控_在线MI/`（姊妹：`资料/Lejepa_shallow模型方案/...`） |
| 数据切片说明 | 同目录 `数据切片与边界过滤说明.md` |
| 本包 | `code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/` |
| 权重 out | `code/train_lab/out/5090_mask_future_dual_expert_accpaper/` |
| A0 数据 | `preprocess_lab/out/openbmi_2s_hop100/`（旧 500pt） |
| A1+ 数据 | `preprocess_lab/out/openbmi_2s_hop100_pf1000/`（**新预处理臂，未改旧代码**） |

框架对齐姊妹包：`5090_three_hier_loss_accpaper`（`shared_hparams` / `chain_all` / `run_arm` / detached bat）。

## 臂一览

| 臂 | 含义 |
|----|------|
| A0_ref | braindecode Shallow 量级参考（官方环） |
| A0 | **主表**：自写 shallow + Expert_cur · 500pt · 仅 Three |
| A1 | 1000pt `X_mask` 单专家 |
| P0 / A2 | + Predictor |
| P1 | 双专家 + Gate + SIGReg（无 Decoder） |
| B1–B10 | 相对 P1 消融（B5→B5a/B5b） |
| **P2** | **主结果**（+Decoder，对齐训练图） |
| C1–C2c | 相对 P2 消融 |
| L1 / A1_600 | 默认不进自动 chain |

## 一键全链（5090）

```powershell
cd code/train_lab/src/step/5090_mask_future_dual_expert_accpaper
# 或双击 run_chain_detached.bat
python chain_all.py
```

断点续跑：

```powershell
python chain_all.py --from P1
# 或 run_chain_resume.bat P1
```

冒烟（每臂仅 fold0）：

```powershell
python chain_all.py --max-folds 1
python run_arm.py --arm P1 --dry-run
```

单臂：

```powershell
python run_arm.py --arm A0 --max-folds 0
python run_arm.py --arm P2 --max-folds 0
```

## 前置条件

1. A0：已有 `openbmi_2s_hop100` npy  
2. A1+：在 `code/preprocess_lab` 跑 `python -m src.datasets.openbmi_pf1000.batch`（写出 `out/openbmi_2s_hop100_pf1000/`；**禁止改旧 preprocess 文件**）  
3. `conda activate cyy`（或仓库 `.venv`）

## 同步

1. `git pull --rebase`  
2. 本包结果登记到 `资料/模型训练/17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/`  
3. `git push`
