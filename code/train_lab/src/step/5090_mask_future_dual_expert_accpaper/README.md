# 5090_mask_future_dual_expert_accpaper

OpenBMI · Acc_paper · **掩码未来表征预测 + 双专家门控**（定稿方案）。  
**训练设备：NVIDIA RTX 5090（RAM 128GB · VRAM 32GB）**

| 项 | 路径 |
|----|------|
| 方案文档 | `资料/模型方案/掩码未来表征预测_双专家门控_在线MI/`（姊妹：`资料/Lejepa_shallow模型方案/...`） |
| 数据切片说明 | 同目录 `数据切片与边界过滤说明.md` |
| 本包 | `code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/` |
| **5060 低内存姊妹包** | `../5060_mask_future_dual_expert_accpaper/` |
| 权重 out | `code/train_lab/out/5090_mask_future_dual_expert_accpaper/` |
| A0 数据 | `preprocess_lab/out/openbmi_2s_hop100/`（旧 500pt · 含 Rest） |
| A1+ 数据 | `preprocess_lab/out/openbmi_2s_hop100_pf1000/`（**三类** past+cur+future；`protocol_version≥3`） |

### A1+ 数据切割（冻结）

| 状态 | 切段 | 标签 |
|------|------|------|
| Left/Right | 从 **cue 起** `[cue, cue+5.6s)`（不读 cue 前；段首 0.5s 仅基线均值） | 1 / 2 |
| Rest（空闲） | Cue 前满 5.6s 同几何（评分 4s + post 1.6s），整段在 Cue 前 | 0 |
| 窗 | past100+cur500+future400；合法 t0∈{0.4…2.0} hop0.1 | — |

旧版 `no_rest` / `protocol_version<3` 的 npy **会被 `data_io` 拒绝**，需 `--reset` 重跑预处理。

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
| U1 / U2 / U3 | 相对 P2 结构升级（时间 Predictor / Spectral Decoder / Gate 熵）；默认不进 chain |
| U12 / U13 / U123 | U 组合附报；默认不进 chain；计划顺序 **U13→U12→U123** |
| **T1 / T1_aux / T1_128** | v2 Token + Phase Query Predictor（相对 P2）；`run_t_chain_guarded.ps1` |
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
python _smoke_local.py   # 含 U1–U3 与 U12/U13/U123
```

单臂：

```powershell
python run_arm.py --arm A0 --max-folds 0
python run_arm.py --arm P2 --max-folds 0
python run_arm.py --arm U13 --max-folds 0
```

## U 系列全量链（5090 · 无内存守护）

顺序：**U1 → U3 → U2 → U13 → U12 → U123**（`shared_hparams` 默认 batch 256/512、patience 20、max_epochs 300；不传 `--batch-train` 即走方案十七原参）。

```powershell
conda activate cyy
cd code/train_lab/src/step/5090_mask_future_dual_expert_accpaper
python chain_u_all.py
# 或双击 run_u_chain_detached.bat
```

断点续跑：

```powershell
python chain_u_all.py --from U3
# 或 run_u_chain_resume.bat U13
```

仅必做 U1/U3/U2（跳过组合）：

```powershell
python chain_u_all.py --skip-combo
```

状态与日志：`u_chain_state.json`、`u_chain_all_stdout.log`。

5060 低内存附报链（batch 128 + mem guard）仍用 `run_u_combo_chain_guarded.ps1`；**5090 全量请用 `chain_u_all.py`**。

## T 系列 v3（5090 · E_pos token · 无 Cross-Attn / 无 Phase）

顺序：**T1 → T1_128**（相对 P2；**不跑 T1_aux**；**不需要 t0_sec**）。

```powershell
python _smoke_local.py   # 含 T1 / T1_128
python run_arm.py --arm T1 --dry-run
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 1          # fold0 冒烟
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole # 正式五折
python chain_t_all.py --max-folds 0
```

状态与日志：`t_chain_guarded_state.json`、`t_chain_guarded.log`。

实验方案：`资料/Lejepa_shallow模型方案/.../实验方案/T系列_Token_PhasePredictor.md`（v3 在线契约见 README 与登记表 §7）

## 前置条件

1. A0：已有 `openbmi_2s_hop100` npy（含 Rest）  
2. A1+：在 `code/preprocess_lab` 跑  
   `python -m src.datasets.openbmi_pf1000.batch --reset`  
   （写出三类 `out/openbmi_2s_hop100_pf1000/`；**禁止改旧** `openbmi_2s_hop100` 源文件）  
3. `conda activate cyy`（或仓库 `.venv`）

## 同步

1. `git pull --rebase`  
2. 本包结果登记到 `资料/模型训练/17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/`  
3. `git push`
