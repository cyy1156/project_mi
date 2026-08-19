# 5060_mask_future_dual_expert_accpaper

OpenBMI · Acc_paper · **掩码未来表征预测 + 双专家门控**（定稿方案 · **本机低内存旁路**）。  
**训练设备：NVIDIA RTX 5060 Laptop（~16GB RAM）**

| 项 | 路径 |
|----|------|
| 方案文档 | `资料/模型方案/掩码未来表征预测_双专家门控_在线MI/` |
| 数据切片说明 | 同目录 `数据切片与边界过滤说明.md` |
| 本包 | `code/train_lab/src/step/5060_mask_future_dual_expert_accpaper/` |
| **5090 全量姊妹包** | `../5090_mask_future_dual_expert_accpaper/` |
| 权重 out | `code/train_lab/out/5060_mask_future_dual_expert_accpaper/` |
| A0 数据 | `preprocess_lab/out/openbmi_2s_hop100/`（旧 500pt） |
| A1+ 数据 | `preprocess_lab/out/openbmi_2s_hop100_pf1000/` |

## 与 5090 包差异

| | **5060（本包）** | 5090 |
|--|------------------|------|
| batch | **128 / 256** | 256 / 512 |
| workers / pin | **0 / 关** | 4 / 开 |
| SIGReg slices | **1024** | 1024 |
| 默认 folds | **fold0**（`--max-folds 1`） | 五折 |
| 默认 chain | 主线 A0→…→P2 | 全 CHAIN（含 B/C） |
| 角色 | 本机冒烟 / 门控 | **大内存全量对照** |

方法结构、臂开关与定稿方案一致；仅工程超参降档。

## 臂一览

与 5090 相同：`A0_ref` / `A0` / `A1` / `P0` / `A2` / `P1` / `B*` / **`P2`** / `C*` / `U*` / `L1`。

## 推荐启动（5060）

```powershell
cd code/train_lab/src/step/5060_mask_future_dual_expert_accpaper

# 无数据冒烟（feat_index / 前向）
python _smoke_local.py

# 单臂 fold0
python run_arm.py --arm A0 --max-folds 1
python run_arm.py --arm P1 --max-folds 1

# 主线门控链（默认每臂 fold0）
python chain_all.py
# 或 run_chain_detached.bat

# 推荐：双层内存看门狗 + 主线全量（fold0）
powershell -File .\run_gate_chain_guarded.ps1
# 完整消融（含 B/C，仍 fold0）
powershell -File .\run_gate_chain_guarded.ps1 -FullChain

# U 单改五折（U1→U3→U2，已跑完可复现）
powershell -File .\run_u_chain_guarded.ps1 -FromArm U1 -MaxFolds 0 -NoConsole

# U 组合附报五折（U13→U12→U123；可 -SkipU123 只跑前两臂）
powershell -File .\run_u_combo_chain_guarded.ps1 -FromArm U13 -MaxFolds 0 -NoConsole
python run_arm.py --arm U13 --max-folds 0 --num-workers 0

# 无看门狗的完整消融链
python chain_all.py --full-chain
```

断点续跑：

```powershell
python chain_all.py --from P1
# 或 run_chain_resume.bat P1
```

## 前置条件

1. A0：已有 `openbmi_2s_hop100` npy  
2. A1+：`cd code/preprocess_lab` → `python -m src.datasets.openbmi_pf1000.batch`  
   （写出 `out/openbmi_2s_hop100_pf1000/`；**禁止改旧 preprocess**）  
3. 建议虚拟内存 / commit_limit 充足（本机方案 16 经验：D: pagefile 固定 48GB）

## 结果登记

`资料/模型训练/17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/`
