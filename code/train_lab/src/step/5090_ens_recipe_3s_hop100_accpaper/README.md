# 方案 26 · 5090 集成推理满配 + 训练配方升级

旁路包：`5090_ens_recipe_3s_hop100_accpaper`（不修改方案 24 历史 run）。

## 结构

| 模块 | 说明 |
|------|------|
| `s26_config.py` | 24 锚点 run、判定线、网格常量 |
| `e1_fusion_core.py` | 温度校准 / val 加权 / 邻域平滑 / 置信早停 |
| `replay_e1.py` | E1a–E1f 回放 CLI |
| `dump_member_probs.py` | 补 T-shallow 等 prob dump |
| `patch_recipe.py` + `recipe_train.py` | R1/R2/R3 训练配方注入 |
| `baseline_shallow_r{1,2}.py` | shallow 配方五折 |
| `baseline_conformer_r3.py` | conformer 配方五折 |
| `fe_bandpower_3s.py` / `fast_kan.py` | E2a 特征 + KAN |
| `fe_riemannian.py` | E2b 切空间特征 |
| `e2_common.py` + `baseline_kan_e2a.py` / `baseline_riemann_e2b.py` | E2 成员训练 + dump |
| `e2_fusion_gate.py` | fold0 融合门控 |

## 快速校验

```powershell
cd F:\Cyy\MI\code\train_lab\src\step\5090_ens_recipe_3s_hop100_accpaper
python verify_imports.py
python smoke_s26_test.py
```

## 5090 全链

```powershell
powershell -File .\run_all_26_5090.ps1
powershell -File .\run_all_26_5090.ps1 -From e1 -SkipR2
powershell -File .\run_26_watch_progress.ps1
```

## 单臂示例

```powershell
# 阶段 0：补 T-shallow dump
python dump_member_probs.py --run-dir "...\run_20260823_123900\three"

# E1 推理满配
python replay_e1.py --arm E1d
python replay_e1.py --arm E1e --four-member

# R1/R2/R3 训练
python baseline_shallow_r1.py --skip-task --three-only
python baseline_shallow_r2.py --skip-task --three-only
python baseline_conformer_r3.py --skip-task --three-only

# E2 异构成员
python baseline_riemann_e2b.py
python baseline_kan_e2a.py --max-folds 1   # fold0 门控
python e2_fusion_gate.py --arm E2a --candidate-run "...\three"
python baseline_kan_e2a.py                 # 过门后全五折
```

## 输出目录

`code/train_lab/out/5090_ens_recipe_3s_hop100_accpaper/`

E1 回放 JSON 默认写在包目录 `replay_e1*.json`；训练 run 按 `run_YYYYMMDD_HHMMSS/three/` 组织。

## 锚点（只读）

见 `s26_config.DEFAULT_MEMBERS`：shallow / T-shallow / eegnet / conformer 四个 24 正式 run。

## 判定线（预注册）

- E1 test Three ≥ 0.6008（+0.5pp vs E 均匀 0.5958）
- R1 Three ≥ 0.5889（+0.5pp vs S3 0.5839）
- E2 fold0 融合 Δ ≥ +0.3pp → 进成员池

详细方案：`资料/模型训练/26_旁路_集成推理满配与训练配方升级_openbmi_accpaper/方案.md`
