# T 系列：E_pos Token Predictor（v3 · 相对 P2）

> **基线对照**：**P2**（0.5703±0.022 @5090）· **A1**（0.5754±0.021）  
> **动机**：v2（Cross-Attn + 语义 Phase 查表）五折 ~0.79，核查为 **标签泄漏**（`E_phase` 依赖 `y`/`t0_sec`）；v3 去掉 Cross-Attn / Phase，仅保留 **位置 token + 可见段上下文**。  
> **代码**：`*_mask_future_dual_expert_accpaper` · 臂 **`T1` / `T1_128`**（链上 **不跑 `T1_aux`**）  
> **前置**：pf1000 三类窗（`protocol_version≥3`）；**v3 不需要 `openbmi_t0_sec.npy`**

---

## 0. 版本沿革

| 版本 | Predictor | Phase | 链 | 状态 |
|------|-----------|-------|-----|------|
| **v2** | Query + **Cross-Attn** | **E_phase(y,t0 查表)** | T1→T1_aux→T1_128 | **作废**（~0.79 泄漏） |
| **v3** | **E_pos + pool(H_vis)** | **无** | **T1→T1_128** | **当前** |

---

## 1. 与 P2 / U1 的关系

| 组件 | P2 | U1 | **T1 v3** |
|------|----|----|-----------|
| Predictor 输入 | `mean(Z_vis)` | `Z_vis` 序列 | `Z_vis` tokens |
| Predictor 结构 | MLP | Attn+TCN→**单向量** | **E_pos(j)+ctx → token 序列**（无 Cross-Attn） |
| L_pred 目标 | `mean(Z_fut)` | 同 P2 | **`Z_fut` token 序列 MSE** |
| Phase / t0 / y | — | — | **Predictor 不用** |
| Expert_cur / future | mean-pool | mean-pool | **AttnPool** |
| Decoder / Gate / SIGReg | 同 P2 | 同 P2 | **同 P2**（波形 + λ_dec=0.2） |
| D | 40 | 40 | 40（T1_128→128） |

**预期**：Test Acc_paper ≈ **A1/P2 量级（~0.55–0.58）**；若仍 >>0.65 须再查泄漏。

**不进主表条件**：五折 Test Acc_paper **稳定 > A1** 且 **std ≤ P2** 方可宣称 v2 结构增益。

---

## 2. 在线契约（训练 / 推理一致）

| 项 | 训练 | 在线推理 |
|----|------|----------|
| 输入 | `X_mask` + `X_full`（target） | **仅 `X_mask`** |
| Predictor | `PosTokenFuturePredictor(H_vis)` | 同左 |
| Query 构造 | `E_pos[j] + MLP(mean(H_vis))` | 同结构 |
| 标签 y / t0 | **不进 Predictor forward** | **不用** |
| Decoder | 仅训练 | 不跑 |
| Phase aux CE | **不跑**（`T1_aux` 仅遗留注册） | 不跑 |

**验收（泄漏）**：
- shuffle `y` → `Z_pre` / loss 路径不变（Predictor 不读 y）
- §3.2.1 future-perturb 单测仍过

---

## 3. Predictor 结构（`PosTokenFuturePredictor`）

```text
H_vis: (B, L_vis, D)   ← 可见段 token（encoder 特征）
ctx  = MLP(mean(H_vis))           # (B, D)
Z_pre[j] = LayerNorm( E_pos[j] + ctx )   # j=0..L_fut-1
```

- **无** MultiheadAttention / Cross-Attn  
- **无** `E_phase`、`phase_lookup`、辅助 CE  
- 读出：`AttnPool(Z_pre)` → Expert_future / Gate（同 P2 余下管线）

---

## 4. 实验臂定义

### T1 · 主臂（必跑）

| 项 | 值 |
|----|-----|
| arm_id | `T1` |
| 开关 | `predictor_pos_token=True` · `pred_token_seq=True` · `expert_attn_pool=True` |
| embed_dim | **40** |
| batch_train | **256**（5090）/ 128（5060·5070） |
| λ_pred / λ_cls / λ_sig / λ_dec | 1.0 / 1.0 / 0.05 / 0.2 |

### T1_128 · 容量对照（T1 后）

| 项 | 值 |
|----|-----|
| arm_id | `T1_128` |
| embed_dim | **128** |
| batch_train | **256**（与 T1 同；5090 正式 run `20260821_152002_T1_128`） |

### T1_aux · 【作废 · 链上不跑】

v2 遗留臂（Cross-Attn + Phase CE + 语义查表）。**禁止**用于主表；仅注册表保留供对照说明。

---

## 5. 跑法

### 5.1 冒烟

```powershell
cd code/train_lab/src/step/5090_mask_future_dual_expert_accpaper
python _smoke_local.py
python run_arm.py --arm T1 --dry-run
python chain_t_all.py --max-folds 1
```

### 5.2 正式五折（5090 推荐）

```powershell
conda activate cyy
# 直连（须 cyy 在 PATH）
python chain_t_all.py --max-folds 0

# 或 guarded（长跑 · 独立进程）
$env:Path = "$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;" + $env:Path
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole

# 断点：T1 已完成，只补 T1_128
powershell -File .\run_t_chain_guarded.ps1 -FromArm T1_128 -MaxFolds 0 -NoConsole
```

链顺序：**T1 → T1_128**（`T_SERIES_ORDER`）。

### 5.3 三机包

| 机位 | 包 | 默认 batch | out |
|------|-----|------------|-----|
| **5090** | `5090_mask_future_dual_expert_accpaper/` | 256/512 | `out/5090_.../` |
| **5070** | `5070_mask_future_dual_expert_accpaper/` | 128/256 | `out/5070_.../` |
| **5060** | `5060_mask_future_dual_expert_accpaper/` | 128/256 | `out/5060_.../` |

5090 v3 代码已落地；5060/5070 待同步 port（若需三机同构再 cherry-pick）。

---

## 6. 执行顺序

```text
T1 fold0 冒烟  →  T1 五折  →  T1_128 五折
```

**不算力**：只跑 **T1 五折**。

---

## 7. 主表 / 附表

| 用途 | 臂 |
|------|-----|
| v1.x 主表 | 仍 **P2** |
| v3 候选 | **T1**（验收通过后） |
| 消融 | T1 vs P2；T1_128 vs T1 |
| 附报 | U1；**v2 T* 作废** |

---

## 8. 结果记录（跑后填 · 本提交不含数值）

见 `资料/模型训练/17_5090_.../结果登记表.md` §7。

---

## 9. 与定稿 §17 对齐

- **保留**：Mask、双专家、CE(cur+final)、SIGReg、波形 Decoder  
- **T1 v3 只改**：Predictor（pos-token）+ token L_pred + AttnPool 读出  
- **不做**：Cross-Attn Query、语义 Phase、T1_aux、Spectral-only（U2）、Gate 熵（U3）
