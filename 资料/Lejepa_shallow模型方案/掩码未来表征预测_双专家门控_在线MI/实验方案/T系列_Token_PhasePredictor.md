# T 系列：Token + Future Query + Phase Predictor（v2.1 · 相对 P2）

> **基线对照**：**P2**（0.5707±0.0112 @5060）· **A1**（0.5717±0.0236）  
> **动机**：U1 只升级 Predictor 输入为序列，但 **L_pred 仍对齐 mean 向量、Expert 仍 mean-pool** → 仅 +0.15 pp。  
> **T1** 补齐：Future Query + **时间几何 Phase** + **token 级 L_pred** + **AttnPool 读出**。  
> **代码**：`5060`/`5070`/`5090_mask_future_dual_expert_accpaper` · 臂 `T1` / `T1_aux` / `T1_128`  
> **前置**：pf1000 须含 `openbmi_t0_sec.npy`（protocol_version≥3）  
> **契约**：与定稿方案 **v1.13 §1.3 / §17.4.1** 一致——**禁止 `y→phase`（尤其 Rest 专用桶）**

---

## 0. 与 P2 / U1 的关系

| 组件 | P2 | U1 | **T1（合法）** |
|------|----|----|----------------|
| Predictor 输入 | `mean(Z_vis)` | `Z_vis` 序列 | `Z_vis` tokens |
| Predictor 结构 | MLP | Attn+TCN→**单向量** | **Query + Cross-Attn → token 序列** |
| L_pred 目标 | `mean(Z_fut)` | 同 P2 | **`Z_fut` token 序列 MSE** |
| Phase 条件 | — | — | **`E_pos` + `E_phase(t0, 窗几何)` · 不用 `y`** |
| Expert_cur / future | mean-pool | mean-pool | **AttnPool** |
| Decoder / Gate / SIGReg | 同 P2 | 同 P2 | **同 P2**（波形 + λ_dec=0.2） |
| D | 40 | 40 | 40（T1_128→128） |

**不进主表条件**：五折 Test Acc_paper **稳定 > A1** 且 **std ≤ P2**，且 Phase **无 `y` 依赖**。

---

## 1. 在线契约（训练 / 推理一致）

| 项 | 训练 | 在线推理 |
|----|------|----------|
| 输入 EEG | `X_mask` + `X_full`（target） | **仅 `X_mask`** |
| Future Phase | **`t0_sec` + 窗几何查表** | **同左** |
| 类别 `y_three` | 仅 CE 监督 | **不得**进 Phase / Query / Gate |
| Future Query | `E_pos[i] + E_phase[i]` | 同结构 |
| Decoder | 仅训练 | 不跑 |
| Phase 辅助 CE | 仅 T1_aux · 训练 | 不跑 |

**禁止**：

1. 从 Future 波形估计 Phase（Future EEG 泄漏）。  
2. 用真实 `y_three` 查 Phase（**标签侧信道**；旧 `y==0→phase=3` 尤其严重）。

---

## 2. Phase 查表规则（合法版 · 须改 `phase_lookup.py`）

窗几何：past100 + cur500 + future400 @ 250 Hz；`t0_sec` = current 段起点相对 cue（合法 0.4～2.0 s）。

Future token 中心时间（相对 cue）：

```text
t_rel_cue = t0_sec + (sample_idx(fi) - 100) / 250
```

| Phase id | 条件（**与 y 无关**） |
|----------|----------------------|
| 0 onset | t_rel < 0.5 s |
| 1 sustain | 0.5 ≤ t_rel < 3.5 s |
| 2 offset | t_rel ≥ 3.5 s |

- **Rest / Left / Right 共用上表**；不再设 `phase=3 rest` 专用桶。  
- `N_PHASE = 3`（或保留 4 维嵌入但 **永不写入 id=3**，不推荐）。  
- API：`future_phase_ids(t0_sec, i_fut)` —— **签名去掉 `y`**。

### 2.1 为何旧规则泄漏（答辩一句）

旧规则：`y==0 → phase=3`，`y∈{1,2} → 0/1/2`。  
三分类本应只从 EEG 推断 Rest；把真实 Rest 标签映射成专用 `E_phase` 再喂网络 = **提前剧透「是否 Rest」**。  
Left/Right 虽共用时间相位、不泄左右，但 Rest 桶仍泄 **Rest vs Task**。

### 2.2 污染对照（可选 · 不进主表）

若需量化侧信道幅度，可另开臂 `T1_leak_y`（保留旧 `y→phase`），结果只进附表备注「泄漏上限」，**不得**与 A1/P2 比主结论。

---

## 3. 实验臂定义

### T1 · 主臂（必跑）

| 项 | 值 |
|----|-----|
| arm_id | `T1` |
| embed_dim | **40** |
| λ_pred | **1.0**（L1-λ 扫完后再回填最优） |
| λ_cls / λ_sig / λ_dec | 1.0 / 0.05 / 0.2（同 P2） |
| Phase | **仅时间几何**（§2） |

**验收**：

- §3.2.1 future 扰动单测仍过  
- Phase 单元测试：同 `t0_sec`、不同 `y` → **phase_ids 完全一致**  
- Test Acc_paper vs **A1**、**P2** Δ；fold1 不得系统性低于 A1  

### T1_aux · Phase 辅助 CE（T1 后可选）

| 项 | 值 |
|----|-----|
| arm_id | `T1_aux` |
| 相对 T1 | + `L_phase = CE(phase_head(Z_pre), phase_ids)`，λ_phase=0.2 |
| `phase_ids` 目标 | **时间几何 id**（§2），**不是** `y_three` |

### T1_128 · 容量对照（最后跑）

| 项 | 值 |
|----|-----|
| arm_id | `T1_128` |
| embed_dim | **128** |
| batch_train | 机位默认降档（5060/5070 常 64） |
| 目的 | 区分结构收益 vs 纯维数 |

---

## 4. 跑法（三机同构）

| 机位 | 包路径 | 默认 batch | 链脚本 |
|------|--------|------------|--------|
| **5090** | `5090_mask_future_dual_expert_accpaper/` | 256/512 | `run_t_chain_guarded.ps1` |
| **5070** | `5070_mask_future_dual_expert_accpaper/` | 128/256 | 同左 |
| 5060 | `5060_mask_future_dual_expert_accpaper/` | 128/256 | 同左（历史锚点） |

```powershell
# 先修 phase_lookup（去 y）并跑冒烟
cd code/train_lab/src/step/5070_mask_future_dual_expert_accpaper
python _smoke_local.py
python run_arm.py --arm T1 --max-folds 1 --num-workers 0

# 正式五折
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole
```

**算力最紧**：只跑 **T1 fold0 + T1 五折**（且必须已去 `y`）。

---

## 5. 主表 / 附表用法

| 用途 | 臂 |
|------|-----|
| v1.x 主表 | 仍 **P2** |
| v2 候选主表 | **T1**（合法 Phase，验收通过） |
| 消融 | T1 vs P2；T1_aux vs T1；T1_128 vs T1 |
| 污染对照 | `T1_leak_y`（可选） |
| 附报 | U1（半升级失败对照） |

---

## 6. 与定稿 §17 对齐

- **保留**：Mask、双专家、CE(cur+final)、SIGReg、波形 Decoder  
- **T1 只改**：Predictor 接口 + L_pred 目标 + Expert 读出 + **无 `y` 的** Phase  
- **不做**：Spectral-only（U2）、Gate 熵（U3）、U 组合堆叠、`y→rest 桶`

详见定稿方案 **v1.13 §17.4.1**。
