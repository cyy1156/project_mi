# 21 系列：LeJEPA 对齐 · 表征预测修正（相对方案十七 · 优先级二）

> **上级框架**：[基于掩码未来表征预测与双专家门控融合的在线MI分类定稿方案.md](../基于掩码未来表征预测与双专家门控融合的在线MI分类定稿方案.md)  
> **前置完成**：方案十七（A→P→B→C→U→T v3 五折 · 5090）  
> **基线对照（只读）**：**A1**（0.5754±0.021）· **A2**（0.5667±0.020）· **P2**（0.5703±0.022）· **B9 oracle**（0.5788±0.019）  
> **动机**：方案十七表明 (i) 主增益在 pf1000 可见上下文（A1），(ii) 同训 `L_pred` 未改善单专家推理（A2&lt;A1），(iii) far future 与当前窗标签存在 **时间语义错位**（含 post-MI / late-MI）。本系列 **不新增 Gate/Decoder/U/T 模块**，仅修正 **预测目标 · 数据过滤 · 训练阶段**，检验 LeJEPA 式辅助任务能否 **真正提升测试期 `z_vis` 分类**。

---

## 0. 系列定位

| 项 | 方案十七 | **方案二十一（本系列）** |
|----|----------|-------------------------|
| 主问题 | 完整训练图 + 结构升级是否超 A1 | **Predictor 能否通过对齐的 JEPA 目标改善 Encoder** |
| 推理路径 | P2 双专家 / A1 单专家 | **统一冻结：仅 `p_cur`（A2 式单专家）** |
| 模块堆叠 | U/T/Decoder 等 | **禁止**（本系列不跑） |
| 数据 | `openbmi_2s_hop100_pf1000` v3 | v3 + **可选 v4 过滤臂**（F_mi） |
| 进主表条件 | P2 定稿 | **仅当某 21 臂五折稳定 ≥ A1 且 std ≤ A1** |

**代码包**：`5090_mask_future_dual_expert_accpaper/` · `train_21_kfold.py` · `scheme21_data.py` · `inwin_jepa.py` · `chain_21_all.py`

---

## 1. 共有在线契约（三臂强制一致）

| 项 | 训练 | 在线推理 |
|----|------|----------|
| 输入 | `X_mask`（+ 训练用 `X_full`） | **仅 `X_mask`** |
| Encoder | 自写 Shallow · `forward_features` | 开 |
| Target 支路 | `X_full` · **no_grad / sg** | 关 |
| Predictor | 按臂定义 | 可前向，**不参与分类决策** |
| 分类 | **仅 `Expert_cur(z_vis)` → `p_cur`** | **仅 `p_cur`** |
| Expert_future / Gate / Decoder | **关** | **关** |
| SIGReg | 按臂（A2_pt 阶段 1 开；F_mi/J1 可选同 A2） | 关 |
| 标签 `y` | 绑定 **当前窗 `X_cur`** | 同左 |
| Predictor 输入 | **禁止 `y` / `t0_sec` 查表**（防泄漏，同 T v3 验收） |

**泄漏验收（每臂 fold0 必过）**：

1. shuffle `y` → Predictor 输出不变；  
2. §3.2.1 future-perturb 单测：仅扰动 target 对应区间时 \(\|\Delta Z_{tgt}\|\) 显著；  
3. 推理脚本不读取 `X_full` / `t0_sec`（除非 logging）。

---

## 2. 共有训练超参（除非臂内注明）

| 项 | 值 |
|----|-----|
| 任务 | **仅 Three**（Rest/L/R） |
| 优化器 | Adam · lr=1e-4 · wd=1e-4 |
| batch | train **256** / eval **512**（5090）；5060/5070 可 128/256 |
| patience | **20**（Acc_paper Val 早停） |
| D | **40** |
| Expert_cur | **D→64→C** |
| Predictor（MLP 臂） | D→2D→D · dropout 0.3（同 P0） |
| λ_sig | **0.05**（开 SIGReg 时） |
| fold | 被试 5-fold · 与方案十七相同划分 |
| 主指标 | **Test Acc_paper** mean±std |

---

## 3. 实验臂总览

| 臂 ID | 名称 | 改什么 | 数据臂 | 优先级 |
|-------|------|--------|--------|--------|
| **F_mi_a** | MI 内锚点过滤（软对齐） | 锚点 `t0≤1.0 s` · 仍 1000pt/400 future | pf1000 **v3 重滤** | **1 · 必做** |
| **F_mi_080** | MI 内 future（硬对齐） | `T_future=0.8s` · `n_times=800` | pf1000 **v4_mi080** | **2 · 必做** |
| **A2_pt** | 两阶段预训练→微调 | 阶段1 无 CE · 阶段2 仅 CE | pf1000 v3 | **3 · 必做** |
| **J1** | 同窗块掩码 JEPA | 仅在 past+cur 内随机遮块 | pf1000 v3 | **4 · 必做** |

**执行链（5090）**：

```text
F_mi_a fold0  →  F_mi_a 五折（若 fold0 过线）
             →  F_mi_080 fold0  →  F_mi_080 五折（若 fold0 过线）
             →  A2_pt fold0  →  A2_pt 五折（若 fold0 过线）
             →  J1 fold0  →  J1 五折（若 fold0 过线）
```

**不算力档**：只跑 **F_mi_a fold0 + A2_pt fold0**（覆盖「改数据 / 改训练」两条假说）。

---

## 4. 臂 F_mi_a · MI 内锚点过滤（软对齐）

### 4.1 目的

在不改 `n_times=1000` 的前提下，**减少 Future 中 post-MI 占比**，检验「时间语义错位」是否是 A2 失败主因。

### 4.2 数据过滤（相对 pf1000 v3）

| 项 | pf1000 v3（方案十七） | **F_mi_a** |
|----|----------------------|------------|
| 几何 | past100 + cur500 + future400 | **同左** |
| 合法锚点 t0 | {0.4, …, 2.0} | **{0.4, …, 1.0}** |
| 约束理由 | — | t0=1.0 时 future=[3.0,4.6)，post-MI 占比约 **38%**；t0&gt;1.0 更高 |
| 每 trial 窗数 | 最多 17 | **最多 7** |
| 标签 | 仍绑 `X_cur` | 同左 |

**说明**：Future **仍可能含少量 post-MI**（t0≥1.0），但 **严格剔除 t0∈(1.0,2.0]**（该段 future 100% 或 majority post-MI）。实现为 **预处理 meta 重滤** 或 **DataLoader 索引表**；须记录 `n_samples_v3 → n_samples_f_mi_a`。

**附报（可选）**：`F_mi_a_strict` 仅 t0=0.4（future 全在 MI 内）— 样本极少，**只报 fold0 样本数与 Acc，不进主链**。

### 4.3 模型与损失（同 A2）

| 模块 | 状态 |
|------|------|
| Predictor | MLP：`z_vis → z_pre` |
| L_pred | MSE(`z_pre`, sg(`z_target_future`)) · 目标仍 **I_future 池化** |
| L_cls | CE(`p_cur`, y) |
| λ_pred / λ_cls | **1.0 / 1.0** |
| SIGReg | **开** · λ_sig=0.05 · 作用于 `z_vis` |

### 4.4 验收

| 门槛 | 条件 |
|------|------|
| fold0 继续 | Test Acc_paper **≥ A2 fold0 + 0.5 pp** 且 **≥ A1 fold0 − 0.3 pp** |
| 五折主报 | mean **≥ A1** 且 std **≤ A1 std × 1.1** |
| 失败 | 记阴性 · **仍跑 A2_pt**（不因 F 失败停整条链） |

### 4.5 预期

- 若 **F_mi_a &gt; A2** 但 **&lt; A1**：支持「错位」假说，**F_mi_080 更值得五折**；  
- 若 **≈ A2**：错位非主因，重点转 **A2_pt / J1**。

---

## 5. 臂 F_mi_080 · MI 内 future 硬对齐（新几何）

### 5.1 目的

**Future 100% 落在 MI 评分区 [0,4) s 内**，消除 post-MI 污染；为此 **缩短 future 并重定义 `n_times`**。

### 5.2 新数据几何 pf800_mi080（实现：运行时裁切）

| 符号 | 值 |
|------|-----|
| T_past | 0.4 s / 100 pt |
| T_cur | 2.0 s / 500 pt |
| **T_future** | **0.8 s / 200 pt** |
| **T_total** | **3.2 s / 800 pt** |
| 锚点 t0 | **t0 ≤ 1.2 s**（须 `openbmi_t0_sec.npy`） |

**实现说明（5090 代码）**：不另开预处理目录；从 pf1000 v3 的 1000pt `X_full` **裁切前 800pt**（`scheme21_data.crop_pf_mi080`），Encoder **`n_times=800`** 独立建模。

**张量**：

```text
X_full = concat(past, cur, future)   # (8, 800)
X_mask = concat(past, cur, zeros(8,200))
```

**§3.2.1 映射**：须 **重标** `feat_index` 与 `I_vis` / `I_future` 边界（`n_times=800` → 新 T′）；**单测门槛同方案十七**（future 扰动比 ≥3）。

### 5.3 模型

| 项 | 值 |
|----|-----|
| Encoder | ShallowFBCSPNet **`n_times=800`**（与 1000pt **分模型**，禁止混载权重） |
| 其余 | 同 F_mi_a |

### 5.4 验收

| 门槛 | 条件 |
|------|------|
| fold0 继续 | Test Acc **≥ F_mi_a fold0** 且 **≥ A1 fold0 − 0.3 pp** |
| 五折 | mean **≥ A1** |
| 与 v3 A1 对比 | 附报 **Δ 样本数 / Δ Acc**，讨论短 future 几何本身的影响 |

### 5.5 风险

- 锚点仍少于 v3（9 vs 17 / trial）；  
- `n_times` 变化使 **与 A1 的 Δ 含几何混杂** → 正文须同时报 **同过滤样本上的 A1_800 对照**（可选臂 **`A1_800`**：无 Predictor，仅 CE，同 pf800_mi080）。

---

## 6. 臂 A2_pt · 两阶段预训练 → 分类微调

### 6.1 目的

检验 **LeJEPA 式「先表征、后分类」** 是否优于方案十七 **CE + L_pred 同训**（A2 阴性）。

### 6.2 数据

- **pf1000 v3 全锚点**（与方案十七相同，不额外过滤）；  
- 可选 **附报**：在 **F_mi_a 索引** 上再跑一遍 A2_pt（臂 **`A2_pt_fmi`**，仅当 F_mi_a fold0 为正时）。

### 6.3 阶段定义

#### 阶段 1 · 表征预训练（`A2_pt_s1`）

| 项 | 值 |
|----|-----|
| epoch 上限 | **30**（或 Val 代理指标早停，见下） |
| 输入 | `X_mask` + `X_full` |
| 损失 | **λ_pred·L_pred + λ_sig·L_SIGReg** |
| λ_pred / λ_sig | **1.0 / 0.05** |
| CE / Expert 分类 | **关**（Expert_cur 不更新或不存在） |
| 早停 | **L_pred Val 均值连续 5 epoch 不降** 或达 30 epoch |
| checkpoint | 存 **`encoder` + `predictor`** 权重 |

#### 阶段 2 · 分类微调（`A2_pt_s2`）

| 项 | 值 |
|----|-----|
| 初始化 | 加载阶段 1 的 Encoder（+ 可选 Predictor 冻结） |
| 损失 | **仅 CE(`p_cur`, y)** |
| Predictor | **冻结**（默认）或 lr×0.1 联合微调（**附报 ablation：`A2_pt_s2_unfreeze_pred`**） |
| Encoder lr | **1e-4**（默认）· 附报 **`A2_pt_s2_enc_low`**：encoder lr=**1e-5** |
| epoch | 同标准 **patience=20** Acc_paper 早停 |
| L_pred | **关** |

**推理**：阶段 2 最优 ckpt · **仅 `p_cur`**。

### 6.4 监控（阶段 1 必 log）

| 量 | 用途 |
|----|------|
| L_pred train/val | 预训练是否收敛 |
| L_SIGReg | 是否坍塌 |
| **线性 probe Acc**（每 5 epoch） | 冻结 Encoder · 在 Val 上训 **线性层** 5 epoch · 观察表征是否可分 |
| z_vis 范数 / 有效秩（可选） | 表征质量 |

### 6.5 验收

| 门槛 | 条件 |
|------|------|
| fold0 | Test Acc **≥ A1 fold0 + 0.5 pp** |
| 五折 | mean **≥ A1** 且 **≥ A2 + 0.5 pp** |
| 失败解读 | 若 L_pred↓ 但 probe↑、Test 仍 &lt;A1 → **表征-分类迁移失败**；若 L_pred 不降 → **目标仍不对齐** |

---

## 7. 臂 J1 · 同窗块掩码 JEPA（in-window）

### 7.1 目的

在 **past+cur（可见 600pt / I_vis）内部** 做随机块掩码，预测 **同时间语义内** 被遮位置的表征，最接近 LeJEPA/JEPA 原意。

### 7.2 数据

- **pf1000 v3**；`X_full` 与 `X_mask` 外几何不变；  
- **Future 400pt 在 J1 中不参与 mask 游戏**（可仍零填于 `X_mask` 以保持 `n_times=1000`，或附报 **`J1_600`** 真 600pt 输入）。

### 7.3 掩码规则（冻结）

| 项 | 规约 |
|----|------|
| 掩码域 | 仅 **raw 时间** past+cur：点索引 **[0, 600)** |
| 块大小 | **50 pt（0.2 s）** × **K=4 块** / 样本（非重叠优先） |
| 选取 | 每样本 **随机** K 块（训练）· **固定种子 per-fold**（Val/Test 可固定或随机，须 meta 记录） |
| X_mask 构造 | 被遮块 **置零**（与 future 零填一致） |
| 目标 | `z_target = sg(mean(Z_full[对应 I_vis 被遮块 token]))` 或 **token 级 MSE**（推荐 **token 级**，复用 T1 v3 的 `E_pos` Predictor **但无 Phase**） |

### 7.4 模型（推荐配置 · `J1_tok`）

| 组件 | 配置 |
|------|------|
| Encoder | Shallow · `n_times=1000` |
| 可见读出 | `H_vis` tokens（I_vis 段） |
| Predictor | **PosTokenFuturePredictor**（同 T1 v3 · **无 Cross-Attn / Phase**） |
| 预测目标 | 被遮块的 **token 序列** MSE |
| L_cls | CE(`p_cur`, y) · **λ_cls=1** |
| L_pred | **λ_pred=1** |
| SIGReg | λ_sig=0.05 · 对 **未遮可见 token 的 pool(z_vis)** |
| Expert_future / Gate / Decoder | **关** |

**附报臂 `J1_mlp`**：仅 pool 级单向量 MSE（与 P0 同构 Predictor）· 用于对比 token 级是否必要。

### 7.5 在线推理

- 输入 **标准 `X_mask`**（future 仍零填；**不做 random mask**）；  
- **仅 `p_cur`**；Predictor 不参与决策。

### 7.6 验收

| 门槛 | 条件 |
|------|------|
| fold0 | Test Acc **≥ A1 fold0 + 0.5 pp** |
| 五折 | mean **≥ A1** |
| 表征 | 线性 probe **≥ A2_pt 同阶段**（附表） |

---

## 8. 对照臂（建议同批登记）

| 臂 | 用途 |
|----|------|
| **A1**（只读） | 方案十七 `20260817_171731_A1` |
| **A2**（只读） | 方案十七 · 同训 L_pred 阴性 |
| **A1_800**（可选） | pf800_mi080 · 无 Predictor · 分离几何效应 |
| **A2_pt_fmi**（可选） | F_mi_a 索引上的两阶段 |

---

## 9. 成功线汇总

| 级别 | 条件 |
|------|------|
| **fold0 继续五折** | Test Acc ≥ **A1 fold0 − 0.3 pp** 且 ≥ **A2 fold0 + 0.5 pp** |
| **弱有用** | 五折 mean ≥ **A1 + 0.5 pp** |
| **进主表候选** | 五折 mean **≥ A1** 且 std **≤ 0.022** · 泄漏验收全过 |
| **系列阴性结案** | F_mi_a / F_mi_080 / A2_pt / J1 **均未达 fold0 继续线** → 主报 **A1 在线** + 本系列 Discussion |

---

## 10. 执行顺序与资源（5090 估算）

| 顺序 | 臂 | fold0 | 五折 | 备注 |
|------|-----|-------|------|------|
| 1 | F_mi_a | ~25 min | ~2 h | 仅重滤索引 · 无新 preprocess |
| 2 | F_mi_080 + A1_800 | ~30 min | ~2.5 h | 运行时裁切 800pt · 无新 preprocess |
| 3 | A2_pt | ~40 min | ~3 h | 两阶段 · 计 s1+s2 |
| 4 | J1_tok | ~30 min | ~2.5 h | token 预测 |
| 附 | J1_mlp | 可选 | 可选 | fold0 即可 |

**guarded 长跑**：同方案十七 · conda `cyy` 在 PATH · 链脚本 **`chain_21_all.py`**（规划名）。

---

## 11. 结果记录（已定稿）

详见 `资料/模型训练/21_5090_旁路_LeJEPA对齐_表征预测修正_openbmi_accpaper/结果登记表.md`。

| 臂 | Test Acc_paper | BalAcc_maj | vs A1 | vs A2 | 结论 | run |
|----|----------------|------------|-------|-------|------|-----|
| **F_mi_a** | **0.5765±0.027** | 0.5773 | **+0.11 pp** | +0.98 pp | 系列最佳 · 未达弱有用线 | `20260821_200453` |
| J1_tok | 0.5733±0.022 | 0.5782 | −0.02 pp | +0.66 pp | std 最低 · < A1 | `20260822_004001` |
| A2_pt | 0.5701±0.025 | 0.5753 | −0.05 pp | +0.34 pp | 两阶段未超 A1 | `20260821_225915` |
| F_mi_080 | 0.5684±0.026 | 0.5701 | −0.07 pp | +0.17 pp | 硬对齐 < F_mi_a | `20260821_222149` |
| A1_800 | — | — | — | — | 未跑 | — |
| J1_mlp | — | — | — | — | 未跑 | — |

**系列总判（2026-08-22）**：阴性结案 → **A1 在线**。

**必附**：

- 各臂相对 v3 的 **样本数变化**；  
- A2_pt **阶段 1** 的 L_pred 曲线 + **线性 probe** 表；  
- J1 的 **mask 块数 K** ablation（附报 K=2,4,6 仅 fold0）。

---

## 12. 论文 Methods 表述要点

1. **问题**：far-future 表征预测与在线 MI 分类标签 **时间不对齐**；  
2. **改法 F**：限制 future 于 MI 评分区（软/硬两档）；  
3. **改法 A2_pt**：分阶段 LeJEPA 式训练，避免 CE 与 L_pred 梯度冲突；  
4. **改法 J1**：同窗块掩码 JEPA，预测目标与可见段 **同 trial 同时序**；  
5. **推理统一**：仅 `X_mask` → Expert_cur，**不增加在线复杂度**；  
6. **与方案十七关系**：P2/U/T 作 **结构升级阴性**；本系列作 **目标对齐修正**。

---

## 13. 与 00_索引 的关系

本文件为 **方案二十一** 唯一主实验文档；实现完成后在 [`00_索引.md`](./00_索引.md) 增加一行 **21 | 本文件 | 必做（post-17）**。
