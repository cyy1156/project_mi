# U 系列：结构升级（相对 P2 定稿 · 后续新增）

> **立场**：不修改、不替代现有 A0–P2 / B / C / L1 冻结方案。  
> 本系列排在 **当前实验计划全部完成后**（至少：主线五折出数 + B/C/L1 按原优先级推进）。  
> 基线对照：**P2 训练定稿**（波形 Decoder + MLP Predictor + Gate 仅两 z）。  
> 动机来源：外部评审建议「保留 v1.x 主框架，只升级 3 处」——见文末对照。  
> **代码已落地**（`5060_` / `5090_mask_future_dual_expert_accpaper`）：臂 `U1`/`U2`/`U3`/`U12`/`U13`/`U123`，`skip_in_auto_chain=True`；跑法同 P2 单臂。

---

## 0. 与现稿的关系

| 现稿（保持不变） | U 系列 |
|------------------|--------|
| Encoder = ShallowFBCSP · D=40 | **保留** |
| Mask / `X_full` no_grad 目标 / SIGReg(z_vis) | **保留** |
| 双专家 + `p=α p_cur+(1-α) p_future` | **保留** |
| 协议 Acc_paper · 仅 Three · Adam | **同协议** |
| Predictor = MLP(D→2D→D)，输入为 `mean(Z[I_vis])` | **U1 升级** |
| Decoder = Linear→8×400→PSD/μβ/time | **U2 对照**（不改写 P2 主表定义） |
| Gate = `[z_vis, z_pre]`；L1 已有 z+p 变体 | **U3 扩展**（加熵，并进扫描） |

执行顺序建议：

```text
（现计划）A0→…→P2→B→C→L1
        ↓
（新增）U1 → U3（可与 L1 Gate 扫描并行设计）→ U2
```

主报仍用 **Test Acc_paper** 五折 mean±std；附报 vs **同一数据协议下的 P2** Δ。

---

## U1 · 时间维 Predictor（必做 · 优先）

### 1. 目的

检验：在 **不换 Encoder** 的前提下，让 Predictor 吃 `Z_vis` **序列**（而非立刻 mean 后的 40 维向量），是否提升未来表征预测与 Acc_paper。

### 2. 相对 P2 唯一变更点

| 项 | P2（冻结） | U1 |
|----|------------|-----|
| `forward_features` → `Z (B,D,T')` | 有 | 同 |
| 读出 | `z_vis = mean(Z[I_vis])` 后进 Predictor / Expert_cur | Predictor 输入改为 `Z_vis = Z[:,:,I_vis]`；Expert_cur 可用 AttentionPool(`Z_vis`) 或仍用 mean（实现时二选一并写死） |
| Predictor | MLP `D→2D→D` | **轻量 Temporal**：1×1 Conv → Temporal Attention（或 1–2 层）→ Dilated TCN → pool → `z_pre (B,D)` |
| Target | `mean(Z_full[I_fut])` | **默认仍 mean 成 (B,D)**，保持 `L_pred` 与 P2 同形；附报可做序列对齐变体（不进主表） |
| Decoder / Gate / SIGReg / 损失权重 | 同 P2 | **同 P2** |

### 3. 优先级

**必做（结构升级主臂）**；仅 Three；五折 Acc_paper。

### 4. 验收

- §3.2.1 索引单测仍通过（扰动 future 主要动 `I_fut`）
- 推理仍只喂 `X_mask`，无 Decoder
- 参数量：Predictor 应明显小于 Encoder（量级写进 meta）
- 主报：Test Acc_paper vs P2 Δ

### 5. 结果（跑后填）

| 指标 | 值 |
|------|-----|
| Acc_paper mean±std | **0.5722±0.0180** |
| vs P2 Δ | **+0.15 pp**（P2=0.5707±0.0112） |
| run 路径 | `5060_.../20260818_194132_U1` |
| Predictor 结构备注 | TemporalAttention + Dilated TCN；Expert_cur 仍 mean-pool |

---

## U2 · Spectral Decoder 对照（必做 · 排在 U1 后）

### 1. 目的

检验：生理约束是否必须「先重建 8×400 波形再 PSD」，还是 **直接** `z_pre → μ/β（及可选 PSD 向量）` 更干净且不伤 Acc。

### 2. 相对 P2 唯一变更点

| 项 | P2（冻结 · 主表） | U2 |
|----|-------------------|-----|
| Decoder | `Linear(D→8×400)` → `X̂` → PSD/μβ/time | **Spectral head**：`z_pre →` 预测 μ/β（及可选粗 PSD bins）；**默认无**全波形重建 |
| `L_dec` | 现稿 1:1:1:0.1（含 time） | `L_spec`（μ/β 为主）；可选极小权重波形项作附报 |
| Predictor / Gate / SIGReg | 同 P2 | **默认同 P2**；若 U1 已采纳，可再开 **U1+U2** 组合臂（记 U12，附报） |

### 3. 与 C 系列关系

- **C1–C2c** 仍按原文档做（相对波形 Decoder 的 P2）
- U2 **不替代** C，而是另开「解码接口」对照；论文叙述：P2=训练图对齐主结果；U2=表征空间生理约束变体

### 4. 优先级

**必做对照**；建议在 U1 五折出数后再跑（避免同时改两处）。

### 5. 结果（跑后填）

| 指标 | 值 |
|------|-----|
| Acc_paper mean±std | **0.5665±0.0236** |
| vs P2 Δ | **−0.42 pp** |
| vs C1 Δ（若已跑） | C1 未五折 |
| run 路径 | `5060_.../20260819_034744_U2` |

---

## U3 · Gate + 专家熵（必做 · 可与 L1 合并登记）

### 1. 目的

检验：Gate 在两表征之外，加入 `H(p_cur)`、`H(p_future)` 是否改善动态权衡（当前自信 → α↑）。

### 2. 相对 P2 / L1 的关系

| 项 | P2 冻结 | L1 已写变体 | U3 |
|----|---------|-------------|-----|
| Gate 输入 | `[z_vis, z_pre]` | `[z_vis, z_pre, p_cur, p_future]` | `[z_vis, z_pre, H(p_cur), H(p_future)]` |
| 其它 | — | — | **同 P2**（或同 U1，若已升主臂） |

说明：U3 与 L1「Gate z+p」同属门控输入扫描；U3 用 **标量熵**，解释更直观。登记时可与 L1 Gate 表合并，但 **实验 ID 仍记 U3**。

### 3. 优先级

**必做**；计算量小，可在 P2 权重协议固定后优先扫（甚至可先于 U1 做小网格，但主文结构故事仍以 U1 为先）。

### 4. 结果（跑后填）

| 指标 | 值 |
|------|-----|
| Acc_paper mean±std | **0.5708±0.0180** |
| vs P2 Δ | **+0.01 pp**（基本持平） |
| vs Gate(z+p) Δ（若已跑） | L1 未跑 |
| run 路径 | `5060_.../20260818_235303_U3` |

---

## 组合臂（可选 · 附报）

> **详细实验计划（门槛 / 开关 / 验收 / 结果表）**：  
> [**U组合_U12_U13_U123.md**](./U组合_U12_U13_U123.md)

| ID | 组合 | 目的 | 优先级（据单改已出数） |
|----|------|------|------------------------|
| **U13** | U1 + U3 | 时间维 Predictor + 熵 Gate（**保留波形 Decoder**） | **建议附报** |
| U12 | U1 + U2 | 时间维 Predictor + Spectral Decoder | 可选 · 低优先 |
| U123 | U1+U2+U3 | 融合版完整形态 | **门槛制 · 默认跳过** |

原则：**一次主文只强调一条主升级路径**；组合臂防堆模块叙事。单改无强增益时，优先只跑 U13。

执行顺序（若开跑）：`U13 → U12 →（门槛）U123`。

---

## 明确不做（本系列）

- 换 Multi-Scale TCN / Patch Transformer 作主 Encoder  
- 取消 SIGReg 或改成频域 SIGReg  
- 取消 mask 训练 / 在线统一  
- 宣称完整复现 LeJEPA  
- 改写 P2 为「唯一主结果」的定义（P2 仍对齐训练图；U 为后续升级）

---

## 外部建议 ↔ 本系列映射

| 外部建议 | 本系列 |
|----------|--------|
| 保留 Shallow + mask + 双专家主框架 | 全文前提 |
| ① 保留时间维，升级 Predictor | **U1** |
| ② Spectral Decoder，减弱波形重建主任务 | **U2** |
| ③ Gate 加入不确定性（熵） | **U3** |
