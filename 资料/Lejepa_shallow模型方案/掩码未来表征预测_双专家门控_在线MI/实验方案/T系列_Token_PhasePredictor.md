# T 系列：Token + Future Query + Phase Predictor（v2 · 相对 P2）

> **基线对照**：**P2**（0.5707±0.0112 @5060）· **A1**（0.5717±0.0236）  
> **动机**：U1 只升级 Predictor 输入为序列，但 **L_pred 仍对齐 mean 向量、Expert 仍 mean-pool** → 仅 +0.15 pp。  
> **T1** 补齐：Future Query + Phase 查表 + **token 级 L_pred** + **AttnPool 读出**。  
> **代码**：`5060_mask_future_dual_expert_accpaper` · 臂 `T1` / `T1_aux` / `T1_128`  
> **前置**：pf1000 须含 `openbmi_t0_sec.npy`（protocol_version≥3）

---

## 0. 与 P2 / U1 的关系

| 组件 | P2 | U1 | **T1** |
|------|----|----|--------|
| Predictor 输入 | `mean(Z_vis)` | `Z_vis` 序列 | `Z_vis` tokens |
| Predictor 结构 | MLP | Attn+TCN→**单向量** | **Query + Cross-Attn → token 序列** |
| L_pred 目标 | `mean(Z_fut)` | 同 P2 | **`Z_fut` token 序列 MSE** |
| Phase 条件 | — | — | **E_pos + E_phase(t0,y 查表)** |
| Expert_cur / future | mean-pool | mean-pool | **AttnPool** |
| Decoder / Gate / SIGReg | 同 P2 | 同 P2 | **同 P2**（波形 + λ_dec=0.2） |
| D | 40 | 40 | 40（T1_128→128） |

**不进主表条件**：五折 Test Acc_paper **稳定 > A1** 且 **std ≤ P2** 方可宣称 v2 增益。

---

## 1. 在线契约（训练 / 推理一致）

| 项 | 训练 | 在线推理 |
|----|------|----------|
| 输入 | `X_mask` + `X_full`（target） | **仅 `X_mask`** |
| Future Phase | **`t0_sec` + `y_three` + 窗几何查表** | **同左**（不用 Future EEG） |
| Future Query | `E_pos[i] + E_phase[i]` | 同结构 |
| Decoder | 仅训练 | 不跑 |
| Phase 辅助 CE | 仅 T1_aux · 训练 | 不跑 |

**禁止**：从 Future 波形估计 Phase（泄漏）。

---

## 2. Phase 查表规则（`phase_lookup.py`）

窗几何：past100 + cur500 + future400 @ 250 Hz；`t0_sec` = current 段起点相对 cue（合法 0.4～2.0 s）。

Future token 中心时间（相对 cue）：

```text
t_rel_cue = t0_sec + (sample_idx(fi) - 100) / 250
```

| y_three | Phase id | 条件 |
|---------|----------|------|
| 1 / 2（MI） | 0 onset | t_rel < 0.5 s |
| 1 / 2 | 1 sustain | 0.5 ≤ t_rel < 3.5 s |
| 1 / 2 | 2 offset | t_rel ≥ 3.5 s（含 post-MI future） |
| 0（Rest） | 3 rest | 专用 idle 桶 |

Rest 与 Left/Right **不共用** MI onset/sustain/offset 表。

---

## 3. 实验臂定义

### T1 · 主臂（必跑 · 5060 第一枪）

| 项 | 值 |
|----|-----|
| arm_id | `T1` |
| embed_dim | **40** |
| batch_train | **128**（5060 默认） |
| λ_pred | **1.0**（L1-λ 扫完后再回填最优） |
| λ_cls / λ_sig / λ_dec | 1.0 / 0.05 / 0.2（同 P2） |
| 相对 P2 改动 | Query Predictor + Phase 查表 + token L_pred + AttnPool |

**验收**：
- §3.2.1 future 扰动单测仍过
- Test Acc_paper vs **A1**、**P2** Δ
- **fold1** 不得系统性低于 A1（5060 老问题）

### T1_aux · Phase 辅助 CE（T1 后可选）

| 项 | 值 |
|----|-----|
| arm_id | `T1_aux` |
| 相对 T1 | + `L_phase = CE(phase_head(Z_pre), phase_ids)`，**λ_phase=0.2** |
| 目的 | 检验显式 phase 监督是否缓解 fold1 多任务抢梯度 |

### T1_128 · 容量对照（最后跑）

| 项 | 值 |
|----|-----|
| arm_id | `T1_128` |
| embed_dim | **128** |
| batch_train | **64**（5060 OOM 降档，meta 记录） |
| 目的 | 区分 **结构收益 vs 纯维数** |

---

## 4. 5060 跑法

### 4.1 冒烟（fold0 · ~1h/臂）

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5060_mask_future_dual_expert_accpaper

python _smoke_local.py                    # 前向 + T 臂注册自检
python run_arm.py --arm T1 --dry-run
python run_arm.py --arm T1 --max-folds 1 --num-workers 0

# 或门控链 fold0
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 1
```

### 4.2 正式五折（~4h/臂 · 建议夜间）

```powershell
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole

# 断点续跑
powershell -File .\run_t_chain_guarded.ps1 -FromArm T1_aux -MaxFolds 0 -NoConsole

# 跳过 T1_128（算力紧）
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -SkipT1_128 -NoConsole
```

### 4.3 输出与登记

| 项 | 路径 |
|----|------|
| 权重 | `code/train_lab/out/5060_mask_future_dual_expert_accpaper/{stamp}_T1/` |
| 链状态 | `t_chain_guarded_state.json` |
| 登记 | `资料/模型训练/17_5060_.../`（跑后填） |

### 4.4 5090 / 5070 跑法

三机包 **代码同构**（`T1` / `T1_aux` / `T1_128` + `run_t_chain_guarded.ps1`）；仅 batch / workers / out 前缀不同。

| 机位 | 包路径 | 默认 batch | 链脚本 |
|------|--------|------------|--------|
| **5090** | `5090_mask_future_dual_expert_accpaper/` | 256/512 | `run_t_chain_guarded.ps1 -MaxFolds 0` |
| **5070** | `5070_mask_future_dual_expert_accpaper/` | 128/256 | 同左（本机正式） |
| 5060 | `5060_mask_future_dual_expert_accpaper/` | 128/256 | 同左（历史锚点） |

```powershell
# 5090 示例
cd code/train_lab/src/step/5090_mask_future_dual_expert_accpaper
python _smoke_local.py
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 1

# 5070 示例（路径相对仓库根）
cd code/train_lab/src/step/5070_mask_future_dual_expert_accpaper
powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole
```

5090/5070 权重分别落 `out/5090_mask_future_dual_expert_accpaper/`、`out/5070_mask_future_dual_expert_accpaper/`。

---

## 5. 执行顺序（5060 推荐）

```text
（建议先）L1-λ 或 C2a-5060  —  可选，不阻塞 T1 代码验证
        ↓
T1 fold0 冒烟  →  T1 五折
        ↓
T1_aux 五折（若 T1 fold1 仍崩）
        ↓
T1_128 五折（容量对照）
```

**算力最紧**：只跑 **T1 fold0 + T1 五折**。

---

## 6. 主表 / 附表用法

| 用途 | 臂 |
|------|-----|
| v1.x 主表 | 仍 **P2**（未定稿前不改） |
| v2 候选主表 | **T1**（若验收通过） |
| 消融 | T1 vs P2；T1_aux vs T1；T1_128 vs T1 |
| 附报 | U1（半升级失败对照） |

---

## 7. 结果记录（跑后填）

### T1

| 指标 | 值 |
|------|-----|
| Test Acc_paper | |
| vs P2 Δ | |
| vs A1 Δ | |
| std | |
| fold0…4 | |
| run 路径 | |

### T1_aux / T1_128

（同上）

---

## 8. 与定稿 §17 对齐

- **保留**：Mask、双专家、CE(cur+final)、SIGReg、波形 Decoder（5090 B/C 因果）
- **T1 只改**：Predictor 接口 + L_pred 目标 + Expert 读出 + Phase 查表
- **不做**：Spectral-only（U2 已负）、Gate 熵（U3/B5b）、U 组合堆叠
