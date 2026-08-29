# 范式对齐：OpenBMI 底座 vs fnz OpenBMI-Align v1

> 日期：2026-08-27 · **修订：2026-08-29** · 状态：**OpenBMI-Align v1 已落地（§2 为现行规格）**  
> **冻结总表**：[`框架冻结确认_20260829.md`](框架冻结确认_20260829.md)（**F5 已冻结：因果平滑 + 多数票单轨**）  
> 目的：冻结 OpenBMI 训练底座所用范式，并规定 fnz **离线采集 + 离线 FT** 与之 **完全一致** 的目标协议；在线推理对齐见 §2.3。  
> 相关：[`marker_spec.md`](marker_spec.md)（Phase1 历史）· [`fnz实验与微调采集方案_20260827.md`](fnz实验与微调采集方案_20260827.md) · `code/preprocess_lab/src/datasets/openbmi/pipeline.py`

---

## 0. 一句话结论

| 链路 | 是否与 OpenBMI 对齐 |
|------|---------------------|
| **OpenBMI 预处理 / 5070 底座训练** | 基准（§1） |
| **BCI2a Exp29 预处理** | ✅ 与 OpenBMI 同代码路径 |
| **fnz OpenBMI-Align v1 采集（§2.1）** | ✅ **Cue = MI 起点，MI 固定 4 s** |
| **fnz OpenBMI-Align v1 离线 FT（§2.4）** | ✅ **与 OpenBMI 预处理同构** |
| **fnz OpenBMI-Align v1 在线推理（§2.3）** | ✅ 前向窗 + `t_cue` 参考 |
| **v2 会话（标定→准入→游戏）** | ✅ **与 v3 同 OpenBMI-Align v1 范式** |
| **历史 v3 session（ws01–ws03 等）** | ❌ 见 **附录 A**（prep+cue 4 s、MI 晚 2 s） |

**ws01–ws03 等历史 session 仍按旧范式采集，不可与 OpenBMI 底座直接混训；新 session 起按 §2 执行。**

---

## 1. 基准：OpenBMI（Lee2019-MI）范式

### 1.1 原始采集（`.mat`）

| 项 | 内容 |
|----|------|
| 数据 | `sess{01,02}_subj{NN}_EEG_MI.mat` · 块 **`EEG_MI_train`** |
| 试次事件 | 每个 Left/Right 试次 **一个 Cue 样本索引 `t`** |
| 通道 | 全导 → 预处理取 8 导：`Cz,C3,C4,CP3,FC4,FC3,CP4,CPz` |
| 采样率 | 原始 ~1000 Hz → 预处理输出 **250 Hz** |
| 试次内阶段 | **无**独立 prep / cue 展示 / post-MI / 试次内 Rest 打点 |
| 语义 | **`t` = Cue/MI 段起点**；被试从 `t` 起做 **4 s** 运动想象 |

> OpenBMI `.mat` 不存 fnz 式 `prep_start` / `mi_start`；Rest 来自 **试次间隔**，不是 label=0 的独立 MI 试次。

### 1.2 预处理切窗（`openbmi_3s_hop100` · 5070 底座）

实现：`preprocess_run_3s_hop100()` · 配置：`code/preprocess_lab/config/openbmi_3s_hop100.yaml`

| 段 | 时间定义 | 切窗 |
|----|----------|------|
| **Task（Left/Right）** | **Cue 后 `[0, 4) s`**（`task_window_cue_0_to_4`） | 段内 **3 s / hop 100 ms** 前向滑窗 |
| **Rest** | **下一 Cue 前最多 4 s**（与上一试次 MI 重叠则缩短） | 同上滑窗 |
| 段级基线 | Cue **前 0.5 s** 减均值 | 在滑窗前完成 |
| 窗内归一化 | 每窗 **z-score** | ✅ |
| 滤波 | CAR · 8–30 Hz | ✅ |

**Task 第一扇窗（相对 Cue）：** `[+0.0 s, +3.0 s]`  
**4 s 段内共 11 扇窗**（末扇 `[+1.0 s, +4.0 s]`）。

**标签：**

| 头 | 标签 |
|----|------|
| `y_three` | 0=Rest（间隔段）· 1=Left · 2=Right |
| `y_task` | 0=Rest · 1=MI（Left+Right 合并） |

### 1.3 OpenBMI 单 trial 时间轴（相对 Cue = 0）

```text
|--[========== MI / Task 4s ==========]--| ... ITI ... |--[ Rest ≤4s before next cue ]--|
 0                                     4              (试次间，非独立试次)
 ↑
 Cue = MI 训练段 t=0
```

---

## 2. fnz OpenBMI-Align v1 范式（目标 · 与 OpenBMI 一致）

> **§2 为 fnz 新 session 的冻结规格。** 采集时序、离线 FT 切窗与 §1 逐项同构；仅保留 fnz 必需的 prep / ITI / 游戏反馈层。

### 2.1 试次时序（采集）

配置：`experiment_game/config/v3_session.yaml`  
状态机：`experiment_game/experiment/trial_v2.py`

| 阶段 | 目标时长 | 事件 | 是否训练段 | 与 OpenBMI |
|------|----------|------|------------|------------|
| **专用试次间 Rest** | **4 s** | `rest_start` … `rest_end`（挂本 trial_id） | **是（Rest 段）** | ✅ 间隔纯静息（**不含 prep**） |
| Prep / Fixation | **2 s** | `prep_start` | 否 | OpenBMI 无此段；**不算 Rest 计分** |
| **Cue = MI onset** | **0 s 延迟** | **`cue` = `mi_start`**（同 LSL 时刻） | **是（起点）** | ✅ **`t` = Cue** |
| **MI** | **固定 4.0 s** | `mi_start` … `mi_end` | **是** | ✅ **`[0, 4) s`** |
| ITI | **≥3 s** | `iti_start` | 否 | 试次末缓冲；**下一试次 Rest 另起** |

**配置项变更（相对历史 v3）：**

| 参数 | 历史 v3 | OpenBMI-Align v1 |
|------|---------|------------------|
| `cue_s` | 2.0 s（MI 延迟） | **0 s** |
| `imagine_s` | 6.0 s（常早停） | **4.0 s**（固定，无 D8 早停） |
| `iti_s` | 3.0 s | **≥3 s**（试次末；不含专用 Rest） |
| Rest | label=0 独立试次或含糊「Cue前4s」 | **仅 `inter_trial_rest_s=4` 专用段**；其后 **prep 2s 独立** |
| `block_gap_s` | 60–90 | **默认 30s**（可 overrides，不永久锁死） |

**单 trial 事件序（相对 Cue = 0；与 `trial_v2.run_round` 一致）：**

```text
| 专用 Rest 4s | prep 2s |==== MI / Task 4s ========| ITI ≥3s |
      ↑ rest_*           0                          4
                         ↑
                         Cue = mi_start = OpenBMI t
```

> 物理序：… → ITI → **专用 Rest 4s** → **prep 2s** → Cue/MI → …。  
> **禁止**把「Cue 前 4s」说成 Rest——Cue 前还有 prep；Rest 只认 `rest_start`–`rest_end`。

### 2.2 试次类型与 Rest 语义

| 类型 | OpenBMI-Align v1 | OpenBMI |
|------|------------------|---------|
| Left / Right | Cue 后 **固定 4 s MI** | Cue 后 4 s MI 段 |
| **Rest** | **专用间隔 4 s**（`rest_start/rest_end`） | Cue 前 ≤4 s 无提示 EEG |
| Rest 想象试次（label=0） | **已取消** | 无 |

→ **三类标签**：Rest=专用间隔段，Left/Right=Task 段。计分：Rest 段多数票为 Rest → **+0.5**（36 试次满分 **54**）。

### 2.3 在线判定（OpenBMI-Align v1 · 已落地）

| 项 | 实现 |
|----|------|
| 默认读出 | **`readout_mode=e1f`**（四成员 three；`e1f_four_member.json`） |
| 参考时钟 | **`t_cue`（= `mi_start`）** |
| 滑窗 | **3 s / hop 100 ms**，锚点相对 Cue |
| 判定点 | 窗尾约 3.0…4.0 s（约 11 档） |
| 输出条件 | 窗尾 ≤ MI 4s 且窗长满 3 s |
| 段级基线 | **Cue 前 0.5 s** 减均值 |
| 窗内归一化 | per-channel **z-score** |
| **窗级** | E1f 融合 → **因果平滑**（n 与前两窗）→ **argmax** |
| **试次计分 = 主判定** | **因果平滑后多数票**（F5 已冻结单轨）；MI +1 / Rest +0.5；**不**用 τ 早停 |
| 首判时刻 | **Cue 后 3.0 s**（首窗 `[0, 3]`） |
| 底座 | 5090 E1f 四成员 fold0（见 json）；对照可选单模 Shallow |

实现：`InferenceService.window_mode=openbmi_hop100` · `v3_config.build_openbmi_judgment_times`

### 2.4 离线 FT 切窗（`ft_subject_from_v3.py` · 与 OpenBMI 完全对齐）

**规格：与 §1.2 `preprocess_run_3s_hop100` 同构。** 建议实现 `--protocol openbmi_align`，直接复用 `task_window_cue_0_to_4` + `segment_to_3s_hop100_windows` + **`t_rest_start/t_rest_end`（优先）**。

| 项 | OpenBMI-Align v1（fnz 离线 FT） | OpenBMI 预处理 |
|----|--------------------------------|----------------|
| Task 切段 | **`[t_cue, t_cue+4s)`** | **`[t_cue, t_cue+4s)`** |
| 首窗锚点 | **0 s**（相对 Cue/段起点） | **0 s** |
| 窗型 | **前向** 3 s / hop 100 ms | **前向** 3 s / hop 100 ms |
| Rest 来源 | **`[t_rest_start, t_rest_end)`**（Align 采集）；无打点时回退 Cue 前 4s | 试次间隔 Cue 前 ≤4 s |
| 段级基线 | **Cue 前 0.5 s** 减均值 | Cue 前 0.5 s |
| 窗内归一化 | 每窗 **z-score** | 每窗 z-score |
| 采样率 | 原生 250 Hz（已 250 Hz 则跳过重采样） | ~1000 Hz → 250 Hz |
| 滤波 | CAR · 8–30 Hz | CAR · 8–30 Hz |
| 标签 | `y_three`: 0/1/2 · `y_task`: 0/1 | 同左 |

**Task 第一扇窗（相对 Cue）：** **`[+0.0 s, +3.0 s]`** — 与 OpenBMI 一致。  
**4 s 段内共 11 扇窗**（末扇 `[+1.0 s, +4.0 s]`）。

**Left/Right 试次切窗流程（伪代码）：**

```python
# 与 preprocess_run_3s_hop100 同路径
seg = task_window_cue_0_to_4(x_filt, int(t_cue_idx), fs=250.0)  # Cue前0.5s基线 + [0,4)s
wins = segment_to_3s_hop100_windows(seg, fs=250.0, zscore=True)  # anchor=0, 11 窗
```

**Rest 切窗流程（Align 采集 · 优先）：**

```python
# 从 trial_table 的 t_rest_start / t_rest_end（Cue 前纯静息，不含 prep）
for row in rows:
    seg = extract_segment_baseline(x_filt, t_rest_start, t_rest_end, fs, baseline_sec=0.5)
    wins = segment_to_3s_hop100_windows(seg, fs=250.0, zscore=True)
# 无 rest 打点的历史 session → 回退 iter_rest_sources_cue_before
```

**相对历史 v3 的关键变更：**

| 项 | 历史 v3 | OpenBMI-Align v1 |
|----|---------|------------------|
| 切段 | `[t_mi_start, t_mi_end]` | **`[t_cue, t_cue+4s)`** |
| 首窗锚点 | `T0_MIN = 0.4 s` | **`0 s`** |
| Rest | Rest 试次 MI 段 | **试次间隔 4 s** |
| 段级基线 | 无 | **Cue 前 0.5 s** |
| 首窗（相对 Cue） | 约 `[+2.4, +5.4] s` | **`[+0.0, +3.0] s`** |

---

## 3. 三链路对照总表（OpenBMI-Align v1）

### 3.1 试次阶段

| 阶段 | OpenBMI | fnz 采集（Align v1） | fnz 离线 FT | fnz 在线（目标） |
|------|---------|----------------------|-------------|------------------|
| 准备/注视 | 无独立段 | 0–2 s prep（可选） | 不切窗 | 不判定 |
| Cue 展示 | **= MI 起点** | **= MI 起点** | 段起点 `t_cue` | 参考 `t_cue` |
| MI 起点 | **Cue (t)** | **Cue = mi_start** | **`t_cue`** | **`t_cue`** |
| MI 长度 | **固定 4 s** | **固定 4 s** | **`[t_cue, t_cue+4s)`** | 前向窗至 +4 s |
| 试次内 Rest | 无 | 无 | 无 | 无 |
| 试次间 Rest | **Cue 前 ≤4 s** | **专用 Rest 4s**（`rest_start/rest_end`，**不含 prep**） | **`[t_rest_start, t_rest_end)`** | 计分 +0.5；可灌 ERD |
| Rest 标签 | 间隔段=0 | 专用 Rest 段=0 | 同左 | — |

### 3.2 切窗几何（3 s / hop 100 ms / 750 点）

| 链路 | 参考点 | 首窗起点（相对 Cue） | 窗型 | 与 OpenBMI 对齐 |
|------|--------|----------------------|------|-----------------|
| OpenBMI 训练 | Cue | **+0.0 s** | 前向滑窗 | 基准 |
| fnz 离线 FT（Align v1） | Cue | **+0.0 s** | 前向滑窗 | ✅ |
| fnz 在线（Align v1 目标） | Cue | **+0.0 s** | 前向 3 s | ✅（P1） |

### 3.3 信号处理

| 步骤 | OpenBMI | fnz 离线 FT（Align v1） | fnz 在线（目标） |
|------|---------|-------------------------|------------------|
| 通道序 | 8 导冻结序 | 同 | 同 |
| CAR + 8–30 Hz | ✅ | ✅ | ✅ |
| 段级基线（Cue−0.5s） | ✅ | ✅ | ✅ |
| 窗内 z-score | ✅ | ✅ | ✅ |

---

## 4. 时间轴叠图（OpenBMI-Align v1 · 相对 Cue = 0）

```text
OpenBMI / fnz Align v1（Task 段 + 切窗）：
|=[==== Task / MI 4s · 11 窗 from 0 ========]|
 0                                          4
 ↑ 离线 FT 首窗 [0, 3s]

fnz Align v1 采集（单 trial 事件序 · 相对 Cue = 0）：
| 专用 Rest 4s | prep 2s |==== MI 4s ========| ITI |
      ↑ rest_*      0                  4
```

---

## 5. 与文档 / 实验的说明

| 文档或说法 | 说明 |
|------------|------|
| `marker_spec.md` 默认 **17 s** | **Phase1 历史**；Align v1 见 §2（专用 Rest + prep + MI4 + ITI） |
| Exp29 / BCI2a「与 OpenBMI 同范式」 | ✅ 预处理同构；fnz **新 session** 按 §2 |
| 历史 v3（ws01–ws03） | ❌ 未对齐，见 **附录 A** |
| 在线默认读出 | **E1f 四成员** + **因果平滑后多数票**（F5 已冻结；τ 早停退出正式 SOP） |

---

## 附录 A. 历史 v3 范式（ws01–ws03 · 未对齐 · 仅供参考）

> 2026-08-27 前已采集 session 的实际参数；**不得**作为 OpenBMI-Align v1 规格。

### A.1 试次时序（历史）

| 阶段 | 配置时长 | 事件 |
|------|----------|------|
| Prep | 2 s | `prep_start` |
| Cue | 2 s | `cue`（MI 未开始） |
| MI | ≤6 s（常早停 ~4.2 s） | `mi_start` … `mi_end` |
| ITI | 3 s | `iti_start` |

**ws01 实测：** Cue→mi_start **2.001 s**；mi_start→mi_end **4.215 s**（`score_5` 早停）。

### A.2 历史离线 FT（未对齐）

| 项 | 历史 v3 |
|----|---------|
| 切段 | `[t_mi_start, t_mi_end]` |
| 首窗锚点 | `T0_MIN = 0.4 s` |
| Rest | Rest 试次 MI 段 |
| 段级基线 | 无 |
| 首窗（相对 Cue） | 约 `[+2.4, +5.4] s` |

### A.3 历史在线（未对齐）

- 参考 **`mi_start`**；**trailing 3 s** 窗；首判 t_rel=0.6 → 窗约 `[-2.4, +0.6] s`（相对 mi_start）。

### A.4 对齐差距量化（ws01）

| 指标 | OpenBMI Task 段 | 历史 v3 MI 段（相对 Cue） |
|------|-----------------|---------------------------|
| 起点 | 0 s | **+2.0 s** |
| 终点 | 4 s | **~+6.2 s** |
| 与 OpenBMI 重叠长度 | 4 s | **~2 s** |
| 首训练窗（离线） | [0, 3] s | **[~2.4, 5.4] s** |

---

## 6. 代码与配置索引

| 用途 | 路径 |
|------|------|
| OpenBMI 3s 预处理 | `code/preprocess_lab/src/datasets/openbmi/pipeline.py` |
| 段提取 + 基线 | `code/preprocess_lab/src/common/steps/epoch_baseline.py` |
| 3s/hop100 滑窗 | `experiment_game/core/windowing.py`（权威；`offline/openbmi_align_cut` 薄包装） |
| OpenBMI 配置 | `code/preprocess_lab/config/openbmi_3s_hop100.yaml` |
| 5070 单模底座（对照） | `…/5070_baseline_…/run_20260822_094942/` |
| **5090 E1f 四成员（默认）** | `experiment_game/config/e1f_four_member.json` · `…/run_20260823_*` |
| 冻结总表 | `experiment_game/docs/框架冻结确认_20260829.md` |
| v3 会话配置 | `experiment_game/config/v3_session.yaml`（`block_gap_s` 默认 **30**） |
| 试次状态机 | `experiment_game/experiment/trial_v2.py` |
| 在线推理 | `experiment_game/experiment/inference_v2.py` |
| 离线 FT 切窗 | `experiment_game/tools/ft_subject_from_v3.py` |
| 事件 / 对齐表 | `experiment_game/experiment/alignment.py` |

---

## 7. 实施清单

- [x] **P0** `ft_subject_from_v3.py --protocol openbmi_align`：Cue+0~+4s + 间隔 Rest + `segment_to_3s_hop100_windows`
- [x] **P0** 采集：`cue_s→0`；`imagine_s=4.0`；`mi_start` 与 `cue` 同刻；试次间 4 s Rest 打点
- [x] **P0** `v3_session.yaml` / `trial_v2.py` 按 §2.1 改时序
- [x] **P1** 在线 `judge()` → 前向窗 + `t_cue` 参考
- [x] **P1** `alignment.py` 写入 `t_cue` / `rest_start` / `rest_end` 列供 FT
- [x] **P2** v2/v3：MI 多数票计分（`MiTrialTracker`）；废除 D8 早停/熔断
- [x] **P2** 块前 Rest `baseline_rest_s=30` → ERD 基线

---

## 8. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-08-27 | 初稿：OpenBMI vs fnz v3 对照 |
| 2026-08-27 | **§2 重写为 OpenBMI-Align v1**：MI 固定 4 s、Cue=MI onset；**§2.4 离线 FT 与 OpenBMI 完全对齐**；历史 v3 移至附录 A |
| 2026-08-27 | MI 多数票计分 + baseline_rest=30 对齐 fnz 方案 |
| 2026-08-29 | Rest 明确为**专用 4s（不含 prep）**；默认 E1f；块间默认 30s；**F5 冻结因果平滑+多数票单轨** |
