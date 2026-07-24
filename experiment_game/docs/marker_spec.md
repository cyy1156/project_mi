# Marker / 事件规格（冻结）

> **版本**：v0.1  
> **日期**：2026-07-22  
> **状态**：Phase 1 冻结  
> **配套**：`游戏内容规格.md`、`项目计划.md` §5

权威时钟：`pylsl.local_clock()`（下称 `t_lsl`）。  
控制器在每个阶段边界**先**取 `t_lsl`，再写 `events.jsonl` 并推 LSL Marker；前端不参与打标。

---

## 1. 单 Trial 时间轴（已冻结）

墙钟合计 **17 s**（含 Transition）。相对 `trial_start`：

| 相对 t (s) | 时长 | 阶段 | 发出的事件（起点，除非注明） | 训练窗 |
|------------|------|------|------------------------------|--------|
| 0 | 2 | Fixation | `fixation` | 否 |
| 2 | 2 | Cue | `cue`（含 `label`∈{1,2}） | 对齐参考 |
| 4 | 4 | MI | `mi_start`；结束时 `mi_end` | **是** Left/Right |
| 8 | 1 | PostMI_hold | （可选不单独打点；计入 trial 内） | 否 Discard |
| 9 | 4 | Rest | `rest_start`；结束时 `rest_end`（`label=0`） | **是** Rest |
| 13 | 3 | Transition | `transition` | 否 Discard |

**切窗约定（与离线对齐）：**

- MI 窗：[`mi_start`, `mi_end`)，时长 4.0 s；等价于 `cue` 后 **+2.0 s 起**连续 4.0 s（Cue 展示 2 s 结束后进入纯静想象）。
- Rest 窗：[`rest_start`, `rest_end`)，时长 4.0 s；**不含** Transition。
- 若下游习惯「cue 后 0–4 s」语义，本系统提供显式 `mi_*`；适配器应优先用 `mi_start`/`mi_end`，避免把 Cue 展示段误切进训练窗。

---

## 2. `events.jsonl` 行格式

每行一个 JSON 对象，UTF-8，字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `t_lsl` | number | ✓ | `local_clock()` |
| `event` | string | ✓ | 见下表 |
| `trial_id` | int \| null | 试次级必填 | 从 1 递增；会话级事件可为 null |
| `label` | int \| null | cue/mi/rest 必填 | 0=Rest, 1=Left, 2=Right |
| `phase` | string | 建议 | `adapt` / `learn` / `acquire` |
| `object` | string | trial 建议 | 如 `cup` |
| `scene` | string | trial 建议 | 如 `home_desk` |
| `subject_id` | string | 会话级 | |
| `session_id` | string | 会话级 | |
| `reason` | string | reject 时 | |
| `payload` | string | 可选 | 与 LSL Marker 字符串一致 |

### 事件字典

| event | 何时 | 必要附加字段 |
|-------|------|----------------|
| `session_start` / `session_end` | 会话起止 | `subject_id`, `session_id` |
| `phase_start` / `phase_end` | 大阶段 | `phase` |
| `trial_start` / `trial_end` | 每 trial | `trial_id`, `object`, `scene`, `label`（本 trial 任务侧；Rest 段另见 rest_*） |
| `fixation` | 注视开始 | `trial_id` |
| `cue` | Cue 画面/提示开始 | `trial_id`, `label`∈{1,2} |
| `mi_start` / `mi_end` | MI 窗起止 | `trial_id`, `label`∈{1,2} |
| `rest_start` / `rest_end` | Rest 窗起止 | `trial_id`, `label=0` |
| `transition` | Transition 开始 | `trial_id` |
| `object_change` | 换物 | 新 `object` |
| `scene_change` | 换景 | 新 `scene` |
| `trial_reject` | 无效试次 | `trial_id`, `reason` |

`trial_start.label`：本 trial 的 Left/Right 任务标签（1 或 2），不是 Rest。

---

## 3. LSL `OpenBCI_Markers`

| 项 | 值 |
|----|-----|
| name | `OpenBCI_Markers` |
| type | `Markers` |
| channel_count | 1 |
| nominal_srate | 0（不规则） |
| channel_format | `string` |
| source_id | `experiment_game_markers` |

**样本内容**：与对应 `events.jsonl` 的 `payload` 相同，推荐：

```text
{event}|trial={trial_id}|label={label}|phase={phase}
```

例：`cue|trial=3|label=1|phase=acquire`

推送时间戳：与该行 `t_lsl` 相同（`outlet.push_sample([payload], timestamp=t_lsl)`）。

---

## 4. 会话目录

```text
data/sessions/{subject_id}_{session_id}_{YYYYMMDD_HHMMSS}/
  session.meta.json
  eeg.csv                 # lsl_connect Recorder：lsl_time + 8ch
  eeg.csv.meta.json       # 录制器 sidecar（可保留）
  events.jsonl
```

`session.meta.json` 至少含：`subject_id`, `session_id`, `phase_mode`, `sample_rate_hz=250`, `channel_labels`, `use_synthetic`, `trial_count`, `created_at`, 路径引用。

通道顺序（冻结）：

```text
C3, C4, CZ, CP3, CP4, CPZ, FC3, FC4
```

---

## 5. Phase 1 验收（对齐）

1. synthetic 连续 ≥20 trial  
2. 每个正式 trial：`mi_end - mi_start ≈ 4.0`（容差 ±0.05 s）  
3. 每个正式 trial：`rest_end - rest_start ≈ 4.0`  
4. `events` 的 `t_lsl` 落在 `eeg.csv` 的 `lsl_time` 覆盖区间内（会话中段）  
5. `t_lsl` 序列无系统性倒退  

工具：`tools/run_phase1_block.py`、`tools/verify_phase1_alignment.py`
