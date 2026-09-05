# 操作者采集控制 UI · 需求说明（扩展版）

> **版本**：v0.3.1  
> **日期**：2026-07-23  
> **状态**：需求扩展稿（先评审，再实现）  
> **目标用户**：实验操作者（非被试）  
> **形态（已定）**：本机网页操作台  
> **配套**：`完整使用流程.md`、`marker_spec.md`、`游戏内容规格.md`、`操作按键设计.md`、`ws_protocol.md`  
> **v0.3 要点**：数据采集按时间节点分文件夹落盘；EEG 与范式 Marker **严格同一 LSL 时钟对齐**  
> **v0.3.1**：Setup 点「开始实验」后 **自动打开诱导网页**（已定）

---

## 1. 一句话目标

提供**本机网页操作台**：每次实验开始前先进入**设置页**选好采集/设备/保存等选项，确认后再打开被试诱导页并开跑；实验中可监控与控制流程；结束后清楚看到数据落盘位置。命令行仍可保留给专家，但不作为日常主路径。

---

## 2. 已拍板决策

| 项 | 决定 |
|----|------|
| UI 形态 | **本机网页操作台**（非独立桌面窗；后期若需要再评估） |
| 启动顺序 | **先设置 → 点「开始实验」→ 自动打开诱导页并开跑** |
| 自动开诱导页 | **已定**：填写/确认 Setup 信息并开始后，**必须自动**打开被试诱导网页（新标签或新窗口）；无需操作者再手动输 URL |
| 被试页 | 独立全屏诱导页；设置与危险操作不放在被试主视野 |
| 扩展方式 | 设置项用**分组 + 配置 schema**，方便后续加字段而不推翻页面结构 |

---

## 3. 要解决的问题

| 痛点 | 目标体验 |
|------|----------|
| 记不住 `--with-acq --real --port` | 设置页勾选/下拉即可 |
| 误用默认「不采数」导致没有 `eeg.csv` | 设置里明确「是否开启采集」，并有保存后果提示 |
| 合成板/真机切换靠改命令 | 设置里一键切换；真机再填 OpenBCI 参数 |
| 数据存哪不清楚 | 设置里选保存策略；结束后一键打开目录 |
| 以后要加滤波、通道、切窗等 | 设置区分「基础 / 设备 / 实验 / 保存 / 高级」，可扩展 |

---

## 4. 整体信息架构：两屏流程

```text
双击 open_operator.bat
        │
        ▼
┌───────────────────────┐
│  屏 A：实验前设置 Setup │  ← 每次开启默认停在这里（填被试/采集/设备/保存）
│  （可折叠高级项）       │
│  [保存为默认] [开始实验] │
└───────────┬───────────┘
            │ 校验通过
            │ ① 进入 Run 视图
            │ ②【自动】打开诱导网页 http://127.0.0.1:8080/
            │ ③ 启动实验编排（等待诱导页 ready 后推进）
            ▼
┌───────────────────────┐
│  屏 B：运行监控 Run     │  ← 操作者留在此页控场
│  状态 / 阶段 / 控制按钮 │
│  [重新打开诱导页] [中止] │  ← 仅作补救（弹窗被拦/误关标签时）
└───────────┬───────────┘
            │ 结束
            ▼
┌───────────────────────┐
│  屏 C：结束与保存 Summary│
│  路径 / 文件清单 / 打开文件夹 │
└───────────────────────┘
```

路由建议（实现时可同页切换视图，不必多文件）：

| 视图 | URL 建议 | 何时可进 |
|------|----------|----------|
| Setup | `/operator.html#setup` | 默认；会话未跑或已结束 |
| Run | `/operator.html#run` | 已「开始实验」后 |
| Summary | `/operator.html#summary` | 会话结束或中止后 |

**原则：**

- 未完成 Setup、未点「开始实验」前：**不**自动打开被试诱导页、不启动正式编排。  
- 点「开始实验」且校验通过后：**必须自动**打开诱导页（见 §6.1），操作者无需再找地址。  
- 诱导页与操作台为**两个浏览器上下文**（推荐新标签）；操作台继续显示 Run，不跳转离开。

---

## 5. 屏 A · 实验前设置（核心）

### 5.1 交互原则

1. **每次**进入操作台默认到 Setup（除非用户勾选「下次直接沿用上次配置」且仍显示摘要条供确认）。  
2. 修改任意影响采集/落盘的项后，「开始实验」前做校验。  
3. 提供「恢复推荐默认」「保存为本地默认」「导出/导入配置 JSON」（导出导入可放扩展）。  
4. 设置分组折叠：日常只看「常用」；「设备细节 / 高级」默认收起。

### 5.2 设置分组总表

#### 分组 0 · 被试与会话（常用）

| 字段 ID | 控件 | 默认 | 说明 |
|---------|------|------|------|
| `subject_id` | 文本 | `sub01` | 被试编号 |
| `session_id` | 文本 | `ses01` | 会话编号 |
| `operator_notes` | 多行文本 | 空 | 写入 meta 备注 |
| `open_subject_page` | 开关 | **开（强制默认）** | 「开始实验」时**自动打开**诱导页；日常实验应保持开启；仅联调专家可关 |

#### 分组 1 · 采集开关（常用，必看）

| 字段 ID | 控件 | 默认（建议） | 说明 |
|---------|------|--------------|------|
| `acq_enabled` | 开关 **是否开启采集** | **开**（操作台主路径） | 关 = 只跑画面，无 `eeg.csv` |
| `board_mode` | 单选 | `synthetic` | `synthetic` 合成板调试 / `cyton` 真机 |
| `acq_warning` | 只读提示 | — | `acq_enabled=false` 时红条：「本次不能训练切窗」 |
| `gui_conflict_hint` | 只读提示 | — | 真机时固定提示关闭 OpenBCI GUI 串口直播 |

逻辑：

- `board_mode=synthetic` 时，串口等真机项禁用（灰显）。  
- `acq_enabled=false` 时，设备参数组禁用，但仍可跑 adapt→learn→gate→acquire 画面。  
- `acq_enabled=true` 且 `cyton`：必须填有效 `serial_port` 才能「开始实验」。

#### 分组 2 · OpenBCI / 设备参数（常用 + 可扩展）

**MVP 必显：**

| 字段 ID | 控件 | 默认 | 说明 |
|---------|------|------|------|
| `serial_port` | 文本或下拉 | `COM3` | 真机串口；支持「刷新串口列表」（第二批） |
| `sample_rate_hz` | 只读或下拉 | `250` | 本项目冻结 250；MVP 只读展示 |
| `channel_count` | 只读 | `8` | 固定 8 |
| `channel_labels` | 只读列表或可编辑文本 | 项目约定 8 名 | MVP 只读；扩展允许编辑并校验数量 |

通道默认顺序（与现有一致）：

```text
C3, C4, CZ, CP3, CP4, CPZ, FC3, FC4
```

**扩展（高级 · 设备，默认折叠）：**

| 字段 ID | 控件 | 默认 | 说明 | 分期 |
|---------|------|------|------|------|
| `filter_enabled` | 开关 | 跟随 lsl_connect 默认 | 采集端滤波总开关 | UI-2+ |
| `bandpass_low_hz` / `bandpass_high_hz` | 数字 | 0.5 / 45 | 带通 | UI-3 |
| `notch_low_hz` / `notch_high_hz` | 数字 | 49 / 51 | 陷波 | UI-3 |
| `gui_udp_stream` | 开关 | 关 | 旁路推给 GUI 监视（非串口直播） | UI-3 |
| `gui_udp_ip` / `gui_udp_port` | 文本/数字 | 225.1.1.1 / 6677 | 仅旁路开启时 | UI-3 |
| `include_accel` | 开关 | 关 | CSV 是否含加速度 | UI-3 |
| `lsl_eeg_name` | 文本 | `OpenBCI_EEG` | 一般勿改 | UI-3 |
| `board_id_override` | 文本 | 空 | 非标板扩展位 | 远期 |

#### 分组 3 · 实验任务参数（常用）

与范式对标，控制 SessionRunner：

| 字段 ID | 控件 | 默认 | 说明 |
|---------|------|------|------|
| `acquire_trials` | 数字 | 40 | 正式采集 trial 数 |
| `learn_trials_per_step` | 数字 | 2 | 学习每 Step 试次数 |
| `skip_adapt` | 开关 | 关 | 跳过适应（调试用） |
| `skip_learn` | 开关 | 关 | 跳过学习（调试用） |
| `skip_gate` | 开关 | 关 | 跳过准入（调试用，正式实验应关） |
| `seed` | 数字/空 | 空或固定 | 左右手随机种子；空=每次随机 |

**扩展（实验 · 高级）：**

| 字段 ID | 说明 | 分期 |
|---------|------|------|
| `object_pool` / 是否换物 | 正式段换物策略 | 已有后端可挂 UI |
| `scene_rotate` | 每 10 trial 换景开关 | UI-2 |
| `timing_profile` | 标准 17s / 缩短调试时序 | UI-3（需后端支持） |
| `allow_subject_gate` | 是否允许被试空格过准入 | 默认否，与按键设计一致 |

#### 分组 4 · 保存数据选择（常用）

| 字段 ID | 控件 | 默认 | 说明 |
|---------|------|------|------|
| `save_root` | 路径文本 +「浏览」* | `experiment_game/data/sessions` | 会话根目录；*浏览依赖后端 API |
| `save_layout` | 单选 | `phase_folders` | 见 §8：`flat`（旧扁平）/ `phase_folders`（按时间节点分目录，**推荐**） |
| `save_eeg` | 开关 | 随 `acq_enabled` | 开启采集时强制为开；不采数时强制关 |
| `save_events` | 开关 | 开 | 必开（关则禁止开始，或灰显不可关） |
| `save_session_meta` | 开关 | 开 | 必开 |
| `save_markers_lsl` | 开关 | 随采集 | 是否推 `OpenBCI_Markers` |
| `save_continuous_master` | 开关 | 开 | 另保留全会话连续 `eeg`+`events` 总档（推荐，防切段丢对齐） |
| `save_phase_slices` | 开关 | 开 | 按 phase/learn_step 写入子文件夹 |
| `save_trial_index` | 开关 | 开 | 生成 `alignment/trial_table.csv` 等索引 |
| `filename_pattern` | 只读预览 | `{subject}_{session}_{timestamp}` | 预览最终**会话根**文件夹名 |
| `copy_to_extra_dir` | 路径可选 | 空 | 结束后额外复制一份（扩展） |
| `auto_phase4` | 开关 | 关 | 结束后自动切窗（扩展） |
| `keep_raw_unfiltered_copy` | 开关 | 关 | 若后端支持双份（远期） |

保存后果提示（设置页底部固定）：

| 条件 | 提示 |
|------|------|
| 采集开 + 分目录 | 「会话根下：连续总档 + 各时间节点子文件夹；Marker 与 EEG 共用 LSL 时钟」 |
| 采集关 | 「仅 events + meta，无脑电，不能 Phase4 训练」 |
| 根目录无写权限 | 拦截「开始实验」 |

> 分文件夹与对齐细则见下文 **§8 数据采集与保存详细规格**。

#### 分组 5 · 操作台偏好（可折叠）

| 字段 ID | 说明 | 分期 |
|---------|------|------|
| `remember_last_config` | 记住上次设置 | UI-1 |
| `skip_setup_if_unchanged` | 勾选后下次进 Run 前仍弹「确认条」 | UI-2 |
| `language` | 预留 zh/en | 远期 |
| `operator_hotkeys` | 是否启用 P/N/G/R/Esc | UI-1 默认开 |

---

## 6. 「开始实验」校验与自动开页

### 6.1 点击「开始实验」后的标准动作（已定）

按序执行，任一步失败则停留 Setup 并提示，**不**打开诱导页、不启动编排：

1. 校验表单（见 §6.2）。  
2. 写入本次 `run_config` 快照；创建会话根目录（及 continuous 等占位，按保存配置）。  
3. 若需采集：启动采集/录制并确认就绪。  
4. **切换到 Run 视图**（操作台本页）。  
5. **自动打开诱导网页**（默认 `http://127.0.0.1:{http_port}/`，可用配置覆盖）：  
   - 方式：`window.open(subject_url, "_blank")` 或由后端调系统默认浏览器打开同一 URL。  
   - 若弹窗被浏览器拦截：Run 页醒目提示「请允许弹窗」+ **「重新打开诱导页」** 大按钮。  
6. 等待诱导页 WebSocket `ready`（超时可配置，如 60s）后，再推进 adapt 等流程；超时则提示检查诱导页是否已开、是否已连接。  
7. 启动 SessionRunner 编排。

> 用户诉求落地：**填完 Setup 信息 → 点开始 → 诱导页自动出来**；操作者继续在操作台控场。

### 6.2 校验清单

点击「开始实验」时按序检查：

1. `subject_id` / `session_id` 非空且合法（建议：字母数字下划线）。  
2. 若 `acq_enabled`：后端能进入采集+录制；真机则 `serial_port` 非空。  
3. `save_events` 与 `save_session_meta` 必须为真。  
4. `save_root` 可创建。  
5. 通过后执行 §6.1 第 2～7 步。

失败时：留在 Setup，字段旁显示错误，不打开被试页。

### 6.3 诱导页 URL 约定

| 项 | 约定 |
|----|------|
| 默认 URL | `http://127.0.0.1:8080/`（与静态服务端口一致） |
| 查询参数（可选） | 如 `?ws=ws://127.0.0.1:8765`，保证跨端口也能连上 |
| 重复点击「重新打开诱导页」 | 允许再开标签；后端以最新 `ready` 客户端为准或广播全员 |
---

## 7. 屏 B · 运行监控（简要）

设置锁定（只读摘要条）：采集开/关、合成/真机、COM、正式 trial 数、保存根目录。

| 区域 | 内容 |
|------|------|
| 步骤条 | adapt → learn1/2/3 → gate → acquire → end |
| 实时状态 | phase / stage / trial / label / object / scene / reject |
| 采集灯 | 未采 / 采中 / 录制中 / 错误 |
| 控制 | 暂停、代确认、准入、Reject、中止、**重新打开诱导页**（补救） |
| 警告 | 无采集时持续黄/红条 |

实验进行中：**禁止改**设备与保存根路径；需改则先中止。

---

## 8. 数据采集与保存详细规格

> 本章是操作台「保存」能力的**权威需求**；实现与验收以此为准。  
> 事件名字段与单 trial 时序仍服从 [`marker_spec.md`](./marker_spec.md)；本章补充**目录分层、对齐规则、记号约定**。

### 8.1 目标

1. **按时间节点分文件夹**：一次实验会话下，按范式大阶段（及学习 Step）分别落盘子目录，便于人工查阅与分阶段质控。  
2. **数据点与范式标记严格对齐**：EEG 样本时间戳与全部 Marker 使用同一权威时钟 `pylsl.local_clock()`（`t_lsl` / `lsl_time`），禁止用浏览器墙钟或操作系统「现在」作为打标主轴。  
3. **记号完备可追溯**：每个关键边界（会话 / 阶段 / trial / fixation·cue·mi·rest·transition / reject）都有事件记录，并与 LSL Marker 字符串一致。  
4. **可训练子集清晰**：主训练集默认只用 `phase=acquire` 且未被 reject 的 MI/Rest 窗；adapt/learn 分目录存放，避免混训。

### 8.2 时间节点定义（用于建文件夹）

| 节点代号 | 文件夹名建议 | 起止边界事件 | 说明 |
|----------|--------------|--------------|------|
| session | （会话根） | `session_start` → `session_end` | 一场实验总容器 |
| adapt | `01_adapt` | `phase_start(adapt)` → `phase_end(adapt)` | 环境适应 |
| learn_s1 | `02_learn_step1` | 学习 Step1 起止（见下） | 完整抓取观察 |
| learn_s2 | `03_learn_step2` | Step2 起止 | 前伸无抓 |
| learn_s3 | `04_learn_step3` | Step3 起止 | 无辅助想象 |
| gate | `05_gate` | 进入准入提示 → 准入确认完成 | 可无 EEG 子文件，但必须有事件 |
| acquire | `06_acquire` | `phase_start(acquire)` → `phase_end(acquire)` | **主数据集** |
| summary | `99_summary` | 会话结束后生成 | 索引、校验报告、可选切窗指针 |

学习 Step 边界：在现有 `phase=learn` 之上，**必须**增补可机读记号（实现时写入 events，并同步 marker_spec）：

| event | 字段 | 含义 |
|-------|------|------|
| `learn_step_start` | `learn_step`∈{1,2,3}, `phase=learn` | 该 Step 第一个 trial 前 |
| `learn_step_end` | `learn_step`∈{1,2,3} | 该 Step 最后一个 trial 后 |

（若暂时仅用 trial_id 约定：`101–199`→step1 等，分目录仍以 `learn_step_start/end` 为准，避免歧义。）

### 8.3 推荐目录布局（`save_layout=phase_folders`）

```text
{save_root}/
  {subject_id}_{session_id}_{YYYYMMDD_HHMMSS}/          ← 会话根（一场实验）
    session.meta.json                                   ← 全局元信息
    run_config.json                                     ← 本次 Setup 快照
    manifest.json                                       ← 所有子路径与时间范围索引

    continuous/                                         ← 全场连续总档（强烈推荐开启）
      eeg.csv                                           ← lsl_time + 8ch，连续不中断
      eeg.csv.meta.json                                 ← 录制器 sidecar（可有）
      events.jsonl                                      ← 全场全部事件（权威事件总表）
      markers.jsonl                                     ← 与 LSL Marker 推送镜像（可选，便于离线）

    by_phase/                                           ← 按时间节点分包
      01_adapt/
        phase.meta.json                                 ← t_start, t_end, n_trials, …
        events.jsonl                                    ← 本节点事件子集（或符号链接说明）
        eeg.csv                                         ← 由 continuous 按 [t_start,t_end) 切出
        README.txt                                      ← 一行人话：本段用途、勿作主训练
      02_learn_step1/
        phase.meta.json
        events.jsonl
        eeg.csv
      03_learn_step2/ …
      04_learn_step3/ …
      05_gate/
        phase.meta.json
        events.jsonl                                    ← 通常无 eeg 切片或极短
      06_acquire/
        phase.meta.json
        events.jsonl
        eeg.csv                                         ← 主分析脑电段
        trials/                                         ← 可选扩展：按 trial 再拆
          trial_001/
            trial.meta.json                             ← label, object, scene, reject?
            events.jsonl
            # eeg 可不按 trial 再切文件，避免碎文件过多；用索引定位即可

    alignment/                                          ← 对齐与记号索引（必做）
      trial_table.csv                                   ← 每 trial 一行，含各 marker 的 t_lsl
      marker_dictionary.json                            ← 本会话用到的 event 枚举与含义
      verify_report.json                                ← 自动对齐校验结果
```

**扁平兼容（`save_layout=flat`）**：仅会话根下 `eeg.csv` + `events.jsonl` + `session.meta.json`（当前 Phase1 行为）。操作台默认应为 **`phase_folders`**。

### 8.4 连续总档 vs 分节点文件（对齐优先策略）

| 策略 | 做法 | 优劣 |
|------|------|------|
| **A. 连续采 + 节点切片（推荐默认）** | 全程只开一条 EEG 录制写入 `continuous/eeg.csv`；节点结束时按 Marker 的 `t_lsl` **切片拷贝**到 `by_phase/.../eeg.csv`；events 实时双写：总表 + 当前节点表 | 无启停缝隙；对齐最稳 |
| B. 每节点停录再开录 | 在 `phase_end` 停 CSV，下一节点再开新文件 | 易在边界丢样/缝隙，**不推荐作主路径** |

需求约定：

- **主路径采用策略 A**。  
- `by_phase/*/eeg.csv` 必须能通过 `phase.meta.json` 中的 `t_start_lsl` / `t_end_lsl` 与 `continuous/eeg.csv` 对上（容差见 §8.7）。  
- 若某节点无采数（`acq_enabled=false`），仍写 events 与 meta，不写 eeg。

### 8.5 标记（Marker）与实验范式严格对齐

#### 8.5.1 权威时钟

- EEG 列名：`lsl_time`（float，秒）。  
- 事件字段：`t_lsl`（float，秒）。  
- LSL Marker 推送时间戳 = 该事件的 `t_lsl`。  
- **禁止**：用 `Date.now()`、Python `time.time()` 作为打标主轴（墙钟仅可写入 meta 的 `created_at` 等人读字段）。

#### 8.5.2 打标时机

控制器状态机在每个边界：

1. `t = local_clock()`  
2. 写 `events.jsonl` 一行（总表 + 当前节点表）  
3. 若开启：`OpenBCI_Markers.push_sample([payload], timestamp=t)`  
4. 再进入该段等待（wait_until）

前端只渲染，**不产生**训练用时间戳。

#### 8.5.3 单 Trial 必打记号（与 marker_spec 一致）

正式/学习 trial 至少包含：

```text
trial_start → fixation → cue → mi_start → mi_end
  →（post_mi_hold 可不单独打点）→ rest_start → rest_end → transition → trial_end
```

另加会话/阶段级：`session_*`、`phase_*`、`learn_step_*`、`object_change` / `scene_change`、`trial_reject`。

#### 8.5.4 记号字符串（payload）规范

与 LSL / `events.payload` 统一：

```text
{event}|trial={trial_id}|label={label}|phase={phase}|step={learn_step}|obj={object}|scene={scene}
```

规则：

- 无值的字段可省略（如 fixation 可无 label）。  
- `label`：`0=Rest, 1=Left, 2=Right`。  
- `phase`：`adapt|learn|acquire|gate`。  
- 例：`mi_start|trial=12|label=2|phase=acquire|obj=cup|scene=home_desk`

#### 8.5.5 `alignment/trial_table.csv`（记号总表）

每行一个 trial（至少 `phase=acquire`；学习段建议同样产出），核心列：

| 列名 | 含义 |
|------|------|
| `trial_id` | 试次号 |
| `phase` | adapt/learn/acquire |
| `learn_step` | 1/2/3 或空 |
| `label` | 1/2（任务手） |
| `object` / `scene` | 物品与场景 |
| `rejected` | 0/1 |
| `t_trial_start` … `t_trial_end` | 各边界 `t_lsl` |
| `t_fixation` / `t_cue` / `t_mi_start` / `t_mi_end` / `t_rest_start` / `t_rest_end` / `t_transition` | 严格对齐用 |
| `mi_dur` / `rest_dur` | 由 end−start 计算，便于质检 |
| `eeg_path` | 指向所属 phase 的 eeg 相对路径 |
| `events_path` | 所属 events 相对路径 |

下游切窗：**优先用 `t_mi_start`–`t_mi_end` 与 `t_rest_start`–`t_rest_end`**，不得用未校准的墙钟。

### 8.6 各节点应保存的内容清单

| 节点 | events | eeg 切片 | 特殊 |
|------|--------|---------|------|
| continuous | 全场 | 全场连续 | 对齐金标准 |
| 01_adapt | ✓ | ✓（若采数） | meta 标注 `train_eligible=false` |
| 02–04 learn | ✓ | ✓ | `learn_step`；`train_eligible=false`（默认可录但不进主训） |
| 05_gate | ✓ | 可选 | 记录准入确认时间 |
| 06_acquire | ✓ | ✓ | `train_eligible=true`（未 reject） |
| alignment | — | — | trial_table + verify_report |
| 99_summary | — | — | 人读摘要、文件勾选结果 |

`phase.meta.json` 最小字段：

```json
{
  "node": "06_acquire",
  "phase": "acquire",
  "learn_step": null,
  "t_start_lsl": 0.0,
  "t_end_lsl": 0.0,
  "n_trials": 40,
  "n_rejected": 1,
  "train_eligible": true,
  "acq_enabled": true,
  "files": { "events": "events.jsonl", "eeg": "eeg.csv" }
}
```

### 8.7 对齐验收（保存后自动跑）

生成 `alignment/verify_report.json`，至少检查：

1. `events` 中 `t_lsl` **单调不减**（允许相等，禁止系统性倒退）。  
2. 每个 acquire trial：`|mi_end - mi_start - 4.0| ≤ 0.05`；`|rest_end - rest_start - 4.0| ≤ 0.05`。  
3. 所有 `t_*` 落在 `continuous/eeg.csv` 的 `[lsl_time_min, lsl_time_max]` 内（会话中段事件）。  
4. 每个 `by_phase/*/eeg.csv` 的首末 `lsl_time` 与对应 `phase.meta` 的 `[t_start, t_end)` 一致（容差 ≤ 2 个采样点，即 ≤ 8 ms @250Hz，实现可配置）。  
5. `trial_table` 行数 = 未中止情况下的计划 trial 数（或给出缺失列表）。  
6. 被 `trial_reject` 的 trial 在表中 `rejected=1`，且 Phase4 默认排除。

操作台 Summary 页应展示：`PASS / FAIL` + 失败条目摘要。

### 8.8 屏 C · 结束与保存（UI）

- 展示会话根路径、`continuous` 与各 `by_phase` 子目录树勾选状态  
- 「打开会话根文件夹」「复制路径」「打开 alignment/verify_report」  
- 对齐结果徽章：通过 / 未通过  
- 「再开一场」回 Setup（可沿用配置）  
- 若 `auto_phase4`：仅对 `06_acquire` + `rejected=0` 切窗，输出仍到 `data/epochs/...`，并在 summary 写指针  

### 8.9 与 Setup `storage` 字段对应

| 配置 | 行为 |
|------|------|
| `save_layout=phase_folders` | §8.3 布局 |
| `save_continuous_master=true` | 写 `continuous/` |
| `save_phase_slices=true` | 写 `by_phase/` |
| `save_trial_index=true` | 写 `alignment/` |
| `save_eeg=false` | 各处不写 eeg，但仍写 events/meta/alignment（无 eeg 校验项跳过并标明） |

### 8.10 实现分期（保存专项）

| 分期 | 内容 |
|------|------|
| **Save-1** | 会话根 + `continuous/` 全场 eeg/events；`alignment/trial_table.csv` + 基础 verify；manifest |
| **Save-2** | `by_phase/` 按节点切片；`phase.meta.json`；Summary 目录树 |
| **Save-3** | `learn_step_*` 事件补齐；可选 `06_acquire/trials/`；verify 报告进操作台徽章 |
| **Save-4** | 与 Phase4 默认只读 acquire 切片对接；自动排除 reject |

---

## 9. 配置持久化与可扩展 Schema

### 9.1 本地默认配置文件（计划）

```text
experiment_game/config/operator_defaults.json
```

- 「保存为默认」写入此文件。  
- 每次打开 Setup 先加载默认，再允许当场改。  
- 每次「开始实验」另存快照到会话目录：`run_config.json`（与 meta 互补，便于复现）。

### 9.2 配置对象草案（可扩展）

```json
{
  "schema_version": 2,
  "subject": { "subject_id": "sub01", "session_id": "ses01", "notes": "" },
  "acquisition": {
    "enabled": true,
    "board_mode": "synthetic",
    "serial_port": "COM3",
    "sample_rate_hz": 250,
    "channel_labels": ["C3", "C4", "CZ", "CP3", "CP4", "CPZ", "FC3", "FC4"],
    "filter": { "enabled": true },
    "markers_lsl": true
  },
  "experiment": {
    "acquire_trials": 40,
    "learn_trials_per_step": 2,
    "skip_adapt": false,
    "skip_learn": false,
    "skip_gate": false,
    "seed": null,
    "open_subject_page": true
  },
  "storage": {
    "save_root": "experiment_game/data/sessions",
    "save_layout": "phase_folders",
    "save_eeg": true,
    "save_events": true,
    "save_session_meta": true,
    "save_continuous_master": true,
    "save_phase_slices": true,
    "save_trial_index": true,
    "auto_phase4": false,
    "extra_copy_dir": null
  },
  "ui": {
    "remember_last_config": true
  },
  "extensions": {}
}
```

**扩展约定：**

- 未知字段放进 `extensions` 或忽略，不导致启动失败（前向兼容）。  
- `schema_version` 升级时写迁移说明。  
- 新功能优先加字段 + Setup 分组，不新开互斥 App。

---

## 10. 与实验任务对标

| 实验阶段 | Setup 相关 | Run 相关 | 保存落点 |
|----------|------------|----------|----------|
| 适应 / 学习 / 准入 / 正式 | `skip_*`、trial 数、种子 | 步骤条高亮、代确认/准入/Reject | `by_phase/01`…`06` |
| 是否采数 | `acq_enabled` | 状态灯；无 eeg 警告 | continuous + 切片有/无 |
| 合成调试 vs 真机 | `board_mode` + 设备参数 | 错误提示（COM/GUI 占用） | meta 记录 |
| 落盘 | `storage.*` | Summary 打开目录 | §8 布局 |
| 记号对齐 | — | 状态机打标 | `alignment/trial_table` + verify |

---

## 11. 系统接口（扩展）

```text
操作者 Setup/Run ──WS/HTTP──► Python 服务
         │                      ├─ 应用 run_config
         │                      ├─ AcquisitionFacade
         │                      ├─ SessionRunner
         │                      └─ 落盘 sessions/...
         └─ 打开 /（被试页）
```

| 方向 | type（草案） | 用途 |
|------|----------------|------|
| UI→服务 | `config_validate` | 开始前校验 |
| UI→服务 | `session_start` | 携带完整 run_config |
| UI→服务 | `acq_start` / `acq_stop` | 若改为「先连设备再开跑」两步式 |
| UI→服务 | `list_serial_ports` | 刷新 COM |
| UI→服务 | `open_folder` | 打开会话目录 |
| 服务→UI | `config_ack` / `acq_status` / `stage` / `session_saved` | 状态回推 |

细节实现时写入 `ws_protocol.md`。

---

## 12. 启动方式

| 方式 | 行为 |
|------|------|
| `open_operator.bat`（计划） | 起服务 → 打开 **操作台 Setup** |
| 现有 `open_induction.bat` | 保留；文档标注为「旧快捷方式 / 被试页」，建议改走操作台 |
| 命令行 | 专家路径；可加 `--config path.json` 对齐同一 schema |

---

## 13. 分期实现

| 分期 | 范围 |
|------|------|
| **UI-1** | Setup 常用设置 + 开始实验 + Run/Summary；**已实现**（`operator.html` + `open_operator`） |
| **UI-2** | 配置持久化 `config/operator_defaults.json`、串口刷新、Run 锁定摘要、沿用配置确认条；**已实现** |
| **UI-3** | 滤波/陷波高级项、配置导入导出；**已实现（基础）**；GUI UDP 仍待 |
| **Save-1** | continuous/ + alignment/trial_table + verify；**已实现** |
| **Save-2** | by_phase 分节点切片；**已实现**（`save_layout=phase_folders`） |
| **Save-3** | `learn_step_*` + Marker payload 含 obj/scene/step；**已实现** |
| **Save-4** | 仅 `acquire` + 未 reject 切窗；Summary「一键 Phase4」/ `auto_phase4`；`99_summary/phase4_pointer.json`；**已实现** |
| **远期** | 极简波形、多语言、GUI UDP 旁路、`extensions` 插件位 |

---

## 14. 验收标准（相对 v0.1 补充）

1. 每次（或默认）先进入 **Setup**，未确认不能误开诱导流程。  
2. Setup 可明确选择：**是否开启采集**、**合成板 vs 真机**、**串口等设备参数**、**保存目录/是否写 eeg**、**是否按时间节点分文件夹**。  
3. 合成板调试可不开真机完成全流程；真机路径能写出 `eeg.csv`。  
4. 关闭采集时有不可忽视的警告，Summary 不宣称可训练。  
5. 开始后 Run 能对标 adapt→learn→gate→acquire 控制。  
6. 配置 schema 预留 `extensions` / `schema_version`，新增设置不必改主流程。  
7. **分文件夹**：会话根下存在 `continuous/` 与 `by_phase/01_adapt`…`06_acquire`（在启用 `phase_folders` 时）。  
8. **严格对齐**：`alignment/trial_table.csv` 含各范式边界 `t_lsl`；`verify_report` 对 MI/Rest 4s 与 EEG 覆盖区间检查通过。  
9. **记号完备**：关键边界均有 events，且 `payload` 与 LSL Marker 一致；主训练仅 `06_acquire` 且 `rejected=0`。  
10. **自动开诱导页**：Setup 点「开始实验」校验通过后，诱导页自动打开；操作台留在 Run；弹窗拦截时有补救按钮。  

---

## 15. 明确不做（本阶段仍成立）

- 在操作台重做 Three.js 诱导  
- 替代 OpenBCI GUI 完整示波  
- 多机远程、账号系统  
- 未评审通过就改范式时序默认值  
- 以「每节点停录再开录」作为对齐主路径（易产生时间缝隙）  

---

## 16. 待确认（已缩小）

| # | 问题 | 建议默认 |
|---|------|----------|
| 1 | 操作台默认「开启采集」是否同意？ | **同意：默认开采集 + 默认合成板**（防无 eeg；又不必插真机） |
| 2 | MVP 要不要波形？ | **不要**；只要状态灯 + 样本/时长 |
| 3 | 准入是否仅操作者？ | **是**（与现按键设计一致） |
| 4 | 「浏览文件夹」MVP 是否必须？ | **可先手填路径**；UI-2 再做系统选目录 |
| 5 | 保存布局默认 `phase_folders` + 连续总档？ | **同意（§8.3 / 策略 A）** |
| 6 | acquire 是否再按 trial 拆物理子文件夹？ | **默认否**（用 trial_table 索引即可；避免碎文件） |

若无异议，实现按上表默认执行。

---

## 17. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-07-23 | v0.1 初稿 |
| 2026-07-23 | v0.2：定本机网页台；强化「每次先设置」；展开采集/合成板/OpenBCI/保存分组；增加可扩展 schema 与分期 |
| 2026-07-23 | v0.3：补充 §8 数据采集与保存详细规格——按时间节点分文件夹、连续总档+切片、Marker/EEG 严格 LSL 对齐、trial_table 与 verify、Save 分期 |
| 2026-07-23 | v0.3.1：明确 Setup「开始实验」后**自动打开诱导网页**；补救「重新打开」；§6.1 标准动作 |
| 2026-07-23 | 实现：补齐 operator.js/css；Save-1/2 落盘；真机错误提示；操作台主路径文档 |
