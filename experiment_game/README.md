# experiment_game

第一人称运动想象实验：**视觉诱导** + OpenBCI 采集/打标/落盘。

## 现行规范（2026-08-29）

| 文档 | 说明 |
|------|------|
| **[docs/框架冻结确认_20260829.md](./docs/框架冻结确认_20260829.md)** | **冻结选择题结论（F5 除外）** |
| **[docs/其他电脑移机攻略_20260829.md](./docs/其他电脑移机攻略_20260829.md)** | **换机 / 新笔记本逐步清单** |
| [docs/范式对齐_OpenBMI与fnz_v3_20260827.md](./docs/范式对齐_OpenBMI与fnz_v3_20260827.md) | OpenBMI-Align v1 时序 / Rest / 在线 |
| [docs/被试登录与按session微调方案_20260827.md](./docs/被试登录与按session微调方案_20260827.md) | 登录 / 目录 / 采后 FT |
| [docs/操作按键设计.md](./docs/操作按键设计.md) | 操作者/被试分键 |
| [docs/完整使用流程.md](./docs/完整使用流程.md) | 安装到开实验（部分章节仍偏旧 Phase，以冻结表为准） |

**现行默认**：E1f 四成员 · Align 时序（专用 Rest4 → prep2 → MI4）· 采后 Leave-Next · 块间默认 30s。

## 其它文档

| 文档 | 说明 |
|------|------|
| [项目计划.md](./项目计划.md) | 早期总计划（多处已过时） |
| [docs/游戏内容规格.md](./docs/游戏内容规格.md) | 画面与流程 |
| [docs/marker_spec.md](./docs/marker_spec.md) | **Phase1 历史** Marker 表 |
| [docs/ws_protocol.md](./docs/ws_protocol.md) | WebSocket 协议 |

依赖：仓库根 `.venv` 安装本目录 `requirements.txt`（含 `websockets`、`scipy`）。

## 操作台（推荐主路径 · 真机采数）

双击 [`open_operator.bat`](./open_operator.bat)，浏览器打开 **Setup**：

1. **登录被试缩写**（如 `syj0828`）  
2. 勾选「开启采集」· 选 **Cyton** 或合成板 · 填 COM  
3. 模式选 **v3 探针**（可选先跑 **v4 帽检**，不强制）  
4. 权重默认 **E1f 四成员**（可选手模对照）  
5. 点「开始实验」→ 诱导页 + 操作台 Run  

合成板联调：板模式选「合成板」；动觉引导会 **自动确认**（无需再点）。

```powershell
cd d:\MI
.\.venv\Scripts\python.exe -m experiment_game.tools.open_operator
```

会话输出示例：

```text
data/sessions/<subject>_<session>_<stamp>/
  eeg.csv / events.jsonl / session.meta.json / run_config.json
  continuous/
  by_phase/01_adapt … 06_acquire/
  alignment/trial_table.csv / verify_report.json
  99_summary/phase4_pointer.json   ← 一键 Phase4 后写入
  manifest.json
```

Summary 页可点「一键 Phase4 切窗」（仅 `acquire` + 未 reject），输出到 `data/epochs/<会话名>/`。Setup 也可勾选「结束后自动 Phase4」。

## Phase 1（无画面联调）

```powershell
cd d:\cyy\MI

.\.venv\Scripts\python.exe `
  -m experiment_game.tools.run_phase1_block --trials 20 --yes

.\.venv\Scripts\python.exe `
  -m experiment_game.tools.verify_phase1_alignment `
  --session experiment_game\data\sessions\<会话目录名>
```

## Phase 2（诱导 MVP）

**一键打开网页（推荐）**：双击 `experiment_game/open_induction.bat`（或 `打开诱导页.bat`）  
或在仓库根执行：

```powershell
cd d:\cyy\MI
.\.venv\Scripts\python.exe `
  -m experiment_game.tools.open_induction
```

默认只跑画面（不采 EEG），浏览器会打开 `http://127.0.0.1:8080/`。  
需要同时采数时：

```powershell
... -m experiment_game.tools.open_induction --with-acq --acquire-trials 4
```

完整参数也可直接调用：

```powershell
.\.venv\Scripts\python.exe `
  -m experiment_game.tools.run_phase2_session --yes --acquire-trials 4
```

常用参数：

| 参数 | 含义 |
|------|------|
| `--with-acq`（仅 open_induction） | 开启脑电采集 |
| `--no-acq` | 只跑画面，不采 EEG |
| `--fast` | 缩短试次时长（仅联调） |
| `--auto-continue` | 无人值守（跳过页面点击，联调用） |
| `--skip-adapt` / `--skip-learn` / `--skip-gate` | 跳过对应阶段 |
| `--no-rotate-objects` | 正式段固定单物品 |
| `--no-rotate-scenes` | 正式段固定单场景 |
| `--real --port COMx` | 真机 Cyton |

诱导页默认：`http://127.0.0.1:8080/`（需能访问 unpkg 加载 Three.js）。

输出仍在 `data/sessions/...`（`eeg.csv` + `events.jsonl` + `session.meta.json`）。

## Phase 4（切窗 → 训练）

```powershell
cd d:\cyy\MI
.\.venv\Scripts\python.exe `
  -m experiment_game.tools.run_phase4_epochs `
  --session experiment_game\data\sessions\<会话目录名>
```

详见 [docs/Phase4_流水线检查清单.md](./docs/Phase4_流水线检查清单.md)。
输出：`data/epochs/<会话名>/`（`X.npy`、`y_task.npy`、`y_three.npy` + train/val）。
