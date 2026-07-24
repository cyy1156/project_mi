# experiment_game

第一人称运动想象实验：**视觉诱导** + OpenBCI 采集/打标/落盘。

## 文档

| 文档 | 说明 |
|------|------|
| [项目计划.md](./项目计划.md) | 总计划与阶段 |
| [docs/游戏内容规格.md](./docs/游戏内容规格.md) | 画面与流程定稿 |
| [docs/marker_spec.md](./docs/marker_spec.md) | 事件 / Marker / 切窗 |
| [docs/ws_protocol.md](./docs/ws_protocol.md) | 前后端 WebSocket 协议 |
| [docs/操作按键设计.md](./docs/操作按键设计.md) | 操作者/被试分键（Phase 3 已实现） |
| [docs/Phase4_流水线检查清单.md](./docs/Phase4_流水线检查清单.md) | 切窗 → 训练检查清单 |
| [docs/完整使用流程.md](./docs/完整使用流程.md) | **从安装到训练的逐步操作手册** |

依赖：在 `lsl_connect` 的 venv 中安装本目录 `requirements.txt`（含 `websockets`、`scipy`）。

## 操作台（推荐主路径 · 真机采数）

双击 [`open_operator.bat`](./open_operator.bat)，浏览器打开 **Setup**：

1. 勾选「开启采集」  
2. 选 **Cyton 真机**，串口填设备管理器中的 COM（默认模板 COM5）  
3. **先关闭 OpenBCI GUI 串口直播**（避免抢 COM）  
4. 正式 trial 首测可改为 4～8；可勾选跳过适应/学习  
5. 落盘选 `phase_folders`（默认）  
6. 点「开始实验」→ 自动打开诱导页 → 操作台留在 Run 控场  

**UI-2：**「保存为本地默认」写入 `config/operator_defaults.json`；真机可点「刷新串口」；勾选「下次显示沿用确认条」。

合成板联调：板模式选「合成板」即可。

```powershell
cd d:\cyy\MI
.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
  -m experiment_game.tools.open_operator
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

.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
  -m experiment_game.tools.run_phase1_block --trials 20 --yes

.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
  -m experiment_game.tools.verify_phase1_alignment `
  --session experiment_game\data\sessions\<会话目录名>
```

## Phase 2（诱导 MVP）

**一键打开网页（推荐）**：双击 `experiment_game/open_induction.bat`（或 `打开诱导页.bat`）  
或在仓库根执行：

```powershell
cd d:\cyy\MI
.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
  -m experiment_game.tools.open_induction
```

默认只跑画面（不采 EEG），浏览器会打开 `http://127.0.0.1:8080/`。  
需要同时采数时：

```powershell
... -m experiment_game.tools.open_induction --with-acq --acquire-trials 4
```

完整参数也可直接调用：

```powershell
.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
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
.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
  -m experiment_game.tools.run_phase4_epochs `
  --session experiment_game\data\sessions\<会话目录名>
```

详见 [docs/Phase4_流水线检查清单.md](./docs/Phase4_流水线检查清单.md)。
输出：`data/epochs/<会话名>/`（`X.npy`、`y_task.npy`、`y_three.npy` + train/val）。
