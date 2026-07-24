# WebSocket 协议（Phase 3 + 操作台）

控制器权威计时；浏览器只渲染。默认 `ws://127.0.0.1:8765`。

## 服务端 → 客户端（诱导页 / 共用）

| type | 字段 | 说明 |
|------|------|------|
| `hello` | `message` | 连接成功（不进入 pending，避免循环） |
| `stage` | `phase`, `stage`, `trial_id`, `label`, `hand`, `anim`, `duration_s`, `object`, `scene`, `learn_step`, `transition_amp` | 阶段切换 |
| `hud` | `text`, `show_cross`, `subtext` | 覆盖层文案 |
| `prompt` | `id`, `title`, `body`, `button`, `allow_subject` | 等待确认；`allow_subject=false` 时仅操作者可确认 |
| `session` | `status`, `phase?`, `message?` | `running` / `gate` / `done` / `error` |
| `operator_state` | `paused`, `phase`, `trial_id`, `label`, `object`, `scene`, `reject_count`, `aborting` | 操作者条状态 |

`phase`: `adapt` | `learn` | `acquire`  
`stage`: `idle` | `fixation` | `cue` | `mi` | `post_mi_hold` | `rest` | `transition` | `trial_start` | `trial_end`  
`hand`: `left` | `right` | `none`  
`anim`: `none` | `full_grasp` | `reach`  
`transition_amp`: `micro` | `swap` | `scene`  
`object`: `cup` | `bottle` | `apple`  
`scene`: `home_desk` | `hospital_desk` | `school_desk`

## 服务端 → 客户端（操作台）

| type | 字段 | 说明 |
|------|------|------|
| `operator_hello` | `defaults`, `builtin_defaults`, `defaults_path`, `serial_ports`, `subject_url` | 连接后下发文件默认配置与串口列表 |
| `config_ack` | `ok`, `errors[]`, `run_config?`, `starting?` | 校验结果；`starting=true` 表示即将开跑 |
| `session_started` | `session_root`, `acq_enabled`, `board_mode`, … | 会话已创建，设置锁定摘要 |
| `acq_status` | `state`, `message` | `idle` / `connecting` / `recording` / `error` / `stopped` |
| `serial_ports` | `ok`, `ports[{device,description}]` | 串口刷新结果 |
| `save_defaults_ack` | `ok`, `message`, `path?`, `run_config?` | 写入 `config/operator_defaults.json` |
| `session_saved` | `root`, `files[]`, `verify`, `train_eligible` | 结束 Summary |
| `open_folder_ack` | `ok`, `path`, `message?` | 打开文件夹回执 |
| `subject_page_opened` | `ok`, `url` | 后端打开诱导页回执 |

## 客户端 → 服务端

| type | 说明 |
|------|------|
| `ready` | 页面加载完成 |
| `continue` | 响应 prompt（可带 `role: subject\|operator`） |
| `sync` | 请求重放 pending |
| `operator` | `{action: pause\|resume\|toggle_pause\|reject\|abort\|gate_ok\|continue}` |
| `ping` | 心跳 |
| `operator_hello` | 操作台握手，拉取默认配置 |
| `config_validate` | `{run_config}` 仅校验 |
| `session_start` | `{run_config}` 校验并通过后开跑 |
| `list_serial_ports` | 刷新 COM 列表 |
| `save_defaults` | `{run_config}` 写入本地默认 JSON |
| `run_phase4` | `{path}` 对会话目录切窗（acquire + 未 reject） |
| `open_subject_page` | 补救打开诱导页 |
| `open_folder` | `{path}` 打开会话目录 |

另有服务端回执：`phase4_ack`（`ok`, `epochs_dir`, `summary`, `pointer`）。

## 原则

- 阶段到达时刻以 Python `pylsl.local_clock()` 打标为准；WS 仅同步 UI。  
- 正式 `acquire` 的 `mi`/`rest` 必须 `anim=none`。  
- 暂停时控制器冻结 trial 时钟；reject 在 `trial_end` 前写入 `trial_reject`。
