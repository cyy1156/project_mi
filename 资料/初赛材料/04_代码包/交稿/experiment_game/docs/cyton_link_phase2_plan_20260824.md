# Cyton 链路二期方案：会话前探针 · 运行中自动重连 · 操作台链路面板

日期：2026-08-24
上游基础：`collect_data/.../lsl_connect/cyton_link.py`（四层模型 L1–L4 + 连接期预检/重试/分类提示，已落地）
本文范围：**连接期之后**的三块能力。只出方案，未改代码；确认后按 F1→F2→F3 实施。

---

## 0. 现状定位

`cyton_link.py` + `board.py connect()` 已解决**连接期**问题（release → probe v → ensure idle → BF prepare → 样本验证 → 分类重试），并有 `CytonConnectReport` / `LinkFailureKind` / `format_link_failure` 三件套。

仍缺的三块（对应「还没做」清单）：

| # | 能力 | 现状缺口 | 优先级 |
|---|---|---|---|
| F1 | 会话前探针 | v3/v2 启动直接走 `start_acquisition()`，链路死时要等 BF 超时+重试+health_check 重试（10–20s）才报错，且报错路径长 | P0 |
| F2 | 运行中 stall 自动重连 | `acquisition_work.py:187` `_run_loop` 里 `fetch_new_batch()` 空返回只 sleep 5ms 继续转——无线中途断流后**状态永远 RUNNING、samples_pushed 冻结**，无人知晓、无人恢复 | P1 |
| F3 | 操作台链路面板 | `last_connect_report`、gap、重连事件都躺在后端，操作台只有一个 `st-acq` 文本（operator.html:339） | P1 |

非目标：不改 LSL inlet 重解析（重连**保留原 outlet**，见 F2）；不做会话自动中止；合成板全部旁路。

---

## 1. F1 会话前探针（P0，≈0.5h）

**目的**：链路死时在 2s 内拒跑并给出 L1/L2 中文指引，不浪费 10–20s 在 BF 超时上；与 v3「硬校验拒跑」哲学一致。

**改动点**：

1. `experiment_game/acquisition/service.py` 加 `AcquisitionFacade.preflight_probe() -> Dict`：
   - `ensure_lsl_connect_on_path()` 后调 `probe_cyton_version(serial_port)`；
   - 合成板直接返回 `{"ok": True, "skipped": True}`；
   - 返回 `{ok, failure_kind, firmware_line, guidance}`，guidance = `format_link_failure(kind, port, probe=probe)`。
2. `orchestrator.py` 三个启动路径（:573 v1/v2、:880、:1157 v3）在 `AcquisitionFacade(...)` 之后、`create()` 之前调用：
   - 失败 → `_emit_acq_status("error", guidance)` + `raise RuntimeError(guidance)`（v3 本就拒跑；v2 同样拒跑，避免半开会话）；
   - 成功 → 把 `firmware_line` 存进 `SessionMeta`/`update_session_meta`（事后审计用）。
3. 不改 `health_check`（它仍负责「连上但推数慢」的 1–2s 预热判定）。

**验收**：
- 拔 dongle 开 v3 → ≤3s 报「无法打开串口…①②④」，无 BF 超时等待；
- dongle 在、Cyton 关机 → 报「无线未配对或板子未开机」指引；
- 正常开机 → 正常进入，meta.json 含 firmware 行。

---

## 2. F2 运行中 stall 检测 + 自动重连（P1，核心≈3h）

### 2.1 检测点：`AcquisitionWorker._run_loop` 内联（不另开监控线程）

循环本就 5ms 一转，空返回时顺手做 stall 判定，零线程协调成本：

```python
# acquisition_work.py _run_loop 内
now = time.monotonic()
if data.shape[1] > 0:
    self._last_data_at = now
elif (
    self._stall_enabled            # 真机才开；合成板 False
    and not self._stop_event.is_set()
    and now - self._last_data_at > self._stall_threshold_sec
):
    self._handle_stall(now)        # 内部走重连序列；期间循环被占用，天然互斥
```

参数进 `AcquisitionConfig`（或 BoardConfig，二选一，建议 **AcquisitionConfig**——它是循环参数）：

```python
stall_detect_enabled: bool = True      # start() 里与 use_synthetic 取与
stall_threshold_sec: float = 4.0       # 250Hz=1000 样本；Cyton 板载缓冲可吸收 <2s 瞬断，4s 才判 stall
reconnect_max_attempts: int = 3
reconnect_cooldown_sec: float = 5.0    # 一轮失败后冷却，避免热循环烧 COM
```

### 2.2 重连序列（全部复用现有件，不新写握手）

```
_handle_stall:
  记 stall 事件 {at, gap_s}
  for attempt in 1..reconnect_max_attempts:
      board.stop_stream_only()          # board.py:238，打断阻塞
      board.disconnect()                # board.py:249，含 force_release_all
      probe = probe_cyton_version(port) # L2 快速判死：probe 不过就不碰 BF，省 15s 超时
      if not probe.ok: 记事件(kind=probe.failure_kind)；sleep(cooldown)；continue
      board.connect()                   # board.py:102 完整链路：release→probe→idle→BF→样本验证
      self._ts_mapper.reset()           # 板卡时间戳归零，必须重锚（start() 里就用它）
      self._last_data_at = monotonic()
      记事件(ok=True, attempt)；return
  记事件(ok=False)；self._link_dead = True   # 上限后不再尝试，循环继续空转（无害）
```

**关键约束**：

- **LSL outlet 不重建**（`_outlet_eeg/_outlet_accel` 保持原对象）。outlet 与板卡无绑定；重建会迫使 `RecorderWorker` 的 inlet 重新 resolve，复杂度外溢。重连后时间戳跳变 → recorder 的 `_record_gap`（recorder_worker.py:299）自动把缺口计入 `estimated_gap_samples`——**缺口度量零改动复用**。
- 重连只发生在采集线程内，`stop()` 天然串行（stop 先 set 事件再 stop_stream_only；`_handle_stall` 每个 sleep 前查 stop_event）。
- 事件列表 `self._link_events: List[Dict]` + `get_link_stats()` 暴露：`{stall_count, reconnect_ok, reconnect_fail, link_dead, last_event, events[-10:]}`。

### 2.3 与下游门控的协同（已建好的安全网）

重连缺口期间的数据是**空洞**，不是垃圾：

- v3：判定窗空洞 → `assess_eeg_window` 判 flatline/low_variance → `signal_bad` → 试次已被 A3 链路剔除（特征卡显示「信号质量不足」）；
- v2：`InferenceService.judge` 同样走 signal_quality 门控，return `signal_bad`，不污染计分/微调；
- 因此 F2 **不需要**暂停试次状态机——让门控吃掉缺口期试次，重连成功后自动恢复有效数据。这是本方案最重要的简化。

### 2.4 事件上行（供 F3 与日志）

- `ServiceManager.get_status()`（service_manager.py:231）追加：`link_stats`（worker.get_link_stats()）+ `last_connect_report` 摘要（attempts、firmware、failure_kind、message 首行）；
- `AcquisitionWorker` 构造时接收可选 `on_link_event(cb)`，`ServiceManager.start_acquisition` 注入 → `_log_warn("[链路] stall 6.2s，重连 1/3…")` 进 event_bus；
- marker：orchestrator 见到 `stall`/`reconnect_ok` 事件时 `markers.push(f"acq_reconnect|attempt={n}|gap_s={x}|ok={0/1}")`，离线可对齐。

### 2.5 上限与降级

- 3 次失败 → `link_dead=True`：orchestrator 广播 `v3_warn`/`acq_status error`「无线断流，自动重连失败：① Cyton 电量 ② 重开机 ③ dongle 距离」；会话**不自动中止**（v3 全试次被门控剔除，报告 n=0 自洽；是否中止由操作者按 Esc）。
- 重连成功但 60s 内再次 stall → 冷却翻倍（防「半死不活」链路抖动风暴），上限 30s。

### 2.6 验收（真机）

1. 会话进行中关 Cyton 电源 → 操作台 4–6s 内出现「stall…重连 1/3」；CSV `estimated_gap_samples ≈ 停摆秒数×250`；
2. 重开电源 → ≤2 轮内 `reconnect_ok`，samples_pushed 恢复增长，后续试次特征卡正常；
3. 关电源不重开 → 3 次后 `link_dead` 横幅；期间试次卡全部「信号质量不足」；
4. 合成板会话 → stall 逻辑完全旁路（`stall_detect_enabled and not use_synthetic`）。

---

## 3. F3 操作台链路面板（P1，≈2h）

### 3.1 后端：orchestrator 2s 心跳广播 `link_status`

会话期起一个 daemon 监控线程（与 eeg publisher 同模式，finally 里 stop）：

```python
每 2s:
    st = facade.manager.get_status()          # 已含 samples_pushed / link_stats / last_connect 摘要
    hz = (pushed - prev_pushed) / 2.0          # streaming_hz 用 Δpushed/Δt，不用 drop_rate
    bridge.broadcast({"type": "link_status",
        "port", "firmware", "state", "worker_running",
        "streaming_hz", "samples_pushed",
        "gap_samples": recorder.estimated_gap_samples,   # recorder get_stats()
        "reconnect_ok", "reconnect_fail", "link_dead",
        "last_event": {...}, "guidance": 仅 link_dead/失败时带首行指引})
```

口径纪律（写进面板脚注，沿用你们的一句话总结）：**drop_rate_pct 是 LSL 写入统计，只配评价录制质量；链路好坏看 streaming_hz + gap + 重连计数**。

### 3.2 前端：run 视图加一行链路卡

operator.html 状态区（:339 `st-acq` 旁）加：

```html
<div><span class="k">链路</span> <span id="link-line">—</span></div>
```

operator.js `link_status` handler 渲染单行（紧凑，不新开面板）：

```
COM3 · v3.1.5 · 249Hz · gap 0 · 重连 0
```

着色规则（复用 stat-ok/mid/bad）：
- `link_dead` 或 hz<100 → bad + 行尾追加指引首行；
- hz<200 或 reconnect_fail>0 → mid；
- 其余 ok。重连成功过（reconnect_ok>0）→ mid 并显示计数，提醒本场有过断流。

### 3.3 验收

- 面板 hz 与 `com3_probe` / CSV 跨度核算一致（±5%）；
- F2 演练时面板实时变 mid→bad→ok，无需刷新；
- v2/v3 共用（handler 不判模式）。

---

## 4. 实施顺序与工时

| 步 | 内容 | 工时 | 依赖 |
|---|---|---|---|
| 1 | F1 preflight + orchestrator 三路径 + meta firmware | 0.5h | 无 |
| 2 | F2 worker stall/重连 + get_status 扩展 + marker | 3h | 无 |
| 3 | F3 心跳广播 + 前端链路行 | 2h | F2 事件字段 |
| 4 | 真机验收（拔/关/开 三幕） | 1h | 1–3 |

F1 可独立先上（收益/风险比最高）；F2 是核心；F3 让 F2 可见，可与 F2 同批。

## 5. 风险与边界

- 重连期间 COM 被其它进程抢（GUI 又开了）→ probe 分类 PORT_BUSY，指引里已有「关 GUI」；
- `fetch_new_batch` 在 disconnect 后调用会抛 → `_handle_stall` 内 try/except 全包，事件记 `unknown`；
- Windows 串口释放慢：disconnect 已含 `force_release_all(0.25)`，connect 首轮前再 release——沿用现有等待，不新加 sleep 玄学；
- 不追求「无感重连」：缺口必须可见（gap 计数 + marker + 面板），宁可显眼不可沉默。
