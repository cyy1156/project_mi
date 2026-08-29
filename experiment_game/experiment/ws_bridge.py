"""WebSocket 广播桥：控制器 → 浏览器。"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable, Optional, Set

import websockets

from experiment_game.experiment.trial_sm import SessionAbort


class WsBridge:
    """
    在后台线程跑 asyncio WebSocket 服务。
    主线程用 broadcast() 推消息；wait_client_event() 等浏览器 continue/ready。
    新连接会重放 pending 的 prompt/stage/hud/session_saved，避免刷新后卡死。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: Set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._client_events: dict[str, threading.Event] = {
            "ready": threading.Event(),
            "continue": threading.Event(),
            "abort": threading.Event(),
            "gate_ok": threading.Event(),
            "split_request": threading.Event(),
            "v2_guidance_confirm": threading.Event(),
        }
        self._on_message: Optional[Callable[[dict], None]] = None
        self._server = None
        self._pending: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.paused = False
        self.reject_requested = False
        self._operator_hook: Optional[Callable[[str, dict], None]] = None

    def set_operator_hook(self, hook: Optional[Callable[[str, dict], None]]) -> None:
        self._operator_hook = hook

    def set_on_message(self, hook: Optional[Callable[[dict], None]]) -> None:
        self._on_message = hook

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def set_pending(self, message: dict[str, Any]) -> None:
        with self._lock:
            mtype = message.get("type")
            if mtype == "prompt":
                self._pending = [m for m in self._pending if m.get("type") != "prompt"]
                self._pending.append(dict(message))
            elif mtype == "questionnaire":
                # 只保留最后一份未提交问卷；提交后由 clear_pending_questionnaire 清除
                self._pending = [
                    m for m in self._pending if m.get("type") != "questionnaire"
                ]
                self._pending.append(dict(message))
            elif mtype in ("stage", "hud", "session", "operator_state", "session_saved", "session_finishing"):
                # session_saved / session_finishing：落盘/对齐期间可能断线，重连须能收到终态
                self._pending = [m for m in self._pending if m.get("type") != mtype]
                # session_saved 到达后不再需要 finishing 过渡态
                if mtype == "session_saved":
                    self._pending = [
                        m for m in self._pending if m.get("type") != "session_finishing"
                    ]
                self._pending.append(dict(message))

    def clear_pending_questionnaire(self) -> None:
        with self._lock:
            self._pending = [
                m for m in self._pending if m.get("type") != "questionnaire"
            ]

    def clear_pending_prompt(self) -> None:
        with self._lock:
            self._pending = [m for m in self._pending if m.get("type") != "prompt"]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run, name="ws-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("WebSocket 服务启动超时")
        if self._start_error is not None:
            self.stop()
            err = self._start_error
            hint = ""
            winerr = getattr(err, "winerror", None)
            errno = getattr(err, "errno", None)
            # Windows 10048 / Linux 98 / macOS 48 = address already in use
            if winerr == 10048 or errno in (98, 48, 10048):
                hint = (
                    f"（端口 {self.host}:{self.port} 已被占用；"
                    "请关闭旧操作台窗口，或换 --ws-port）"
                )
            raise RuntimeError(
                f"WebSocket 服务启动失败: {err}{hint}"
            ) from err

    def stop(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            self._thread = None
            self._loop = None
            return
        try:
            if loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                try:
                    fut.result(timeout=3.0)
                except Exception:  # noqa: BLE001
                    pass
                loop.call_soon_threadsafe(loop.stop)
        except Exception:  # noqa: BLE001
            pass
        thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        self._server = None
        self._clients.clear()

    def broadcast(self, message: dict[str, Any]) -> None:
        if message.get("type") in (
            "prompt",
            "questionnaire",
            "stage",
            "hud",
            "session",
            "operator_state",
            "session_saved",
            "session_finishing",
        ):
            self.set_pending(message)
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)

            def _log_err(f: Any) -> None:
                try:
                    exc = f.exception()
                except Exception:  # noqa: BLE001
                    return
                if exc is not None:
                    print(f"[ws] broadcast 失败 type={message.get('type')}: {exc!r}")

            fut.add_done_callback(_log_err)
        except Exception as exc:  # noqa: BLE001
            print(f"[ws] broadcast 调度失败 type={message.get('type')}: {exc!r}")

    def clear_pending_session_saved(self) -> None:
        with self._lock:
            self._pending = [
                m
                for m in self._pending
                if m.get("type") not in ("session_saved", "session_finishing")
            ]

    def clear_event(self, name: str) -> None:
        ev = self._client_events.get(name)
        if ev is not None:
            ev.clear()

    def set_event(self, name: str) -> None:
        """供编排器主动触发客户端事件（如换场等待中的第二次 B）。"""
        ev = self._client_events.get(name)
        if ev is not None:
            ev.set()

    def wait_client_event(self, name: str, timeout: Optional[float] = None) -> bool:
        """等待客户端事件；每 50ms 轮询一次，期间响应 abort。"""
        ev = self._client_events.get(name)
        if ev is None:
            raise KeyError(name)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            if self.should_abort():
                raise SessionAbort("operator abort")
            if deadline is None:
                slice_s = 0.05
            else:
                slice_s = min(0.05, max(0.0, deadline - time.monotonic()))
                if slice_s <= 0:
                    return False
            if ev.wait(timeout=slice_s):
                return True

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _boot() -> None:
            try:
                await self._serve()
            except BaseException as exc:  # noqa: BLE001
                self._start_error = exc
            finally:
                self._ready.set()

        loop.create_task(_boot())
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _serve(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            # 允许局域网监控端跨主机连接（否则偶发握手被拒）
            origins=None,
        )

    async def _shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()

    async def _send_pending(self, ws: Any, *, skip_prompt: bool = False) -> None:
        with self._lock:
            msgs = [dict(m) for m in self._pending]
        for message in msgs:
            if skip_prompt and message.get("type") == "prompt":
                continue
            try:
                await ws.send(json.dumps(message, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                break

    async def _handler(self, ws: Any) -> None:
        self._clients.add(ws)
        # 新连接先打招呼（不进 pending，避免 ready↔hello 死循环重置动画）
        try:
            await ws.send(
                json.dumps(
                    {"type": "hello", "message": "connected"},
                    ensure_ascii=False,
                )
            )
        except Exception:  # noqa: BLE001
            pass
        await self._send_pending(
            ws, skip_prompt=self._client_events["continue"].is_set()
        )
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "continue":
                    self.clear_pending_prompt()
                    self._client_events["continue"].set()
                    try:
                        await ws.send(
                            json.dumps({"type": "continue_ack"}, ensure_ascii=False)
                        )
                    except Exception:  # noqa: BLE001
                        pass
                elif mtype in self._client_events:
                    self._client_events[mtype].set()
                if mtype == "operator":
                    self._handle_operator(msg)
                if mtype in ("ready", "sync"):
                    # continue 已确认后不再重放 prompt（避免点击后弹窗又弹回）
                    if not self._client_events["continue"].is_set():
                        await self._send_pending(ws)
                if self._on_message is not None:
                    self._on_message(msg)
        finally:
            self._clients.discard(ws)

    def _handle_operator(self, msg: dict[str, Any]) -> None:
        action = str(msg.get("action") or "")
        if action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "toggle_pause":
            self.paused = not self.paused
        elif action == "reject":
            self.reject_requested = True
        elif action == "abort":
            self._client_events["abort"].set()
            self.paused = False
            self.broadcast({
                "type": "session",
                "status": "aborting",
                "message": "正在中止会话…",
            })
        elif action == "gate_ok":
            self.clear_pending_prompt()
            self._client_events["gate_ok"].set()
            self._client_events["continue"].set()
        elif action == "continue":
            self.clear_pending_prompt()
            self._client_events["continue"].set()
        if self._operator_hook is not None:
            try:
                self._operator_hook(action, msg)
            except Exception:  # noqa: BLE001
                pass

    def clear_reject(self) -> None:
        self.reject_requested = False

    def is_paused(self) -> bool:
        return bool(self.paused)

    def should_abort(self) -> bool:
        return self._client_events["abort"].is_set()

    def is_rejected(self) -> bool:
        return bool(self.reject_requested)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return
        try:
            data = json.dumps(message, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"JSON 序列化失败 type={message.get('type')}: {exc}") from exc
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


def hand_from_label(label: Optional[int]) -> str:
    if label == 1:
        return "left"
    if label == 2:
        return "right"
    return "none"
