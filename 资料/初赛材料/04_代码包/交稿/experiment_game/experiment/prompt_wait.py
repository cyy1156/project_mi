"""共享 prompt / continue 等待（phase2 适应、v2 环境说明等）。"""

from __future__ import annotations

from typing import Callable, Optional

from pylsl import local_clock

from experiment_game.experiment.trial_sm import SessionAbort, wait_until
from experiment_game.experiment.ws_bridge import WsBridge

OnConsole = Callable[[str], None]


def wait_prompt_continue(
    bridge: WsBridge,
    on_console: OnConsole,
    *,
    prompt_id: str,
    title: str,
    body: str,
    button: str,
    allow_subject: bool = True,
    timeout_s: float = 600.0,
    auto_continue: bool = False,
    after_broadcast: Optional[Callable[[], None]] = None,
) -> None:
    """广播 prompt 并等待 continue / gate_ok（被试或操作者）。"""
    bridge.clear_event("continue")
    bridge.clear_event("gate_ok")
    prompt = {
        "type": "prompt",
        "id": prompt_id,
        "title": title,
        "body": body,
        "button": button,
        "allow_subject": allow_subject,
    }
    bridge.broadcast(prompt)
    if after_broadcast is not None:
        after_broadcast()
    who = "被试或操作者" if allow_subject else "操作者（G / 代确认 / 点按钮）"
    on_console(f"[prompt] {title} — 等待{who}「{button}」…")
    if auto_continue:
        bridge.clear_pending_prompt()
        wait_until(local_clock() + 0.3)
        return

    deadline = local_clock() + float(timeout_s)
    last_rebroadcast = local_clock()
    while local_clock() < deadline:
        if bridge.should_abort():
            raise SessionAbort("operator abort")
        if bridge.is_paused():
            if bridge.wait_client_event("continue", timeout=0.2):
                bridge.clear_pending_prompt()
                bridge.clear_event("continue")
                bridge.clear_event("gate_ok")
                return
            if bridge.wait_client_event("gate_ok", timeout=0.05):
                bridge.clear_pending_prompt()
                bridge.clear_event("continue")
                bridge.clear_event("gate_ok")
                return
            continue
        if bridge.wait_client_event("continue", timeout=0.4):
            bridge.clear_pending_prompt()
            bridge.clear_event("continue")
            bridge.clear_event("gate_ok")
            return
        if bridge.wait_client_event("gate_ok", timeout=0.1):
            bridge.clear_pending_prompt()
            bridge.clear_event("continue")
            bridge.clear_event("gate_ok")
            return
        now = local_clock()
        if now - last_rebroadcast >= 8.0:
            if not (
                bridge._client_events["continue"].is_set()
                or bridge._client_events["gate_ok"].is_set()
            ):
                bridge.broadcast(prompt)
                last_rebroadcast = now

    bridge.clear_pending_prompt()
    raise TimeoutError(f"等待 continue 超时: {prompt_id}")


ENV_ADAPT_BODY = (
    "你将看到第一人称双手与桌面上的目标物。任务是：根据提示，在脑中想象用左手或右手去抓取。"
    "实验过程请尽量身体静止、不要真实动手。"
)
