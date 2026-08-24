"""v2 后端→前端 适配器：TrialStateMachineV2 的 on_stage 回调 → ws 广播。

用法（orchestrator v2 模式接线）：
    send = lambda msg: bridge.broadcast(msg)          # 现有 ws_bridge 的群发函数
    sm = TrialStateMachineV2(..., on_stage=make_stage_forwarder(send))
演示/联调：直接打开 web/v2_subject.html?demo=calibration 或 ?demo=game
（内置消息播放器，按同一契约回放，无需后端）。
"""

from __future__ import annotations

from typing import Callable, Optional


def make_stage_forwarder(send: Callable[[dict], None]) -> Callable:
    """把 trial_v2 的 on_stage(stage, ctx, data) 转成 {type:"v2_stage",...} 广播。

    ctx: TrialContextV2（trial_id/label/mode/round_no）；data: dict 或 None。
    trial_end 的 summary（TrialVerdict）转可序列化 dict。
    """

    def forwarder(stage: str, ctx, data) -> None:
        payload_ctx = None
        if ctx is not None and hasattr(ctx, "label"):
            payload_ctx = {
                "trial_id": getattr(ctx, "trial_id", None),
                "label": getattr(ctx, "label", None),
                "mode": getattr(ctx, "mode", None),
                "round": getattr(ctx, "round_no", None),
            }
        payload_data = data
        if data is not None and hasattr(data, "summary") and data.summary is not None:
            s = data.summary
            payload_data = {
                "summary": {
                    "label": s.label, "preds": list(s.preds),
                    "n_correct": s.n_correct, "correct": bool(s.correct),
                    "reach": s.reach, "reach_time": s.reach_time,
                }
            }
        elif isinstance(data, dict):
            payload_data = {k: v for k, v in data.items()}
            if stage == "trial_end" and "summary" in data and isinstance(data["summary"], dict):
                payload_data = {"summary": data["summary"]}
        try:
            send({"type": "v2_stage", "stage": stage, "ctx": payload_ctx, "data": payload_data})
        except Exception:  # 广播失败不阻塞范式
            pass

    return forwarder
