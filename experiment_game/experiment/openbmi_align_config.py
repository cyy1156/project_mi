"""OpenBMI-Align v1 共享常量（v2 / v3 采集 · 在线 · 离线切窗）。"""

from __future__ import annotations

from typing import Any, List, Protocol, Tuple

WIN_S = 3.0
HOP_S = 0.1
BASELINE_BEFORE_CUE_S = 0.5
MI_TASK_SEC_DEFAULT = 4.0

ONLINE_WINDOW_MODE_OPENBMI = "openbmi_hop100"
ONLINE_WINDOW_MODE_D8 = "d8_grid"


class _JudgmentTimesCfg(Protocol):
    online_window_mode: str
    imagine_s: float
    win_s: float
    win_hop_s: float
    judgment_step_s: float
    judgment_times: tuple


def build_openbmi_judgment_times(
    imagine_s: float,
    *,
    win_s: float = WIN_S,
    hop_s: float = HOP_S,
) -> Tuple[float, ...]:
    """判定点 = 3s 前向窗的窗尾（相对 Cue）。MI=4s → 11 档：3.0…4.0。"""
    if float(imagine_s) < float(win_s) - 1e-9:
        return ()
    out: List[float] = []
    t_end = float(win_s)
    hop = float(hop_s)
    while t_end <= float(imagine_s) + 1e-9:
        out.append(round(t_end, 6))
        t_end = round(t_end + hop, 6)
    return tuple(out)


def build_d8_judgment_times(step_s: float, imagine_s: float) -> Tuple[float, ...]:
    out: List[float] = []
    t = float(step_s)
    while t <= float(imagine_s) + 1e-9:
        out.append(round(t, 6))
        t += float(step_s)
    return tuple(out)


def rebuild_judgment_times(cfg: _JudgmentTimesCfg) -> None:
    mode = getattr(cfg, "online_window_mode", ONLINE_WINDOW_MODE_OPENBMI)
    if mode == ONLINE_WINDOW_MODE_OPENBMI:
        cfg.judgment_times = build_openbmi_judgment_times(
            cfg.imagine_s,
            win_s=getattr(cfg, "win_s", WIN_S),
            hop_s=getattr(cfg, "win_hop_s", HOP_S),
        )
    else:
        step = float(getattr(cfg, "judgment_step_s", 0.6))
        cfg.judgment_times = build_d8_judgment_times(step, cfg.imagine_s)


def openbmi_timing_field_defaults() -> dict[str, Any]:
    return {
        "prep_s": 2.0,
        "cue_s": 0.0,
        "imagine_s": MI_TASK_SEC_DEFAULT,
        "iti_s": 3.0,
        "inter_trial_rest_s": 4.0,
        "online_window_mode": ONLINE_WINDOW_MODE_OPENBMI,
        "win_s": WIN_S,
        "win_hop_s": HOP_S,
        "baseline_before_cue_s": BASELINE_BEFORE_CUE_S,
        "primary_judge_mode": "majority",
    }
