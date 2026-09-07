"""试次时序（秒）— 与 docs/marker_spec.md 一致；时长可在操作台配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TrialTiming:
    fixation_s: float = 2.0
    cue_s: float = 2.0
    mi_s: float = 4.0
    post_mi_hold_s: float = 1.0
    rest_s: float = 4.0
    transition_s: float = 3.0

    @property
    def total_s(self) -> float:
        return (
            self.fixation_s
            + self.cue_s
            + self.mi_s
            + self.post_mi_hold_s
            + self.rest_s
            + self.transition_s
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "fixation_s": self.fixation_s,
            "cue_s": self.cue_s,
            "mi_s": self.mi_s,
            "post_mi_hold_s": self.post_mi_hold_s,
            "rest_s": self.rest_s,
            "transition_s": self.transition_s,
        }

    @property
    def segments(self) -> List[Dict[str, Any]]:
        """按时间顺序的阶段构成（用于界面时间轴展示）。"""
        return [
            {"key": "fixation", "zh": "注视", "s": self.fixation_s, "train": False},
            {"key": "cue", "zh": "提示", "s": self.cue_s, "train": False},
            {"key": "mi", "zh": "运动想象", "s": self.mi_s, "train": True},
            {"key": "post_mi_hold", "zh": "保持", "s": self.post_mi_hold_s, "train": False},
            {"key": "rest", "zh": "静息", "s": self.rest_s, "train": True},
            {"key": "transition", "zh": "过渡", "s": self.transition_s, "train": False},
        ]


DEFAULT_TIMING = TrialTiming()

# 仅用于联调/界面验收，不用于正式采数
FAST_TIMING = TrialTiming(
    fixation_s=0.6,
    cue_s=0.8,
    mi_s=1.6,
    post_mi_hold_s=0.4,
    rest_s=0.8,
    transition_s=0.6,
)

# 各段允许范围：(最小, 最大)。MI/Rest 是训练窗来源，下限更严。
TIMING_BOUNDS: Dict[str, tuple] = {
    "fixation_s": (0.2, 30.0),
    "cue_s": (0.2, 30.0),
    "mi_s": (1.0, 30.0),
    "post_mi_hold_s": (0.0, 30.0),
    "rest_s": (4.0, 60.0),
    "transition_s": (0.5, 60.0),
}

_TIMING_ZH = {
    "fixation_s": "注视 Fixation",
    "cue_s": "提示 Cue",
    "mi_s": "运动想象 MI",
    "post_mi_hold_s": "保持 PostMI",
    "rest_s": "静息 Rest",
    "transition_s": "过渡 Transition",
}


def timing_from_dict(raw: Optional[Dict[str, Any]]) -> TrialTiming:
    """从 run_config.experiment.timing 构造 TrialTiming；缺失/非法键取默认。"""
    base = DEFAULT_TIMING.to_dict()
    if isinstance(raw, dict):
        for k in base:
            v = raw.get(k)
            if v is None or v == "":
                continue
            try:
                base[k] = float(v)
            except (TypeError, ValueError):
                continue
    return TrialTiming(**base)


def validate_timing_dict(raw: Optional[Dict[str, Any]]) -> List[str]:
    """校验时序配置，返回错误列表（空 = 合法）。"""
    errors: List[str] = []
    if raw is None:
        return errors
    if not isinstance(raw, dict):
        return ["timing 须为对象（各阶段秒数）"]
    for k, (lo, hi) in TIMING_BOUNDS.items():
        if k not in raw or raw[k] is None or raw[k] == "":
            continue
        try:
            v = float(raw[k])
        except (TypeError, ValueError):
            errors.append(f"{_TIMING_ZH[k]} 时长须为数字")
            continue
        if v < lo or v > hi:
            errors.append(
                f"{_TIMING_ZH[k]} 时长须在 {lo:g}–{hi:g}s（当前 {v:g}s）"
            )
    return errors
