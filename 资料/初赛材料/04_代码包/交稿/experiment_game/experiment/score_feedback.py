"""每窗判对 +1 → 伸手档位 / 拿杯（arm_reach 反馈，非 D8 加权）。"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional, Tuple

ARM_REACH_CAP = 5.0
ARM_REACH_LEVELS = 4


def map_score_to_arm(
    score: float,
    *,
    early_stop: float = ARM_REACH_CAP,
    n_levels: int = ARM_REACH_LEVELS,
) -> Tuple[int, bool]:
    """连续分 → (arm_level 0..n_levels, cup_grasp)。

    默认 early_stop=5、n_levels=4：
      score≤0 → (0, False)
      score 1..4 → level 1..min(3, score)（**第 1 次命中即可见**）
      score≥5 → (4, True) 拿杯
    """
    s = float(score)
    stop = float(early_stop)
    n = max(1, int(n_levels))
    if s >= stop:
        return n, True
    if s <= 0:
        return 0, False
    # 第 1 次命中即可见：不再把 [0, stop/n) 映射到 level 0
    level = min(n - 1, max(1, int(s)))
    return level, False


def arm_progress_from_score(
    score: float,
    *,
    early_stop: float = ARM_REACH_CAP,
) -> float:
    """连续伸手进度 0..1（score/early_stop，封顶 1）。"""
    stop = float(early_stop)
    if stop <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, float(score) / stop))


def enrich_stage_data(
    stage: str,
    ctx: Any,
    data: Optional[Dict],
    *,
    early_stop: float = 5.0,
    peak_by_trial: Optional[MutableMapping[int, float]] = None,
    n_levels: int = 4,
) -> Optional[Dict]:
    """为 judge / score_reach / touch 附加 arm_level、cup_grasp、arm_progress；同试次单调不回缩。

    ``peak_by_trial`` 存本试次历史最高 **score**（非 level），signal_bad 时用峰值重算
    level/progress，避免把 level 误当 score 导致 progress 回缩。
    """
    tid = getattr(ctx, "trial_id", None) if ctx is not None else None
    peaks = peak_by_trial
    if stage == "trial_start" and peaks is not None and tid is not None:
        peaks.pop(int(tid), None)

    if data is None:
        return None
    if not isinstance(data, dict):
        return data

    out = dict(data)
    label = getattr(ctx, "label", None) if ctx is not None else None

    if stage not in ("judge", "trial_end"):
        return out

    # Rest：无伸手、无拿杯
    if label == 0:
        out["arm_level"] = 0
        out["cup_grasp"] = False
        out["arm_progress"] = 0.0
        return out

    if out.get("signal_bad"):
        prev_score = (
            float(peaks.get(int(tid), 0.0))
            if peaks is not None and tid is not None
            else 0.0
        )
        level, _grasp = map_score_to_arm(
            prev_score, early_stop=early_stop, n_levels=n_levels
        )
        out["arm_level"] = int(level)
        out["cup_grasp"] = False
        out["arm_progress"] = arm_progress_from_score(
            prev_score, early_stop=early_stop
        )
        return out

    if out.get("score") is not None:
        score = float(out["score"])
    elif out.get("cup_grasp"):
        score = float(early_stop)
    else:
        return out

    if peaks is not None and tid is not None:
        key = int(tid)
        score = max(float(peaks.get(key, 0.0)), score)
        peaks[key] = score

    level, grasp = map_score_to_arm(
        score, early_stop=early_stop, n_levels=n_levels
    )
    progress = arm_progress_from_score(score, early_stop=early_stop)

    out["arm_level"] = int(level)
    out["cup_grasp"] = bool(grasp or level >= ARM_REACH_LEVELS)
    out["arm_progress"] = float(progress)
    return out
