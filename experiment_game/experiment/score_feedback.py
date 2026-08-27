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
      [0, 1.25)→0 … [3.75, 5)→3；≥5 → (4, True)
    """
    s = float(score)
    stop = float(early_stop)
    n = max(1, int(n_levels))
    if s >= stop:
        return n, True
    if s <= 0:
        return 0, False
    bin_w = stop / n
    level = int(s // bin_w)
    level = max(0, min(n - 1, level))
    return level, False


def enrich_stage_data(
    stage: str,
    ctx: Any,
    data: Optional[Dict],
    *,
    early_stop: float = 5.0,
    peak_by_trial: Optional[MutableMapping[int, int]] = None,
    n_levels: int = 4,
) -> Optional[Dict]:
    """为 judge / score_reach / touch 附加 arm_level、cup_grasp；同试次单调不回缩。"""
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
        return out

    if out.get("signal_bad"):
        prev = int(peaks.get(int(tid), 0)) if peaks is not None and tid is not None else 0
        out["arm_level"] = prev
        out["cup_grasp"] = False
        return out

    if out.get("score") is not None:
        level, grasp = map_score_to_arm(float(out["score"]))
    elif out.get("cup_grasp"):
        level, grasp = ARM_REACH_LEVELS, True
    else:
        return out

    if peaks is not None and tid is not None:
        key = int(tid)
        level = max(int(peaks.get(key, 0)), int(level))
        peaks[key] = level

    out["arm_level"] = int(level)
    out["cup_grasp"] = bool(grasp or level >= ARM_REACH_LEVELS)
    return out
