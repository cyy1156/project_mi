"""三分类标签约定：0=Rest, 1=Left, 2=Right。"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

LABEL_NAMES = {0: "Rest", 1: "Left", 2: "Right"}
LABEL_NAMES_ZH = {0: "静息", 1: "左手", 2: "右手"}


def label_name(label: Any, *, zh: bool = False) -> Optional[str]:
    if label is None:
        return None
    try:
        i = int(label)
    except (TypeError, ValueError):
        return None
    table = LABEL_NAMES_ZH if zh else LABEL_NAMES
    return table.get(i)


def normalize_p_three(p: Any) -> Optional[List[float]]:
    """→ 长度 3 的 float 列表；无效则 None。"""
    if p is None:
        return None
    try:
        vals = [float(x) for x in list(p)]
    except (TypeError, ValueError):
        return None
    if len(vals) < 3:
        return None
    return [round(vals[0], 4), round(vals[1], 4), round(vals[2], 4)]


def attach_judge_names(
    data: Optional[dict],
    *,
    label: Any = None,
) -> Optional[dict]:
    """为 judge data 附加 pred_name / correct；规范化 p_three。"""
    if data is None or not isinstance(data, dict):
        return data
    out = dict(data)
    if "p_three" in out:
        out["p_three"] = normalize_p_three(out.get("p_three"))
    pred = out.get("pred")
    if pred is not None and not out.get("signal_bad"):
        out["pred_name"] = label_name(pred)
        if label is not None:
            try:
                out["correct"] = int(pred) == int(label)
            except (TypeError, ValueError):
                out["correct"] = None
    return out
