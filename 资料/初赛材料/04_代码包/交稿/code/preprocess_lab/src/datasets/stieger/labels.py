"""Stieger targetnumber → 本项目双头标签。"""
from __future__ import annotations

# 官方: 1=right, 2=left, 3=up, 4=down(rest)
# 本项目 y_three: 1=左, 2=右, 0=静息
TARGET_TO_LABELS: dict[int, tuple[int, int]] = {
    2: (1, 1),  # left  → task, left
    1: (1, 2),  # right → task, right
    4: (0, 0),  # down  → rest
}
DROP_TARGETS = {3}  # both-hands up


def map_target(targetnumber: int) -> tuple[int, int] | None:
    """返回 (y_task, y_three)；应丢弃则返回 None。"""
    t = int(targetnumber)
    if t in DROP_TARGETS:
        return None
    return TARGET_TO_LABELS.get(t)