"""范式与质量过滤（第一版仅 LR）。"""
from __future__ import annotations

from src.datasets.stieger.labels import map_target

ALLOWED_TARGETS_BY_TASK :dict[int,set[int]] ={
    1:{1,2},
    2:{4},
    3:{1,2,4},
}

DEFAULT_USE_TASKS: tuple[int, ...] = (1,2,3)
def keep_trial(
    tasknumber: int,
    targetnumber: int,
    artifact: int | float,
    triallength: float,
    *,
    use_tasks: tuple[int, ...] = DEFAULT_USE_TASKS,
    min_feedback_sec: float = 4.0,
    allowed_by_task:dict[int, set[int]] | None = None,
) -> bool:
    """
        是否保留该试次。
        - 非 use_tasks → 丢
        - artifact==1 → 丢
        - 反馈时长 < min_feedback_sec → 丢
        - target 不在该 task 允许集合，或 map_target 失败（如双手）→ 丢
    """
    task = int(tasknumber)
    target = int(targetnumber)
    allowed_map = allowed_by_task or ALLOWED_TARGETS_BY_TASK

    if task not in use_tasks:
        return False
    if artifact is not None and int(artifact) == 1:
        return False
    if float(triallength) < min_feedback_sec:
        return False

    allowed = allowed_map.get(task)
    if allowed is None or target not in allowed:
        return False
    if map_target(target) is None:
        return False
    return True