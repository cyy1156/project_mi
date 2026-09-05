"""仓库路径解析（重构总册 §4.6 / W1）。

落盘一律优先 repo 相对路径；读入时经 resolve() 还原。
可用环境变量 MI_REPO_ROOT 覆盖探测结果。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Union

_PathLike = Union[str, Path]

# experiment_game/core/paths.py → parents[2] = 仓库根
_DEFAULT_REPO = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """探测仓库根；``MI_REPO_ROOT`` 可覆盖。"""
    env = (os.environ.get("MI_REPO_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_REPO.resolve()


def to_stored(path: _PathLike, *, root: Optional[Path] = None) -> str:
    """落盘用：尽量写成相对 ``root``（默认 repo_root）的正斜杠路径。"""
    p = Path(path).expanduser()
    base = (root or repo_root()).resolve()
    try:
        return str(p.resolve().relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def resolve(stored: _PathLike, *, root: Optional[Path] = None) -> Path:
    """读入配置路径：相对 → 拼 repo；绝对且在 repo 内 → 原样；repo 外 → 原样并警告。"""
    base = (root or repo_root()).resolve()
    raw = str(stored).strip()
    if not raw:
        raise ValueError("empty path")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        return (base / p).resolve()
    resolved = p.resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        warnings.warn(
            f"path outside repo_root ({base}): {resolved}",
            UserWarning,
            stacklevel=2,
        )
    return resolved


def look_like_absolute_windows(s: str) -> bool:
    """粗检：是否像 ``D:\\...`` / ``C:/...`` 绝对盘符路径。"""
    t = (s or "").strip()
    if len(t) >= 3 and t[1] == ":" and t[0].isalpha() and t[2] in ("\\", "/"):
        return True
    return False
