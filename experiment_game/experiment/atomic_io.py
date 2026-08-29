"""原子写文件，避免崩溃截断。

2026-08-29 实现平移至 experiment_game/core/atomic_io.py；
本文件仅为兼容转发（旧 import 路径继续可用），新代码请直接 import core。
"""

from experiment_game.core.atomic_io import (  # noqa: F401
    atomic_copy_files_into,
    atomic_write_json,
    atomic_write_text,
)

__all__ = ["atomic_copy_files_into", "atomic_write_json", "atomic_write_text"]
