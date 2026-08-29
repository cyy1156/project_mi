"""experiment_game 核心层：零项目依赖（仅标准库/第三方库）。

依赖规则（见 docs/重构实施方案_20260829.md §1.2）：
- core/ 不得 import 任何其它项目包（experiment / acquisition / offline / tools）。
- 依赖方向由 experiment/tests/test_dep_directions.py 用 AST 锁死。
"""

from experiment_game.core import atomic_io, channel_layout, jsonl, windowing  # noqa: F401

__all__ = ["atomic_io", "channel_layout", "jsonl", "windowing"]
