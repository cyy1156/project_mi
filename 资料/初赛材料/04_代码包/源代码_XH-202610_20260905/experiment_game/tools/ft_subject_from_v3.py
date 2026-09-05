"""CLI 薄壳：实现已下沉至 ``experiment_game.pipeline.finetune``。

将本模块替换为 pipeline 实现，保留所有 ``from …ft_subject_from_v3 import _safe_int`` 等兼容导入。
"""

from __future__ import annotations

import sys

from experiment_game.pipeline import finetune as _impl

sys.modules[__name__] = _impl

if __name__ == "__main__":
    _impl.main()
