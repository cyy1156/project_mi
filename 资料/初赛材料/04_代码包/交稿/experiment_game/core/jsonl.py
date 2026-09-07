"""JSONL 容错读取（重构阶段 2 · events.jsonl）。

坏行跳过并计数；不改写文件。core 零项目依赖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl_tolerant(path: Path | str) -> Tuple[List[Dict[str, Any]], int]:
    """逐行解析 JSONL。

    Returns:
        (rows, n_bad_lines)
    """
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    n_bad = 0
    if not p.is_file():
        return rows, 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                n_bad += 1
    return rows, n_bad
