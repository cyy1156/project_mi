"""原子写文件，避免崩溃截断。"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Union


def atomic_write_text(path: Union[str, Path], text: str, *, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, p)


def atomic_write_json(path: Union[str, Path], data: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=indent, default=str) + "\n",
    )


def atomic_copy_files_into(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    names: Iterable[str],
) -> list[str]:
    """先拷到 staging，再 os.replace 进目标目录（避免半写 current/）。"""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    staging = dst.parent / f".promote_staging_{os.getpid()}_{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    try:
        for name in names:
            sp = src / name
            if not sp.is_file():
                continue
            shutil.copy2(sp, staging / name)
            copied.append(name)
        for name in copied:
            os.replace(staging / name, dst / name)
        return copied
    finally:
        shutil.rmtree(staging, ignore_errors=True)
