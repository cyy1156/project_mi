"""§6.2 防回归：核心业务层禁止硬编码盘符（tools/ 入口允许 CLI 默认 COM）。"""

from __future__ import annotations

import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]

# 只扫核心层；tools/ 为操作入口可保留 CLI 默认串口
_SCAN_DIRS = ("core", "experiment", "acquisition", "pipeline", "runtime", "offline")
_SKIP_NAME = {"test_ci_no_hardcoded_paths.py"}

# 允许的 COM 默认文件（schema / facade 默认，由 UI/机位覆盖）
_COM_ALLOWED = {
    "run_config.py",
    "service.py",  # AcquisitionFacade
    "orchestrator.py",  # fallback COM5
    "defaults_store.py",
}

_DRIVE = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:/")


def _iter_core_py():
    for name in _SCAN_DIRS:
        root = _PKG / name
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "tests" in p.parts or p.name in _SKIP_NAME:
                continue
            yield p


def test_no_absolute_windows_drive_in_core_code():
    """core/experiment/acquisition 等不应出现 D:\\ 硬编码（docstring 中的示例除外）。"""
    offenders = []
    for path in _iter_core_py():
        text = path.read_text(encoding="utf-8", errors="replace")
        in_doc = False
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                # 粗略跳过三引号文档行
                count = stripped.count('"""') + stripped.count("'''")
                if count == 1:
                    in_doc = not in_doc
                continue
            if in_doc or stripped.startswith("#"):
                continue
            if _DRIVE.search(line):
                offenders.append(f"{path.relative_to(_PKG)}:{i}: {stripped[:80]}")
    assert not offenders, "发现硬编码盘符:\n" + "\n".join(offenders[:20])


def test_com_defaults_limited_to_config_surface():
    """COM 默认仅允许出现在配置/门面层，不散落到业务逻辑。"""
    com = re.compile(r"\bCOM[0-9]+\b")
    offenders = []
    for path in _iter_core_py():
        if path.name in _COM_ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if com.search(line):
                offenders.append(f"{path.relative_to(_PKG)}:{i}: {stripped[:80]}")
    assert not offenders, "发现散落 COM 硬编码:\n" + "\n".join(offenders[:20])
