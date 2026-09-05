"""从 baselines_single / Self_development 脚本安全取出 build_model（避免路径阴影）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
STEP = HERE.parent


def load_sibling_module(unique_name: str, script: Path, *, prefer_dir: Path | None = None) -> ModuleType:
    """exec 外部脚本；可选将 prefer_dir 临时插到 sys.path 最前，保证其 md_fold_detail 等可导入。

    加载结束后若污染了 ``shared_hparams`` / ``md_fold_detail``，恢复为 baselines_2s_hop100 版本，
    避免 task_runner 用到 baselines_single 的 SHARED。
    """
    script = script.resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    inserted: str | None = None
    saved = {k: sys.modules.get(k) for k in ("shared_hparams", "md_fold_detail")}
    if prefer_dir is not None:
        inserted = str(prefer_dir.resolve())
        if inserted in sys.path:
            sys.path.remove(inserted)
        sys.path.insert(0, inserted)
    try:
        # 强制旧脚本看到 prefer_dir 下的同名模块
        for k in ("shared_hparams", "md_fold_detail"):
            sys.modules.pop(k, None)
        spec = importlib.util.spec_from_file_location(unique_name, script)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {script}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if inserted is not None and inserted in sys.path:
            sys.path.remove(inserted)
        # 恢复 baselines_2s_hop100 侧模块；必要时重新从 HERE 加载
        here = str(HERE)
        if here in sys.path:
            sys.path.remove(here)
        sys.path.insert(0, here)
        for k, prev in saved.items():
            if prev is not None and "/baselines_2s_hop100/" in getattr(prev, "__file__", "").replace("\\", "/"):
                sys.modules[k] = prev
            else:
                sys.modules.pop(k, None)
                try:
                    sys.modules[k] = importlib.import_module(k)
                except Exception:
                    if prev is not None:
                        sys.modules[k] = prev
        # 若 task_runner 已 import，刷新其 SHARED 绑定
        tr = sys.modules.get("task_runner")
        sh = sys.modules.get("shared_hparams")
        if tr is not None and sh is not None and hasattr(sh, "SHARED"):
            tr.SHARED = sh.SHARED
            if hasattr(sh, "shared_as_dict"):
                tr.shared_as_dict = sh.shared_as_dict
            if hasattr(sh, "SharedTrainHP"):
                tr.SharedTrainHP = sh.SharedTrainHP


def load_baselines_single(name: str) -> ModuleType:
    """name 如 baseline_dbn.py → 优先用 baselines_single 目录解析依赖。"""
    d = STEP / "baselines_single"
    return load_sibling_module(f"_bs1s_{name}", d / name, prefer_dir=d)


def load_selfdev(name: str) -> ModuleType:
    d = STEP / "Self_development_model"
    # Self_development 脚本依赖 baselines_single 的 md_fold_detail / shared_hparams
    return load_sibling_module(
        f"_selfdev1s_{name}",
        d / name,
        prefer_dir=STEP / "baselines_single",
    )
