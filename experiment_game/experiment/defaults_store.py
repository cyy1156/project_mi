"""操作台本地默认配置：config/operator_defaults.json（本机文件，不进版本库）。

落盘时将 save_root 尽量写成相对仓库根的路径，避免换机粘绝对盘符。
缺失时回退读取 operator_defaults.example.json。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiment_game.core.atomic_io import atomic_write_json
from experiment_game.core.paths import to_stored
from experiment_game.experiment.run_config import merge_run_config, validate_run_config

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = _PKG_ROOT / "config" / "operator_defaults.json"
EXAMPLE_PATH = _PKG_ROOT / "config" / "operator_defaults.example.json"


def defaults_path(*, repo_pkg: Optional[Path] = None) -> Path:
    root = Path(repo_pkg) if repo_pkg else _PKG_ROOT
    return root / "config" / "operator_defaults.json"


def example_defaults_path(*, repo_pkg: Optional[Path] = None) -> Path:
    root = Path(repo_pkg) if repo_pkg else _PKG_ROOT
    return root / "config" / "operator_defaults.example.json"


def _relativize_for_disk(
    cfg: Dict[str, Any],
    *,
    repo_root: Optional[Path],
) -> Dict[str, Any]:
    """写盘副本：save_root 优先相对 repo_root。"""
    out = copy.deepcopy(cfg)
    storage = out.setdefault("storage", {})
    raw = str(storage.get("save_root") or "").strip()
    if not raw:
        return out
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    p = Path(raw)
    if not p.is_absolute():
        storage["save_root"] = raw.replace("\\", "/")
        return out
    storage["save_root"] = to_stored(p, root=root)
    return out


def _resolve_read_path(path: Path) -> Path:
    if path.is_file():
        return path
    example = path.with_name("operator_defaults.example.json")
    if example.is_file():
        return example
    if path == DEFAULTS_PATH and EXAMPLE_PATH.is_file():
        return EXAMPLE_PATH
    return path


def load_operator_defaults(
    path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    读取并合并为完整 run_config。
    返回 (config, error_message)；本地文件不存在时尝试 example，再内置默认。
    """
    requested = Path(path) if path else DEFAULTS_PATH
    p = _resolve_read_path(requested)
    if not p.is_file():
        return merge_run_config(None), None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return merge_run_config(None), f"读取默认配置失败: {exc}"
    if not isinstance(raw, dict):
        return merge_run_config(None), "默认配置须为 JSON 对象"
    cfg, errors = validate_run_config(raw, repo_root=repo_root)
    if errors:
        return merge_run_config(raw), "; ".join(errors)
    return cfg, None


def save_operator_defaults(
    cfg: Dict[str, Any],
    path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """校验后原子写入；落盘 save_root 尽量相对化。返回 (ok, message, normalized_cfg)。"""
    p = Path(path) if path else DEFAULTS_PATH
    normalized, errors = validate_run_config(cfg, repo_root=repo_root)
    if errors:
        return False, "; ".join(errors), None
    disk = _relativize_for_disk(normalized, repo_root=repo_root or _REPO_ROOT)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, disk)
    except OSError as exc:
        return False, f"写入失败: {exc}", None
    return True, str(p), normalized
