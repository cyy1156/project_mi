"""操作台本地默认配置：config/operator_defaults.json。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiment_game.experiment.run_config import merge_run_config, validate_run_config

_PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = _PKG_ROOT / "config" / "operator_defaults.json"


def defaults_path(*, repo_pkg: Optional[Path] = None) -> Path:
    root = Path(repo_pkg) if repo_pkg else _PKG_ROOT
    return root / "config" / "operator_defaults.json"


def load_operator_defaults(
    path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    读取并合并为完整 run_config。
    返回 (config, error_message)；文件不存在时返回内置默认且 error=None。
    """
    p = Path(path) if path else DEFAULTS_PATH
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
        # 仍返回合并结果，但带警告（允许部分损坏时可用）
        return merge_run_config(raw), "; ".join(errors)
    return cfg, None


def save_operator_defaults(
    cfg: Dict[str, Any],
    path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """校验后写入；返回 (ok, message, normalized_cfg)。"""
    p = Path(path) if path else DEFAULTS_PATH
    normalized, errors = validate_run_config(cfg, repo_root=repo_root)
    if errors:
        return False, "; ".join(errors), None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"写入失败: {exc}", None
    return True, str(p), normalized
