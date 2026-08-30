"""从 v3/v2 配置构建推理注册表（单模 Shallow 或 E1f 四成员）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from adapt_engine.e1f import E1fRegistry, E1fStackConfig
from adapt_engine.registry import ModelRegistry

_REPO = Path(__file__).resolve().parents[2]


def is_e1f_mode(cfg: Any) -> bool:
    return str(getattr(cfg, "readout_mode", "") or "").lower() == "e1f"


def _resolve_overlay_path(cfg: Any, *, repo_root: Path) -> Optional[Path]:
    raw = getattr(cfg, "e1f_overlay_path", "") or ""
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p if p.is_file() else None


def apply_e1f_overlay(
    stack: E1fStackConfig,
    overlay_path: Path | str | None,
    *,
    repo_root: Path | None = None,
) -> E1fStackConfig:
    """叠加被试 current 的 e1f_overlay.json（all4 FT 后四员权重）。"""
    if not overlay_path:
        return stack
    root = repo_root or _REPO
    p = Path(overlay_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.is_file():
        return stack
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return stack
    members = blob.get("members") or {}
    if not isinstance(members, dict) or not members:
        return stack
    return stack.with_member_overrides(members).resolve_paths(repo_root=root)


def load_e1f_stack(cfg: Any, *, repo_root: Path | None = None) -> E1fStackConfig:
    root = repo_root or _REPO
    path = getattr(cfg, "e1f_config_path", "") or "experiment_game/config/e1f_four_member.json"
    stack = E1fStackConfig.load_json(path, repo_root=root).resolve_paths(repo_root=root)
    overlay = _resolve_overlay_path(cfg, repo_root=root)
    if overlay is not None:
        stack = apply_e1f_overlay(stack, overlay, repo_root=root)
    return stack


def build_registry(
    cfg: Any,
    *,
    repo_root: Path | None = None,
    device: str = "cpu",
) -> Union[ModelRegistry, E1fRegistry]:
    root = repo_root or _REPO
    if is_e1f_mode(cfg):
        stack = load_e1f_stack(cfg, repo_root=root)
        missing = stack.missing_paths(repo_root=root)
        if missing:
            raise FileNotFoundError(
                "E1f 成员权重缺失（需从 5090 同步 fold0/best_three.pt）：\n  "
                + "\n  ".join(missing[:8])
            )
        return E1fRegistry(stack, device=device)

    task = root / cfg.s3_task_ckpt
    three = root / cfg.s3_three_ckpt
    if not task.is_file():
        raise FileNotFoundError(f"缺 task 权重: {task}")
    if not three.is_file():
        raise FileNotFoundError(f"缺 three 权重: {three}")
    return ModelRegistry(task, three, device=device)


def verify_registry_paths(cfg: Any, *, repo_root: Path | None = None) -> list[str]:
    root = repo_root or _REPO
    if is_e1f_mode(cfg):
        return load_e1f_stack(cfg, repo_root=root).missing_paths(repo_root=root)
    errs: list[str] = []
    task = root / cfg.s3_task_ckpt
    three = root / cfg.s3_three_ckpt
    if not task.is_file():
        errs.append(f"缺 task 权重: {task}")
    if not three.is_file():
        errs.append(f"缺 three 权重: {three}")
    return errs
