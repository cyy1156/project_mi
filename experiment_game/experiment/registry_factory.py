"""从 v3/v2 配置构建推理注册表（单模 Shallow 或 E1f 四成员）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from adapt_engine.e1f import E1fRegistry, E1fStackConfig
from adapt_engine.registry import ModelRegistry

_REPO = Path(__file__).resolve().parents[2]


def is_e1f_mode(cfg: Any) -> bool:
    return str(getattr(cfg, "readout_mode", "") or "").lower() == "e1f"


def load_e1f_stack(cfg: Any, *, repo_root: Path | None = None) -> E1fStackConfig:
    root = repo_root or _REPO
    path = getattr(cfg, "e1f_config_path", "") or "experiment_game/config/e1f_four_member.json"
    stack = E1fStackConfig.load_json(path, repo_root=root)
    return stack.resolve_paths(repo_root=root)


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
