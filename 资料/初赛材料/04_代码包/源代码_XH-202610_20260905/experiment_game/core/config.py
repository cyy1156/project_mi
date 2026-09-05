"""配置加载（总册 §1.3 / §4.6）：env > machine.json > protocol.yaml。

本期落地解析链骨架；run_config 仍由 ``experiment.run_config`` 合并会话快照。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from experiment_game.core.paths import repo_root, resolve

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def default_protocol_path(*, root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / "experiment_game" / "config" / "protocol.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("需要 PyYAML：pip install pyyaml")
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根须为对象: {path}")
    return data


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 根须为对象: {path}")
    return data


def load_machine_json(machine_dir: Path) -> Dict[str, Any]:
    p = Path(machine_dir) / "machine.json"
    if not p.is_file():
        return {}
    return load_json(p)


def env_overrides(prefix: str = "MI_") -> Dict[str, Any]:
    """读取 ``MI_*`` 环境变量为扁平覆盖表。"""
    out: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix) or not v:
            continue
        key = k[len(prefix) :].lower()
        out[key] = v
    return out


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = v
    return merged


def load_layered_config(
    *,
    root: Optional[Path] = None,
    machine_dir: Optional[Path] = None,
    protocol_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """解析链：protocol.yaml < machine.json < MI_* env。"""
    base = Path(root) if root is not None else repo_root()
    proto_p = Path(protocol_path) if protocol_path else default_protocol_path(root=base)
    cfg: Dict[str, Any] = {}
    if proto_p.is_file():
        cfg = load_yaml(proto_p)

    if machine_dir is not None:
        m = load_machine_json(Path(machine_dir))
        if m:
            cfg = deep_merge(cfg, {"site": m})

    env = env_overrides()
    if "repo_root" in env:
        cfg = deep_merge(cfg, {"site": {"repo_root": env["repo_root"]}})
    if "serial_port" in env:
        cfg = deep_merge(cfg, {"site": {"board": {"serial_port": env["serial_port"]}}})
    if "conda_env" in env:
        cfg = deep_merge(
            cfg, {"site": {"python": {"conda_env": env["conda_env"]}}}
        )

    # 规范化 protocol 内相对路径字段（若有）
    paths = cfg.get("paths")
    if isinstance(paths, dict):
        fixed = {}
        for k, v in paths.items():
            if isinstance(v, str) and v:
                try:
                    fixed[k] = str(resolve(v, root=base))
                except ValueError:
                    fixed[k] = v
            else:
                fixed[k] = v
        cfg["paths"] = fixed
    return cfg
