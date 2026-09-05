from __future__ import annotations

from pathlib import Path

import yaml

def load_config(path:str|Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置应该为映射表：{path}")
    # 相对 out_dir：相对「配置文件所在目录的上一级」= preprocess_lab/
    out =Path(cfg["out_dir"])
    if not out.is_absolute():
        lab_root =path.resolve().parent.parent
        cfg["out_dir"] = str((lab_root/out).resolve())
    return cfg
