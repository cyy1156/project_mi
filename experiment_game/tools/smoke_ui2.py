#!/usr/bin/env python3
"""UI-2 冒烟：默认配置读写 + 串口枚举（无需浏览器/采数）。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.experiment.defaults_store import (
    load_operator_defaults,
    save_operator_defaults,
)
from experiment_game.experiment.run_config import default_run_config
from experiment_game.experiment.serial_ports import list_serial_ports


def main() -> int:
    cfg = default_run_config()
    cfg["subject"]["subject_id"] = "ui2smoke"
    cfg["acquisition"]["serial_port"] = "COM9"
    cfg["ui"]["skip_setup_if_unchanged"] = True

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "operator_defaults.json"
        ok, msg, saved = save_operator_defaults(cfg, path, repo_root=_REPO_ROOT)
        assert ok, msg
        loaded, err = load_operator_defaults(path, repo_root=_REPO_ROOT)
        assert err is None, err
        assert loaded["subject"]["subject_id"] == "ui2smoke"
        assert loaded["acquisition"]["serial_port"] == "COM9"
        assert loaded["ui"]["skip_setup_if_unchanged"] is True
        print("defaults_ok:", path)
        print(json.dumps({"subject": loaded["subject"], "ui": loaded["ui"]}, ensure_ascii=False))

    ports = list_serial_ports()
    print("serial_ports:", json.dumps(ports, ensure_ascii=False))
    print("UI2_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
