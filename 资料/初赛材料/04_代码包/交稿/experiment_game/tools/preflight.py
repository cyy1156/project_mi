"""启动预检 Preflight（总册 §5.4 / W3）。

检查：Python 依赖、大文件 ckpt/replay、defaults 绝对路径、磁盘/端口（只报不杀）。
机位 ``check_deps`` 与 ``python -m experiment_game.tools.preflight`` 共用。
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiment_game.core.paths import look_like_absolute_windows, repo_root

CORE_PKGS = ("websockets", "scipy", "numpy")
ACQ_PKGS = ("brainflow", "pylsl", "yaml")  # yaml = PyYAML
OPTIONAL_PKGS = ("torch",)

# 相对仓库根；与 E1f / FT 默认底座对齐（5090 fold0）
DEFAULT_THREE_REL = (
    "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260823_095327/three/fold0/best_three.pt"
)
DEFAULT_TASK_REL = (
    "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260823_095327/task/fold0/best_task.pt"
)
REPLAY_T0_REL = "code/preprocess_lab/out/openbmi_3s_hop100_t0"
REPLAY_ALL_REL = "code/preprocess_lab/out/openbmi_3s_hop100"


@dataclass
class CheckItem:
    name: str
    ok: bool
    level: str  # "fail" | "warn" | "ok"
    detail: str = ""


@dataclass
class PreflightReport:
    items: List[CheckItem] = field(default_factory=list)
    repo: Path = field(default_factory=repo_root)

    @property
    def ok(self) -> bool:
        return not any(i.level == "fail" for i in self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "repo": str(self.repo),
            "items": [
                {"name": i.name, "ok": i.ok, "level": i.level, "detail": i.detail}
                for i in self.items
            ],
        }


def _pkg_ok(name: str) -> tuple[bool, str]:
    mod = "yaml" if name == "yaml" else name
    if importlib.util.find_spec(mod) is None:
        return False, "MISSING"
    try:
        m = __import__(mod)
        return True, str(getattr(m, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001
        return False, f"import error: {exc}"


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def run_preflight(
    *,
    root: Optional[Path] = None,
    require_acq: bool = False,
    require_replay: bool = False,
    check_ports: Sequence[int] = (8080, 8765),
) -> PreflightReport:
    repo = Path(root) if root is not None else repo_root()
    report = PreflightReport(repo=repo)

    for pkg in CORE_PKGS:
        ok, info = _pkg_ok(pkg)
        report.items.append(
            CheckItem(pkg, ok, "fail" if not ok else "ok", info)
        )

    for pkg in OPTIONAL_PKGS:
        ok, info = _pkg_ok(pkg)
        report.items.append(
            CheckItem(pkg, ok, "warn" if not ok else "ok", info)
        )

    for pkg in ACQ_PKGS:
        label = "PyYAML" if pkg == "yaml" else pkg
        ok, info = _pkg_ok(pkg)
        level = "fail" if (require_acq and not ok) else ("warn" if not ok else "ok")
        report.items.append(CheckItem(label, ok, level, info))

    for rel, label in (
        (DEFAULT_THREE_REL, "DEFAULT_THREE ckpt"),
        (DEFAULT_TASK_REL, "DEFAULT_TASK ckpt"),
    ):
        p = repo / rel
        ok = p.is_file()
        report.items.append(
            CheckItem(
                label,
                ok,
                "warn" if not ok else "ok",
                str(p.relative_to(repo)).replace("\\", "/") if ok else f"请拷贝 {rel}",
            )
        )

    t0 = repo / REPLAY_T0_REL
    all_r = repo / REPLAY_ALL_REL
    replay_ok = t0.is_dir() or all_r.is_dir()
    detail = (
        f"t0={t0.is_dir()} all={all_r.is_dir()}"
        if replay_ok
        else f"请拷贝 {REPLAY_T0_REL}/ 或 {REPLAY_ALL_REL}/"
    )
    report.items.append(
        CheckItem(
            "OpenBMI replay out",
            replay_ok,
            "fail" if (require_replay and not replay_ok) else ("warn" if not replay_ok else "ok"),
            detail,
        )
    )

    defaults = repo / "experiment_game" / "config" / "operator_defaults.json"
    if defaults.is_file():
        try:
            blob = json.loads(defaults.read_text(encoding="utf-8"))
            sr = str((blob.get("storage") or {}).get("save_root") or "")
            if look_like_absolute_windows(sr):
                report.items.append(
                    CheckItem(
                        "operator_defaults save_root",
                        False,
                        "warn",
                        f"含绝对路径 {sr!r}；建议删除本机 defaults 或改相对路径后重开操作台",
                    )
                )
            else:
                report.items.append(
                    CheckItem("operator_defaults save_root", True, "ok", "relative or empty")
                )
        except (OSError, json.JSONDecodeError) as exc:
            report.items.append(
                CheckItem("operator_defaults", False, "warn", str(exc))
            )
    else:
        report.items.append(
            CheckItem(
                "operator_defaults.json",
                True,
                "ok",
                "缺失（将回退 example）— 正常",
            )
        )

    try:
        usage = shutil.disk_usage(str(repo))
        free_gb = usage.free / (1024**3)
        ok = free_gb >= 5.0
        report.items.append(
            CheckItem(
                "disk_free",
                ok,
                "warn" if not ok else "ok",
                f"{free_gb:.1f} GB free",
            )
        )
    except OSError as exc:
        report.items.append(CheckItem("disk_free", False, "warn", str(exc)))

    for port in check_ports:
        free = _port_free(int(port))
        report.items.append(
            CheckItem(
                f"port_{port}",
                free,
                "warn" if not free else "ok",
                "free" if free else "in use（只报不杀）",
            )
        )

    return report


def format_report(report: PreflightReport) -> str:
    lines = [
        "=== experiment_game Preflight ===",
        f"repo: {report.repo}",
        f"python: {sys.executable}",
        f"version: {sys.version.split()[0]}",
        "",
    ]
    for i in report.items:
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[i.level]
        lines.append(f"  [{mark}] {i.name}: {i.detail}")
    lines.append("")
    lines.append("PASS" if report.ok else "FAIL（存在 fail 项）")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="experiment_game 启动预检")
    ap.add_argument("--require-acq", action="store_true")
    ap.add_argument("--require-replay", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    report = run_preflight(
        require_acq=bool(args.require_acq),
        require_replay=bool(args.require_replay),
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
