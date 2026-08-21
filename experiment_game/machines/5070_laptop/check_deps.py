"""Check experiment_game (+ optional acquisition) deps on this machine.

Machine: 5070_laptop (hostname cyy, conda env cyy, repo D:\\MI).
"""
from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACHINE_JSON = HERE / "machine.json"

# experiment_game/requirements.txt
CORE = ("websockets", "scipy", "numpy")
# collect_data LSL path (needed when recording / Cyton)
ACQ = ("brainflow", "pylsl", "yaml")  # yaml = PyYAML


def _ok(name: str) -> tuple[bool, str]:
    mod = "yaml" if name == "yaml" else name
    spec = importlib.util.find_spec(mod)
    if spec is None:
        return False, "MISSING"
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        return True, str(ver)
    except Exception as e:  # noqa: BLE001
        return False, f"import error: {e}"


def main() -> int:
    meta = {}
    if MACHINE_JSON.is_file():
        meta = json.loads(MACHINE_JSON.read_text(encoding="utf-8"))

    print("=== experiment_game dependency check ===")
    print(f"machine_id : {meta.get('machine_id', '5070_laptop')}")
    print(f"hostname   : {platform.node()}")
    print(f"python     : {sys.executable}")
    print(f"version    : {sys.version.split()[0]}")
    print()

    missing: list[str] = []

    print("-- core (experiment_game/requirements.txt) --")
    for pkg in CORE:
        ok, info = _ok(pkg)
        mark = "OK " if ok else "FAIL"
        print(f"  [{mark}] {pkg}: {info}")
        if not ok:
            missing.append(pkg)

    print()
    print("-- acquisition optional (LSL / BrainFlow) --")
    for pkg in ACQ:
        label = "PyYAML" if pkg == "yaml" else pkg
        ok, info = _ok(pkg)
        mark = "OK " if ok else "FAIL"
        print(f"  [{mark}] {label}: {info}")
        if not ok:
            missing.append(label)

    print()
    if missing:
        print("MISSING:", ", ".join(missing))
        print("Install into conda env cyy:")
        print("  conda activate cyy")
        print("  python -m pip install -r experiment_game\\requirements.txt")
        print(
            "  python -m pip install -r "
            "collect_data\\LSL_connect_model\\LSL_connect_model\\requirements.txt"
        )
        return 1

    print("All checked packages are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
