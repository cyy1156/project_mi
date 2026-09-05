"""Wait until no run_p_track.py is active, then start the next arm if needed."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / "out" / "5060_ciacnet_mi_accpaper"  # code/train_lab/out/...
PY = Path(r"D:\cyy\MI\.venv\Scripts\python.exe")


def p_track_running() -> bool:
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine",
            ],
            text=True,
            errors="ignore",
        )
    except Exception:
        return False
    return "run_p_track.py" in out


def latest_summary(arm: str) -> dict | None:
    d = OUT / arm
    if not d.exists():
        return None
    runs = sorted(d.glob("run_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        try:
            s = json.loads(r.read_text(encoding="utf-8"))
            if s.get("mean_acc") is not None and len(s.get("subjects", [])) >= 9:
                return s
        except Exception:
            continue
    return None


def main() -> None:
    # Wait for current P1 to finish
    print("waiting for existing run_p_track to finish...", flush=True)
    while p_track_running():
        time.sleep(30)
    print("no runner active", flush=True)

    p1 = latest_summary("P1")
    if p1 is None:
        print("launching P1", flush=True)
        subprocess.check_call([str(PY), "-u", str(HERE / "run_p_track.py"), "--arm", "P1"], cwd=str(HERE))
        p1 = latest_summary("P1")
    else:
        print(f"P1 done mean_acc={p1['mean_acc']:.4f}", flush=True)

    p2 = latest_summary("P2")
    if p2 is None:
        print("launching P2", flush=True)
        subprocess.check_call([str(PY), "-u", str(HERE / "run_p_track.py"), "--arm", "P2"], cwd=str(HERE))
        p2 = latest_summary("P2")
    else:
        print(f"P2 already done mean_acc={p2['mean_acc']:.4f}", flush=True)

    print("ALL DONE", flush=True)
    if p1:
        print("P1", p1["mean_acc"], p1["mean_kappa"])
    if p2:
        print("P2", p2["mean_acc"], p2["mean_kappa"])


if __name__ == "__main__":
    main()
