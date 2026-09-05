#!/usr/bin/env python3
"""操作台 WS 端到端冒烟：session_start → ready → 合成采数 → session_saved。"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import websockets

from experiment_game.experiment.orchestrator import OperatorService

HTTP_PORT = 18080
WS_PORT = 18765


async def client_drive(ws_url: str) -> dict:
    saved = None
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "operator_hello"}))
        # drain hello
        await asyncio.wait_for(ws.recv(), timeout=5)

        cfg = {
            "schema_version": 2,
            "subject": {"subject_id": "opsmoke", "session_id": "ws01", "notes": "ws_smoke"},
            "acquisition": {
                "enabled": True,
                "board_mode": "synthetic",
                "serial_port": "COM5",
                "markers_lsl": True,
            },
            "experiment": {
                "acquire_trials": 2,
                "learn_trials_per_step": 1,
                "skip_adapt": True,
                "skip_learn": True,
                "skip_gate": True,
                "seed": 7,
                "open_subject_page": False,
                "ready_timeout_s": 30,
            },
            "storage": {
                "save_root": str(
                    (_REPO_ROOT / "experiment_game" / "data" / "sessions").resolve()
                ),
                "save_layout": "phase_folders",
                "save_events": True,
                "save_session_meta": True,
                "save_continuous_master": True,
                "save_phase_slices": True,
                "save_trial_index": True,
            },
        }
        await ws.send(json.dumps({"type": "session_start", "run_config": cfg}))

        # subject ready (same WS client is fine)
        await ws.send(json.dumps({"type": "ready"}))

        deadline = time.time() + 180
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "config_ack" and not msg.get("ok"):
                raise RuntimeError(f"config_ack fail: {msg.get('errors')}")
            if t == "session_saved":
                saved = msg
                break
            if t == "session" and msg.get("status") == "error":
                raise RuntimeError(msg.get("message") or "session error")
        if saved is None:
            raise TimeoutError("no session_saved")
        return saved


def main() -> int:
    svc = OperatorService(http_port=HTTP_PORT, ws_port=WS_PORT)
    t = threading.Thread(target=svc.serve_forever, name="op-svc", daemon=True)
    t.start()
    time.sleep(0.8)
    try:
        saved = asyncio.run(client_drive(f"ws://127.0.0.1:{WS_PORT}"))
    finally:
        svc.stop()

    root = Path(saved["root"])
    verify = saved.get("verify") or {}
    need = [
        root / "eeg.csv",
        root / "continuous" / "eeg.csv",
        root / "by_phase" / "06_acquire" / "eeg.csv",
        root / "alignment" / "verify_report.json",
        root / "run_config.json",
    ]
    missing = [str(p.relative_to(root)) for p in need if not p.is_file()]
    print("session:", root)
    print("verify.passed:", verify.get("passed"))
    print("files_ok:", not missing, "missing:", missing)
    print("train_eligible:", saved.get("train_eligible"))
    ok = (
        verify.get("passed") is True
        and not missing
        and bool(saved.get("acq_enabled"))
    )
    print("OPERATOR_SMOKE_OK" if ok else "OPERATOR_SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
