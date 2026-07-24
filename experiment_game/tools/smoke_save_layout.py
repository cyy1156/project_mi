#!/usr/bin/env python3
"""合成板冒烟：短会话 + 落盘布局/对齐（无需浏览器）。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.acquisition import AcquisitionFacade
from experiment_game.experiment.alignment import write_alignment_bundle
from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher
from experiment_game.experiment.session import SessionMeta, create_session_dir, write_session_meta
from experiment_game.experiment.session_layout import finalize_session_layout
from experiment_game.experiment.session_runner import Phase2Config, SessionRunner
from experiment_game.experiment.ws_bridge import WsBridge


def main() -> int:
    out = _REPO_ROOT / "experiment_game" / "data" / "sessions"
    paths = create_session_dir(out, "smoke", "layout")
    write_session_meta(
        paths.meta_json,
        SessionMeta(
            subject_id="smoke",
            session_id="layout",
            use_synthetic=True,
            trial_count=2,
            notes="smoke_layout",
        ),
    )
    events = EventLogger(paths.events_jsonl)
    markers = MarkerPublisher(enabled=True)
    # 无客户端时仍可跑：auto_continue
    bridge = WsBridge(port=8777)
    bridge.start()
    acq = AcquisitionFacade(use_synthetic=True)
    try:
        acq.create()
        acq.start(paths.eeg_csv)
        time.sleep(1.0)
        events.emit("session_start", subject_id="smoke", session_id="layout", phase="phase2")
        cfg = Phase2Config(
            acquire_trials=2,
            learn_trials_per_step=2,
            seed=1,
            skip_adapt=True,
            skip_learn=True,
            skip_gate=True,
            auto_continue=True,
            rotate_objects=False,
            rotate_scenes=False,
        )
        runner = SessionRunner(events, markers, bridge, config=cfg)
        runner.run_all()
        events.emit("session_end", subject_id="smoke", session_id="layout", phase="phase2")
    finally:
        try:
            acq.stop()
        except Exception:  # noqa: BLE001
            pass
        acq.shutdown()
        events.close()
        markers.close()
        bridge.stop()

    finalize_session_layout(
        paths.root,
        save_layout="phase_folders",
        save_continuous=True,
        save_phase_slices=True,
        acq_enabled=True,
    )
    report = write_alignment_bundle(paths.root, acq_enabled=True)
    print("session:", paths.root)
    print("verify:", json.dumps(report, ensure_ascii=False, indent=2))
    cont = paths.root / "continuous" / "eeg.csv"
    align = paths.root / "alignment" / "trial_table.csv"
    acquire = paths.root / "by_phase" / "06_acquire" / "eeg.csv"
    ok = cont.is_file() and align.is_file() and acquire.is_file() and report.get("passed")
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
