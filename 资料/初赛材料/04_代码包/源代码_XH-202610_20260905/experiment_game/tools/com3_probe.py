#!/usr/bin/env python3
"""COM3 串口诊断：BrainFlow 握手 + 采集质量。"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LSL = _REPO / "collect_data" / "LSL_connect_model" / "LSL_connect_model"
for p in (_REPO, _LSL):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM3"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0


def probe_handshake(port: str) -> bool:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    from lsl_connect.board import CytonBoard

    BoardShim.enable_dev_board_logger()
    CytonBoard.force_release_all(settle_sec=0.35)
    print(f"=== BrainFlow handshake {port} ===")
    print("  (先停裸串口遗留流，再 BrainFlow 握手)")
    ok_idle = CytonBoard.ensure_serial_idle(port)
    if not ok_idle:
        print("  [warn] 停流预检未收到 OpenBCI 版本 — 检查 Cyton 开机/dongle 配对")
    CytonBoard.force_release_all(settle_sec=0.35)

    params = BrainFlowInputParams()
    params.serial_port = port
    params.timeout = 15
    try:
        board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        board.prepare_session()
        print("prepare_session: OK")
        board.start_stream(450000)
        print("start_stream: OK")
        time.sleep(2.0)
        data = board.get_current_board_data(250)
        print(f"samples in buffer: {data.shape[1]} (expect ~500)")
        board.stop_stream()
        board.release_session()
        CytonBoard.force_release_all(settle_sec=0.25)
        print("Cyton connect: SUCCESS")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Cyton connect: FAIL -> {exc}")
        CytonBoard.force_release_all(settle_sec=0.25)
        return False


def probe_pipeline(port: str, duration: float) -> None:
    from experiment_game.acquisition import AcquisitionFacade

    print(f"\n=== Full pipeline {duration:.0f}s ({port}) ===")
    with tempfile.TemporaryDirectory(prefix="com3_probe_") as tmp:
        csv_path = Path(tmp) / "eeg.csv"
        acq = AcquisitionFacade(use_synthetic=False, serial_port=port)
        try:
            acq.create()
            acq.start(csv_path)
            hc = acq.health_check()
            print(f"health_check: {hc}")
            t0 = time.time()
            while time.time() - t0 < duration:
                time.sleep(1.0)
            report = acq.stop()
            quality = report.get("quality") or {}
            rows = sum(1 for _ in csv_path.open(encoding="utf-8")) - 1
            print(f"quality: {quality}")
            print(f"csv_rows: {rows}  expected~{int(duration * 250)}")
            eff_hz = rows / max(duration, 0.1)
            print(f"effective_hz: {eff_hz:.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"pipeline FAIL: {exc}")
            try:
                acq.shutdown()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    print(f"COM probe port={PORT} duration={DURATION}s")
    ok = probe_handshake(PORT)
    if not ok:
        print("\n[HINT] 常见原因:")
        print("  1. 刚跑过 com3_raw_probe 且发了 b 开流 — 请先: python com3_raw_probe.py COM3")
        print("     (脚本会自动 s 停流；或 Cyton 关机再开)")
        print("  2. OpenBCI GUI 已占用 COM3 — 请先完全关闭 GUI")
        print("  3. Cyton 未开机 / dongle 未配对")
        print("  4. COM 口选错（设备管理器确认 dongle 对应端口）")
        return 1
    probe_pipeline(PORT, DURATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
