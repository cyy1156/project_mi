"""验收：ServiceManager 连续 start → stop → start（合成板）。"""

from __future__ import annotations

import time

from lsl_connect.config_loader import build_service_manager_config
from lsl_connect.service_manager import ServiceManager
from lsl_connect.state import ServiceState


def main() -> None:
    cfg, _ = build_service_manager_config()
    cfg.board_config.use_synthetic = True
    cfg.recording.auto_start = False

    mgr = ServiceManager(cfg)
    assert mgr.get_state() == ServiceState.IDLE

    for cycle in range(3):
        print(f"--- cycle {cycle + 1}: start ---")
        assert mgr.start_acquisition(), mgr.get_status().get("last_error")
        assert mgr.get_state() == ServiceState.RUNNING
        time.sleep(1.5)
        pushed = mgr.get_status()["samples_pushed"]
        assert pushed > 100, f"cycle {cycle + 1}: samples_pushed={pushed}"

        print(f"--- cycle {cycle + 1}: stop ---")
        assert mgr.stop_acquisition()
        assert mgr.get_state() == ServiceState.IDLE
        time.sleep(0.2)

    print("[PASS] start-stop-start x3 通过")


if __name__ == "__main__":
    main()
