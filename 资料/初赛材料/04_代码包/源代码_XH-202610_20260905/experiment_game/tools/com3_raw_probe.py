#!/usr/bin/env python3
"""Raw OpenBCI Cyton serial probe — 不经过 BrainFlow，直接发 v/? 看 dongle 是否回字。

默认 **不发 b 开流**（开流后 BrainFlow 无法握手，需先 s 停流）。
加 --stream 可额外测 b 开流，退出前会自动 s 停流。
"""

from __future__ import annotations

import sys
import time

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM3"
WITH_STREAM = "--stream" in sys.argv


def main() -> int:
    try:
        import serial
    except ImportError:
        print("需要 pyserial: pip install pyserial")
        return 2

    print(f"=== Raw serial probe {PORT} @ 115200 ===")
    print("期望: 发 'v' 后应收到 OpenBCI 版本字符串（welcome characters）")
    print("若 bytes=0 → dongle 在 COM 上，但 Cyton 主板未连上或未开机")
    if not WITH_STREAM:
        print("提示: 默认不测 b 开流；需要时加 --stream\n")
    else:
        print("提示: --stream 模式，退出前会自动 s 停流\n")

    try:
        ser = serial.Serial(
            PORT,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2.0,
            write_timeout=2.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"打开 {PORT} 失败: {exc}")
        print("→ 端口被占用，或设备已拔出")
        return 1

    ser.dtr = True
    ser.rts = True
    time.sleep(0.5)
    ser.reset_input_buffer()

    cmds: list[tuple[bytes, str]] = [(b"v", "version"), (b"?", "board query")]
    if WITH_STREAM:
        cmds.append((b"b", "start stream"))

    last_len = 0
    version_ok = False
    for cmd, label in cmds:
        ser.reset_input_buffer()
        ser.write(cmd)
        time.sleep(1.0)
        n = ser.in_waiting
        raw = ser.read(n) if n else b""
        last_len = len(raw)
        if label == "version" and b"OpenBCI" in raw:
            version_ok = True
        text = raw.decode("ascii", errors="replace").strip()
        safe = text.encode("ascii", errors="backslashreplace").decode("ascii")
        print(
            f"  cmd={label!r:14} bytes={len(raw):3d}  "
            f"resp={safe[:100]!r}  hex={raw[:32].hex()}"
        )

    if WITH_STREAM:
        ser.write(b"s")
        time.sleep(0.6)
        print("  cmd='stop stream'  (auto s before close)")

    ser.close()

    if not version_ok:
        print(
            "\n[结论] COM 口能开，但 Cyton 无 OpenBCI 版本回应。\n"
            "  → 请检查: ① Cyton 电源开关 ② dongle 与板卡配对 ③ 换 USB 口/重插 dongle"
        )
        return 1

    print("\n[结论] 串口正常，Cyton 在线。")
    print("  下一步: python com3_probe.py COM3 30")
    if WITH_STREAM:
        print("  (已自动 s 停流，可直接跑 com3_probe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
