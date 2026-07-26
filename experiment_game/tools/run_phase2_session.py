#!/usr/bin/env python3
"""
Phase 2：诱导 MVP 会话（适应 → 学习 → 准入 → 正式）+ WebSocket + 静态页。

用法（仓库根，lsl_connect venv）:

  .\\collect_data\\LSL_connect_model\\LSL_connect_model\\.venv\\Scripts\\python.exe ^
    -m experiment_game.tools.run_phase2_session --acquire-trials 4 --yes

浏览器打开终端打印的 http://127.0.0.1:8080/ ，按页面提示继续。
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.acquisition import AcquisitionFacade, DEFAULT_CHANNEL_LABELS
from experiment_game.experiment import (
    EventLogger,
    MarkerPublisher,
    SessionMeta,
    create_session_dir,
    update_session_meta,
    write_session_meta,
)
from experiment_game.experiment.http_static import StaticServer
from experiment_game.experiment.session_runner import Phase2Config, SessionRunner
from experiment_game.experiment.timing import DEFAULT_TIMING, FAST_TIMING
from experiment_game.experiment.ws_bridge import WsBridge

_WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase2 诱导 MVP 会话")
    p.add_argument("--subject", default="sub01")
    p.add_argument("--session", default="ses_p2")
    p.add_argument("--acquire-trials", type=int, default=4)
    p.add_argument("--learn-trials", type=int, default=2, help="每个学习 Step 的试次数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", default=True)
    p.add_argument("--real", action="store_true")
    p.add_argument("--port", default="COM3")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--no-acq", action="store_true", help="只跑画面流程，不采 EEG")
    p.add_argument("--skip-adapt", action="store_true")
    p.add_argument("--skip-learn", action="store_true")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--auto-continue", action="store_true", help="无人值守联调（跳过点击）")
    p.add_argument("--fast", action="store_true", help="缩短试次时长（联调/验收）")
    p.add_argument("--no-rotate-objects", action="store_true", help="正式段不换物")
    p.add_argument("--no-rotate-scenes", action="store_true", help="正式段不换景")
    p.add_argument("--open-browser", action="store_true", default=True)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "experiment_game" / "data" / "sessions",
    )
    args = p.parse_args(argv)

    use_synthetic = not args.real
    open_browser = args.open_browser and not args.no_browser

    print("=== Phase 2 诱导 MVP ===")
    print(f"web={_WEB_ROOT}")
    print(f"acquire_trials={args.acquire_trials} learn_per_step={args.learn_trials}")
    print(f"acq={'off' if args.no_acq else ('synthetic' if use_synthetic else args.port)}")
    if not args.yes and not args.auto_continue:
        input("准备好后按 Enter 启动服务…")

    paths = create_session_dir(args.out, args.subject, args.session)
    meta = SessionMeta(
        subject_id=args.subject,
        session_id=args.session,
        phase_mode="phase2_full",
        use_synthetic=use_synthetic if not args.no_acq else True,
        trial_count=args.acquire_trials,
        object="cup",
        scene="home_desk",
        notes="phase2_induction_mvp",
    )
    write_session_meta(paths.meta_json, meta)

    events = EventLogger(paths.events_jsonl)
    markers = MarkerPublisher(enabled=not args.no_acq)
    bridge = WsBridge(port=args.ws_port)
    http = StaticServer(_WEB_ROOT, port=args.http_port)
    acq: AcquisitionFacade | None = None

    cfg = Phase2Config(
        acquire_trials=args.acquire_trials,
        learn_trials_per_step=args.learn_trials,
        seed=args.seed,
        skip_adapt=args.skip_adapt,
        skip_learn=args.skip_learn,
        skip_gate=args.skip_gate,
        auto_continue=args.auto_continue,
        rotate_objects=not args.no_rotate_objects,
        rotate_scenes=not args.no_rotate_scenes,
    )
    timing = FAST_TIMING if args.fast else DEFAULT_TIMING
    if args.fast:
        print(f"FAST 时序启用，单 trial ≈ {timing.total_s:.1f}s（仅联调）")
    runner = SessionRunner(events, markers, bridge, timing=timing, config=cfg)

    try:
        bridge.start()
        http.start()
        print(f"诱导页: {http.url}")
        print(f"WebSocket: {bridge.url}")
        if open_browser and not args.auto_continue:
            webbrowser.open(http.url)

        if not args.no_acq:
            acq = AcquisitionFacade(use_synthetic=use_synthetic, serial_port=args.port)
            acq.create()
            print("启动采集与录制…")
            acq.start(paths.eeg_csv)
            time.sleep(1.5)

        events.emit(
            "session_start",
            subject_id=args.subject,
            session_id=args.session,
            phase="phase2",
        )
        markers.push(
            f"session_start|subject={args.subject}|session={args.session}|phase=phase2"
        )

        if not args.auto_continue:
            runner.wait_browser_ready()
        runner.run_all()

        events.emit(
            "session_end",
            subject_id=args.subject,
            session_id=args.session,
            phase="phase2",
        )
        markers.push("session_end|phase=phase2")
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        bridge.broadcast({"type": "session", "status": "error", "message": str(exc)})
        return 1
    finally:
        if acq is not None:
            try:
                report = acq.stop()
                print(f"录制停止: {report.get('message')}")
            except Exception as exc:  # noqa: BLE001
                print(f"停止录制异常: {exc}", file=sys.stderr)
            acq.shutdown()
        events.close()
        markers.close()
        http.stop()
        bridge.stop()
        update_session_meta(paths.meta_json, session_dir=str(paths.root))

    print(f"会话目录: {paths.root}")
    if not args.no_acq:
        print(
            "校验: python -m experiment_game.tools.verify_phase1_alignment "
            f"--session {paths.root} --min-trials {args.acquire_trials}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
