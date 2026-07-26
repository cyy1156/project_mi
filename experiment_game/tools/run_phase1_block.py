#!/usr/bin/env python3
"""
Phase 1：无画面正式 block —— synthetic/真机 + 试次状态机 + events/eeg 落盘。

用法（在仓库根 d:\\cyy\\MI，使用 lsl_connect 的 venv）:

  .\\collect_data\\LSL_connect_model\\LSL_connect_model\\.venv\\Scripts\\python.exe ^
    -m experiment_game.tools.run_phase1_block --trials 20 --synthetic

按 Enter 开始；Ctrl+C 可中断（会尽量 stop 录制）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 保证仓库根在 path 上
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.acquisition import AcquisitionFacade, DEFAULT_CHANNEL_LABELS
from experiment_game.experiment import (
    EventLogger,
    MarkerPublisher,
    SessionMeta,
    TrialStateMachine,
    create_session_dir,
    update_session_meta,
    write_session_meta,
)


def _label_name(lab: int) -> str:
    return {0: "Rest", 1: "Left", 2: "Right"}.get(lab, str(lab))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase1 无画面正式采集 block")
    p.add_argument("--trials", type=int, default=20, help="试次数（验收 ≥20）")
    p.add_argument("--subject", default="sub01")
    p.add_argument("--session", default="ses01")
    p.add_argument("--synthetic", action="store_true", default=True)
    p.add_argument("--real", action="store_true", help="使用真机 Cyton（关闭 synthetic）")
    p.add_argument("--port", default="COM3", help="真机串口")
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "experiment_game" / "data" / "sessions",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-markers", action="store_true", help="不创建 LSL Marker 流")
    p.add_argument("--yes", action="store_true", help="跳过按键确认")
    args = p.parse_args(argv)

    use_synthetic = not args.real
    if args.trials < 1:
        print("trials 必须 ≥ 1", file=sys.stderr)
        return 2

    print("=== Phase 1 无画面正式 block ===")
    print(f"subject={args.subject} session={args.session} trials={args.trials}")
    print(f"board={'synthetic' if use_synthetic else args.port}")
    print(f"channels={','.join(DEFAULT_CHANNEL_LABELS)}")
    print(f"out={args.out}")
    if not args.yes:
        input("准备好后按 Enter 开始…")

    paths = create_session_dir(args.out, args.subject, args.session)
    meta = SessionMeta(
        subject_id=args.subject,
        session_id=args.session,
        phase_mode="acquire",
        use_synthetic=use_synthetic,
        trial_count=args.trials,
        object="cup",
        scene="home_desk",
    )
    write_session_meta(paths.meta_json, meta)

    events = EventLogger(paths.events_jsonl)
    markers = MarkerPublisher(enabled=not args.no_markers)
    acq = AcquisitionFacade(
        use_synthetic=use_synthetic,
        serial_port=args.port,
    )

    def on_stage(stage: str, ctx, label) -> None:
        lab = "" if label is None else f" {_label_name(int(label))}"
        print(f"  [trial {ctx.trial_id:02d}] {stage}{lab}", flush=True)

    sm = TrialStateMachine(events, markers, on_stage=on_stage)

    try:
        acq.create()
        print("启动采集与录制…")
        acq.start(paths.eeg_csv)
        # 给 LSL / recorder 一点缓冲时间
        time.sleep(1.5)

        events.emit(
            "session_start",
            subject_id=args.subject,
            session_id=args.session,
            phase="acquire",
            trial_id=None,
            label=None,
        )
        markers.push(
            f"session_start|subject={args.subject}|session={args.session}|phase=acquire"
        )

        print("开始正式 block…")
        schedule = sm.run_block(
            args.trials,
            object_name="cup",
            scene="home_desk",
            phase="acquire",
            seed=args.seed,
        )
        print(f"标签顺序: {schedule}")

        events.emit(
            "session_end",
            subject_id=args.subject,
            session_id=args.session,
            phase="acquire",
            trial_id=None,
            label=None,
        )
        markers.push("session_end|phase=acquire")
    except KeyboardInterrupt:
        print("\n用户中断，正在停止…", file=sys.stderr)
        events.emit("session_end", subject_id=args.subject, session_id=args.session, phase="acquire")
        return 130
    finally:
        try:
            report = acq.stop()
            print(f"录制停止: {report.get('message')}")
        except Exception as exc:  # noqa: BLE001
            print(f"停止录制异常: {exc}", file=sys.stderr)
        acq.shutdown()
        events.close()
        markers.close()
        update_session_meta(
            paths.meta_json,
            trial_count=args.trials,
            session_dir=str(paths.root),
        )

    print(f"会话目录: {paths.root}")
    print("可用 verify 脚本检查对齐:")
    print(
        f"  python -m experiment_game.tools.verify_phase1_alignment --session {paths.root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
