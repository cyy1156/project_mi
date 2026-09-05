#!/usr/bin/env python3
"""
Phase 4：会话目录 → (N,1,8,500) + y_task / y_three。

用法（仓库根 .venv）:

  .\\.venv\\Scripts\\python.exe ^
    -m experiment_game.tools.run_phase4_epochs ^
    --session experiment_game\\data\\sessions\\sub01_ses_p1_20260722_110447

输出默认：experiment_game/data/epochs/<session_name>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.offline.pipeline import preprocess_session, save_bundle


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase4：events+eeg → epochs npy")
    p.add_argument(
        "--session",
        type=Path,
        required=True,
        help="会话目录（含 eeg.csv + events.jsonl）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 data/epochs/<session名>）",
    )
    p.add_argument(
        "--phases",
        default="acquire",
        help="逗号分隔 phase 过滤，默认 acquire；传 all 不过滤",
    )
    p.add_argument("--no-filter", action="store_true", help="跳过 CAR/陷波/带通")
    p.add_argument("--no-split", action="store_true", help="不写 train_/val_")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--window-mode",
        choices=["fixed", "slide"],
        default="fixed",
        help="fixed=每阶段起点 1 窗；slide=阶段区间内滑窗",
    )
    p.add_argument(
        "--win-sec",
        type=float,
        default=2.0,
        help="窗长（秒），默认 2.0 → 500@250Hz",
    )
    p.add_argument(
        "--hop-ms",
        type=float,
        default=100.0,
        help="滑窗步长（毫秒），仅 slide 模式生效",
    )
    p.add_argument(
        "--baseline-s",
        type=float,
        default=0.5,
        help="窗内基线校正时长（秒）",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有输出目录中的 npy",
    )
    args = p.parse_args(argv)

    session = args.session
    if not session.is_absolute():
        session = (_REPO_ROOT / session).resolve()
    if not session.is_dir():
        print(f"会话目录不存在: {session}", file=sys.stderr)
        return 1

    if args.phases.strip().lower() == "all":
        phases = None
    else:
        phases = [s.strip() for s in args.phases.split(",") if s.strip()]

    out = args.out
    if out is None:
        out = (
            _REPO_ROOT
            / "experiment_game"
            / "data"
            / "epochs"
        )
        name = session.name
        if args.window_mode == "slide":
            name += f"_slide_w{args.win_sec:g}s_h{args.hop_ms:g}ms"
        elif abs(args.win_sec - 2.0) > 1e-9:
            name += f"_w{args.win_sec:g}s"
        out = out / name
    elif not out.is_absolute():
        out = (_REPO_ROOT / out).resolve()

    out.mkdir(parents=True, exist_ok=True)
    if (out / "X.npy").is_file() and not args.force:
        print(f"输出已存在: {out}（加 --force 覆盖）", file=sys.stderr)
        return 1

    print(f"session={session}")
    print(f"phases={phases}")
    print(
        f"window: mode={args.window_mode} win={args.win_sec}s "
        f"hop={args.hop_ms}ms baseline={args.baseline_s}s"
    )
    bundle = preprocess_session(
        session,
        phases=phases,
        apply_filter=not args.no_filter,
        window_mode=args.window_mode,
        win_sec=args.win_sec,
        hop_ms=args.hop_ms,
        baseline_s=args.baseline_s,
    )
    save_bundle(
        bundle,
        out,
        also_train_val=not args.no_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    s = bundle.summary()
    print(f"out={out}")
    print(f"N={s['n']} X={s['X_shape']}")
    print(f"y_task={s['y_task_counts']} y_three={s['y_three_counts']}")
    if s.get("skipped"):
        print(f"skipped={len(s['skipped'])}")
    if s["n"] == 0:
        print("警告：未切出任何窗（检查 eeg 覆盖与 phase 过滤）", file=sys.stderr)
        return 2
    print("PHASE4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
