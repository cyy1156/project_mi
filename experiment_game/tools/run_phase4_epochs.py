#!/usr/bin/env python3
"""
Phase 4：会话目录 → (N,1,8,1000) + y_task / y_three。

用法（仓库根，lsl_connect venv）:

  .\\collect_data\\...\\.venv\\Scripts\\python.exe ^
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
            / session.name
        )
    elif not out.is_absolute():
        out = (_REPO_ROOT / out).resolve()

    print(f"session={session}")
    print(f"phases={phases}")
    bundle = preprocess_session(
        session,
        phases=phases,
        apply_filter=not args.no_filter,
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
