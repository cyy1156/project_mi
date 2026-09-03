# -*- coding: utf-8 -*-
"""Exp35 编排：R → export → F/M 回放 → S 决策（可选 CSV）→ 可选 D / H。

用法：
  python run_exp35.py --stage p0
  python run_exp35.py --stage p0 --write-csv
  python run_exp35.py --stage d
  python run_exp35.py --stage h --h-arms H0,H1 --max-folds 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_STEP = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(_STEP / script), *extra]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_STEP))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=("r", "export", "fm", "s", "d", "h", "p0", "all"),
        default="p0",
    )
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--skip-test-export", action="store_true", help="P0 回放不需要 test dump")
    ap.add_argument("--h-arms", default="H0,H1,H2,H3,H4")
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--prefer-tag", default="full_20260902_1930")
    args = ap.parse_args()

    prefer = ["--prefer-tag", args.prefer_tag]

    if args.stage in ("r", "p0", "all"):
        _run("write_ranking_doc.py", [])

    if args.stage in ("export", "p0", "all"):
        exp_args = ["--track", "a59", *prefer]
        if args.skip_test_export or args.stage == "p0":
            exp_args.append("--skip-test")
        _run("export_member_probs.py", exp_args)

    if args.stage in ("fm", "p0", "all"):
        _run("replay_fusion_grid.py", ["--suite", "FM", *prefer])

    if args.stage in ("s", "p0", "all"):
        replay = (
            Path(__file__).resolve().parents[3]
            / "out"
            / "5070_challenge_rankflip_accpaper"
            / "replay"
            / "replay_FM_latest.json"
        )
        s_args = ["--replay-json", str(replay), *prefer]
        if args.write_csv:
            s_args.append("--write-csv")
        _run("make_submission_candidates.py", s_args)
        _run("paired_sig_test.py", ["--replay-json", str(replay)])
        # 用回放结果刷新排名文档权重表（含常驻路径）
        _run("write_ranking_doc.py", ["--replay-json", str(replay)])

    if args.stage in ("d", "all"):
        _run("export_member_probs.py", ["--track", "b8", "--arm", "ft", "--skip-test", *prefer])
        _run("export_member_probs.py", ["--track", "b8", "--arm", "scratch", "--skip-test", *prefer])
        _run("replay_fusion_grid.py", ["--suite", "D", *prefer])

    if args.stage in ("h", "all"):
        _run(
            "run_shallow_recipe_h.py",
            ["--arm", args.h_arms, "--max-folds", str(args.max_folds)],
        )

    print("DONE stage=", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
