#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exp35 全量编排：轨 R → F/M/S（含 CSV + 显著性）→ D；轨 H 默认跳过。

用法（conda cyy）：
  python run_exp35_full.py
  python run_exp35_full.py --resume
  python run_exp35_full.py --with-h            # 显式开轨 H
  python run_exp35_full.py --skip-csv
  python run_exp35_full.py --h-fold0-only      # 仅当 --with-h
  python run_exp35_full.py --only p0           # 仅 P0
  python run_exp35_full.py --only d
  python run_exp35_full.py --only h            # 等价于只跑 H（隐含 with-h）

默认 run_tag=full_YYYYMMDD_HHMMSS；日志写到方案目录 logs/。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
if not (REPO / "code").is_dir():
    REPO = Path(r"D:\MI")

STEP = REPO / "code" / "train_lab" / "src" / "step" / "5070_challenge_rankflip_accpaper"
OUT = REPO / "code" / "train_lab" / "out" / "5070_challenge_rankflip_accpaper"
DOC = (
    REPO
    / "资料"
    / "模型训练"
    / "35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper"
)
LOG_DIR = DOC / "logs"
STATE_DIR = OUT / "_full_runs"

PREFER_TAG = "full_20260902_1930"
H_SMOKE_ARMS = ("H0", "H1", "H2", "H3", "H4", "H5")


def _py() -> str:
    cand = Path(os.path.expandvars(r"%USERPROFILE%\.conda\envs\cyy\python.exe"))
    if cand.is_file():
        return str(cand)
    return sys.executable


def _log(fp: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(cmd: list[str], *, cwd: Path, log: Path) -> None:
    _log(log, "EXEC " + " ".join(cmd) + f"  (cwd={cwd})")
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(p.stdout or "")
        f.write(f"\n[exit={p.returncode} elapsed={time.time()-t0:.1f}s]\n")
    # 同步关键台尾部，便于盯进度
    tail = "\n".join((p.stdout or "").splitlines()[-30:])
    if tail.strip():
        print(tail, flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    _log(log, f"OK ({time.time()-t0:.1f}s)")


def _marker(state_root: Path, name: str) -> Path:
    return state_root / f"{name}.done"


def _done(state_root: Path, name: str) -> bool:
    return _marker(state_root, name).is_file()


def _mark(state_root: Path, name: str, payload: dict | None = None) -> None:
    m = _marker(state_root, name)
    m.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "done_at": datetime.now().isoformat(timespec="seconds"),
        **(payload or {}),
    }
    m.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _py_script(name: str, *args: str) -> list[str]:
    return [_py(), str(STEP / name), *args]


def stage_r(log: Path, state: Path, resume: bool) -> None:
    if resume and _done(state, "R"):
        _log(log, "SKIP R (resume)")
        return
    _run(_py_script("write_ranking_doc.py"), cwd=STEP, log=log)
    _mark(state, "R")


def stage_export_a59(log: Path, state: Path, resume: bool, prefer: str) -> None:
    if resume and _done(state, "export_a59"):
        _log(log, "SKIP export_a59 (resume)")
        return
    _run(
        _py_script(
            "export_member_probs.py",
            "--track",
            "a59",
            "--prefer-tag",
            prefer,
            "--skip-test",
        ),
        cwd=STEP,
        log=log,
    )
    _mark(state, "export_a59")


def stage_fm(log: Path, state: Path, resume: bool, prefer: str) -> None:
    if resume and _done(state, "FM"):
        _log(log, "SKIP FM (resume)")
        return
    _run(
        _py_script("replay_fusion_grid.py", "--suite", "FM", "--prefer-tag", prefer),
        cwd=STEP,
        log=log,
    )
    _mark(state, "FM")


def stage_s(log: Path, state: Path, resume: bool, prefer: str, write_csv: bool) -> None:
    key = "S_csv" if write_csv else "S_decision"
    if resume and _done(state, key):
        _log(log, f"SKIP {key} (resume)")
        return
    replay = OUT / "replay" / "replay_FM_latest.json"
    if not replay.is_file():
        raise FileNotFoundError(replay)
    args = ["--replay-json", str(replay), "--prefer-tag", prefer]
    if write_csv:
        args.append("--write-csv")
    _run(_py_script("make_submission_candidates.py", *args), cwd=STEP, log=log)
    _run(_py_script("paired_sig_test.py", "--replay-json", str(replay)), cwd=STEP, log=log)
    _run(
        _py_script("write_ranking_doc.py", "--replay-json", str(replay)),
        cwd=STEP,
        log=log,
    )
    _mark(state, key, {"replay": str(replay), "write_csv": write_csv})


def stage_d(log: Path, state: Path, resume: bool, prefer: str) -> None:
    if resume and _done(state, "D"):
        _log(log, "SKIP D (resume)")
        return
    for arm in ("ft", "scratch"):
        _run(
            _py_script(
                "export_member_probs.py",
                "--track",
                "b8",
                "--arm",
                arm,
                "--prefer-tag",
                prefer,
                "--skip-test",
            ),
            cwd=STEP,
            log=log,
        )
    _run(
        _py_script("replay_fusion_grid.py", "--suite", "D", "--prefer-tag", prefer),
        cwd=STEP,
        log=log,
    )
    _mark(state, "D")


def _latest_h_gate(run_tag: str) -> Path | None:
    # run_shallow_recipe_h 写 gate_{tag}.json，tag 含各臂后缀时以目录最新为准
    h_dir = OUT / "H"
    if not h_dir.is_dir():
        return None
    cands = sorted(h_dir.glob("gate_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def stage_h(
    log: Path,
    state: Path,
    resume: bool,
    run_tag: str,
    *,
    fold0_only: bool,
    max_full_extra: int,
) -> None:
    """1) fold0 冒烟 H0–H5  2) H0 满折  3) 过闸臂最多 max_full_extra 个满折。"""
    smoke_key = "H_smoke"
    if not (resume and _done(state, smoke_key)):
        arms = ",".join(H_SMOKE_ARMS)
        _run(
            _py_script(
                "run_shallow_recipe_h.py",
                "--arm",
                arms,
                "--max-folds",
                "1",
                "--run-tag",
                f"{run_tag}_smoke",
            ),
            cwd=STEP,
            log=log,
        )
        _mark(state, smoke_key)
    else:
        _log(log, "SKIP H_smoke (resume)")

    if fold0_only:
        _log(log, "H fold0-only：跳过满折")
        return

    gate_path = _latest_h_gate(run_tag)
    promote: list[str] = []
    if gate_path and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        promote = list((gate.get("fold0_gate") or {}).get("promote_full") or [])
        _log(log, f"H gate file={gate_path.name} promote_full={promote}")
    else:
        _log(log, "WARN: 未找到 H gate json，仅跑 H0 满折")

    # 满折列表：H0 必跑 + 过闸臂（排除 H0）截断
    full_arms: list[str] = []
    h0_key = "H_full_H0"
    if not (resume and _done(state, h0_key)):
        full_arms.append("H0")
    else:
        _log(log, "SKIP H_full_H0 (resume)")

    extra = [a for a in promote if a != "H0"][: max(0, int(max_full_extra))]
    for a in extra:
        key = f"H_full_{a}"
        if resume and _done(state, key):
            _log(log, f"SKIP {key} (resume)")
            continue
        full_arms.append(a)

    for arm in full_arms:
        key = f"H_full_{arm}"
        _run(
            _py_script(
                "run_shallow_recipe_h.py",
                "--arm",
                arm,
                "--max-folds",
                "0",
                "--run-tag",
                f"{run_tag}_full_{arm}",
            ),
            cwd=STEP,
            log=log,
        )
        _mark(state, key, {"arm": arm, "max_folds": 0})

    _mark(
        state,
        "H",
        {
            "promote_full": promote,
            "full_ran": full_arms,
            "gate": str(gate_path) if gate_path else None,
        },
    )


def stage_finalize(log: Path, state: Path, run_tag: str) -> None:
    """汇总 latest 指针到 metrics JSON。"""
    if _done(state, "finalize") and False:
        return
    summary = {
        "experiment": 35,
        "run_tag": run_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "replay_FM": str(OUT / "replay" / "replay_FM_latest.json"),
            "replay_D": str(OUT / "replay" / "replay_D_latest.json"),
            "decision": str(OUT / "submissions" / "decision_latest.json"),
            "ranking_doc": str(DOC / "总结" / "跨域三分类成员排名对照表.md"),
            "registry": str(DOC / "总结" / "结果登记表.md"),
        },
        "markers": sorted(p.stem for p in state.glob("*.done")),
    }
    # 附带关键决策
    dec = OUT / "submissions" / "decision_latest.json"
    if dec.is_file():
        summary["decision"] = json.loads(dec.read_text(encoding="utf-8"))
    fm = OUT / "replay" / "replay_FM_latest.json"
    if fm.is_file():
        payload = json.loads(fm.read_text(encoding="utf-8"))
        summary["fm_decision"] = payload.get("decision")
        summary["fm_arm_means"] = {
            k: {"val_acc_mean": v.get("val_acc_mean"), "val_acc_std": v.get("val_acc_std")}
            for k, v in (payload.get("results") or {}).items()
        }
    dpath = OUT / "replay" / "replay_D_latest.json"
    if dpath.is_file():
        payload = json.loads(dpath.read_text(encoding="utf-8"))
        summary["d_arm_means"] = {
            k: {"val_acc_mean": v.get("val_acc_mean"), "val_acc_std": v.get("val_acc_std")}
            for k, v in (payload.get("results") or {}).items()
        }

    out = DOC / "总结" / f"metrics_full_{run_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = DOC / "总结" / "metrics_full_latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    _log(log, f"wrote {out}")
    _mark(state, "finalize", {"metrics": str(out)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", default="", help="默认 full_YYYYMMDD_HHMMSS")
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    ap.add_argument("--resume", action="store_true", help="跳过已完成 marker")
    ap.add_argument("--with-h", action="store_true", help="v0.2：显式启动轨 H（默认跳过）")
    ap.add_argument("--skip-h", action="store_true", help="兼容旧开关；默认已跳过 H")
    ap.add_argument("--skip-d", action="store_true")
    ap.add_argument("--skip-csv", action="store_true", help="轨 S 不写 test CSV")
    ap.add_argument("--h-fold0-only", action="store_true", help="H 只 fold0 冒烟")
    ap.add_argument(
        "--h-max-full-extra",
        type=int,
        default=1,
        help="除 H0 满折外，最多再满折几个过闸臂（默认 1，合计≤2）",
    )
    ap.add_argument(
        "--only",
        choices=("all", "p0", "d", "h"),
        default="all",
        help="只跑某一段",
    )
    args = ap.parse_args()

    run_tag = args.run_tag.strip() or f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state = STATE_DIR / run_tag
    state.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"exp35_full_{run_tag}.log"

    _log(log, f"=== Exp35 FULL start run_tag={run_tag} py={_py()} ===")
    _log(log, f"STEP={STEP}")
    _log(log, f"OUT={OUT}")
    _log(log, f"state={state}")

    t_all = time.time()
    try:
        do_p0 = args.only in ("all", "p0")
        do_d = args.only in ("all", "d") and not args.skip_d
        do_h = args.only in ("all", "h") and (args.with_h or args.only == "h") and not args.skip_h

        if do_p0:
            stage_r(log, state, args.resume)
            stage_export_a59(log, state, args.resume, args.prefer_tag)
            stage_fm(log, state, args.resume, args.prefer_tag)
            stage_s(
                log,
                state,
                args.resume,
                args.prefer_tag,
                write_csv=not args.skip_csv,
            )

        if do_d:
            stage_d(log, state, args.resume, args.prefer_tag)

        if do_h:
            stage_h(
                log,
                state,
                args.resume,
                run_tag,
                fold0_only=bool(args.h_fold0_only),
                max_full_extra=int(args.h_max_full_extra),
            )

        stage_finalize(log, state, run_tag)
    except Exception as exc:
        _log(log, f"FAILED: {exc}")
        raise

    _log(log, f"=== Exp35 FULL DONE elapsed={time.time()-t_all:.1f}s ===")
    print(f"\nLog: {log}")
    print(f"Metrics: {DOC / '总结' / f'metrics_full_{run_tag}.json'}")
    print(f"Out: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
