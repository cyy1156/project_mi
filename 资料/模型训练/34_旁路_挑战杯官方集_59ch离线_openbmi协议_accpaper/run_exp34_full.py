#!/usr/bin/env python3
"""Exp34 全量编排：预处理 → 轨 A 四成员 LOSO → E1f-A59 主 CSV
→ 轨 B FT(+可选 scratch) → E1f-B8 → 轨 C Leave-Next。

用法（conda cyy）：
  python D:\\MI\\资料\\模型训练\\34_旁路_挑战杯官方集_59ch离线_openbmi协议_accpaper\\run_exp34_full.py
  python ...\\run_exp34_full.py --skip-preprocess --skip-track-c
  python ...\\run_exp34_full.py --only A          # 只跑轨 A+E1f
  python ...\\run_exp34_full.py --resume          # 已有 fold 则跳过该成员重训

默认 run_tag=full_YYYYMMDD_HHMMSS，四成员共用，便于 E1f --auto-latest。
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

REPO = Path(r"D:\MI")
PRE = REPO / "code" / "preprocess_lab"
STEP_A = REPO / "code" / "train_lab" / "src" / "step" / "5070_challenge_mi_59ch_accpaper"
STEP_B = REPO / "code" / "train_lab" / "src" / "step" / "5070_challenge_mi_8ch_ft_accpaper"
OUT_A = REPO / "code" / "train_lab" / "out" / "5070_challenge_mi_59ch_accpaper"
OUT_B = REPO / "code" / "train_lab" / "out" / "5070_challenge_mi_8ch_ft_accpaper"
DOC = REPO / "资料" / "模型训练" / "34_旁路_挑战杯官方集_59ch离线_openbmi协议_accpaper"
LOG_DIR = DOC / "logs"
REG = DOC / "总结" / "结果登记表.md"

MEMBERS_A = ("shallow", "shallow_b", "eegnet", "conformer")
MEMBERS_B = ("shallow", "shallow_b", "eegnet", "conformer")


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


def _run(cmd: list[str], *, cwd: Path, log: Path, env: dict | None = None) -> None:
    _log(log, "EXEC " + " ".join(cmd) + f"  (cwd={cwd})")
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(p.stdout or "")
        f.write(f"\n[exit={p.returncode} elapsed={time.time()-t0:.1f}s]\n")
    if p.returncode != 0:
        # also print tail
        tail = "\n".join((p.stdout or "").splitlines()[-40:])
        print(tail, flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    _log(log, f"OK ({time.time()-t0:.1f}s)")


def _member_three_dir(out_root: Path, model: str, data_tag: str, run_tag: str, *, arm: str | None = None) -> Path:
    if arm:
        name = f"{model}_challenge_mi_3s_8ch_{arm}"
    else:
        name = f"{model}_challenge_mi_3s_59ch"
    return out_root / name / data_tag / f"run_{run_tag}" / "three"


def _folds_done(three_dir: Path, n_folds: int = 6) -> bool:
    if not three_dir.is_dir():
        return False
    for k in range(n_folds):
        if not (three_dir / f"fold{k}" / "best_three.pt").is_file():
            return False
        if not (three_dir / f"fold{k}" / "val_prob.npy").is_file():
            return False
    return (three_dir / "summary.json").is_file()


def stage_preprocess(py: str, log: Path) -> None:
    x59 = PRE / "out" / "challenge_mi_3s_59ch" / "challenge_X.npy"
    x8 = PRE / "out" / "challenge_mi_3s_8ch" / "challenge_X.npy"
    if x59.is_file() and x8.is_file():
        import numpy as np

        s59 = np.load(x59, mmap_mode="r").shape
        s8 = np.load(x8, mmap_mode="r").shape
        if s59[0] >= 900 and s8[0] >= 900:
            _log(log, f"skip preprocess (exists {s59} / {s8})")
            return
    _run([py, "-m", "src.datasets.challenge_mi.batch_3s", "--mode", "59"], cwd=PRE, log=log)
    _run([py, "-m", "src.datasets.challenge_mi.batch_3s", "--mode", "8"], cwd=PRE, log=log)


def stage_train_a(py: str, log: Path, run_tag: str, *, resume: bool, max_folds: int) -> None:
    for m in MEMBERS_A:
        three = _member_three_dir(OUT_A, m, "challenge_mi_3s_59ch", run_tag)
        if resume and max_folds == 0 and _folds_done(three, 6):
            _log(log, f"skip A/{m} (6 folds done @ {three})")
            continue
        script = {
            "shallow": "baseline_shallow.py",
            "shallow_b": "baseline_shallow_b.py",
            "eegnet": "baseline_eegnet.py",
            "conformer": "baseline_conformer.py",
        }[m]
        cmd = [py, script, "--run-tag", run_tag]
        if max_folds > 0:
            cmd.extend(["--max-folds", str(max_folds)])
        _run(cmd, cwd=STEP_A, log=log)


def stage_e1f_a(py: str, log: Path, run_tag: str) -> Path:
    # point explicitly to this run_tag members
    member_args = []
    for m in MEMBERS_A:
        three = _member_three_dir(OUT_A, m, "challenge_mi_3s_59ch", run_tag)
        if not (three / "fold0" / "val_prob.npy").is_file():
            raise FileNotFoundError(f"missing {three}")
        member_args.append(f"{m}={three}")
    out_json = OUT_A / "e1f_a59" / f"e1f_full_{run_tag}.json"
    _run(
        [py, "fit_e1f_a59.py", "--member-runs", *member_args, "--out", str(out_json)],
        cwd=STEP_A,
        log=log,
    )
    csv_out = OUT_A / "submissions" / f"submission_exp34_e1f_a59_sens_{run_tag}.csv"
    _run(
        [py, "predict_e1f_submission.py", "--e1f-json", str(out_json), "--out-csv", str(csv_out)],
        cwd=STEP_A,
        log=log,
    )
    return csv_out


def stage_train_b(
    py: str,
    log: Path,
    run_tag: str,
    *,
    resume: bool,
    max_folds: int,
    scratch: bool,
) -> None:
    arm = "scratch" if scratch else "ft"
    for m in MEMBERS_B:
        three = _member_three_dir(OUT_B, m, "challenge_mi_3s_8ch", run_tag, arm=arm)
        if resume and max_folds == 0 and _folds_done(three, 6):
            _log(log, f"skip B/{arm}/{m}")
            continue
        script = {
            "shallow": "baseline_shallow.py",
            "shallow_b": "baseline_shallow_b.py",
            "eegnet": "baseline_eegnet.py",
            "conformer": "baseline_conformer.py",
        }[m]
        cmd = [py, script, "--run-tag", run_tag]
        if scratch:
            cmd.append("--scratch")
        if max_folds > 0:
            cmd.extend(["--max-folds", str(max_folds)])
        _run(cmd, cwd=STEP_B, log=log)


def stage_e1f_b(py: str, log: Path, run_tag: str, *, arm: str = "ft") -> Path:
    member_args = []
    for m in MEMBERS_B:
        three = _member_three_dir(OUT_B, m, "challenge_mi_3s_8ch", run_tag, arm=arm)
        if not (three / "fold0" / "val_prob.npy").is_file():
            raise FileNotFoundError(three)
        member_args.append(f"{m}={three}")
    # fit_e1f_b8 writes its own path; also pass member-runs
    _run(
        [py, "fit_e1f_b8.py", "--arm", arm, "--member-runs", *member_args],
        cwd=STEP_B,
        log=log,
    )
    # latest for this arm
    e1f_dir = OUT_B / "e1f_b8"
    cands = sorted(e1f_dir.glob(f"e1f_{arm}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(e1f_dir)
    e1f_json = cands[0]
    csv_out = OUT_B / "submissions" / f"submission_exp34_e1f_b8_sens_{arm}_{run_tag}.csv"
    _run(
        [py, "predict_e1f_submission.py", "--e1f-json", str(e1f_json), "--out-csv", str(csv_out)],
        cwd=STEP_B,
        log=log,
    )
    return csv_out


def stage_track_c(py: str, log: Path, subjects: list[str]) -> None:
    cmd = [py, "-m", "experiment_game.tools.run_leave_next_e1f_task_ramp"]
    for s in subjects:
        cmd.extend(["--subject", s])
    cmd.extend(["--ft-scope", "all4"])
    _run(cmd, cwd=REPO, log=log)


def _read_summary_acc(three_dir: Path) -> tuple[float | None, float | None]:
    p = three_dir / "summary.json"
    if not p.is_file():
        return None, None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj.get("val_acc_mean"), obj.get("val_acc_std")


def write_status_md(run_tag: str, artifacts: dict) -> Path:
    path = DOC / "总结" / f"全量运行状态_{run_tag}.md"
    lines = [
        f"# Exp34 全量运行状态 · `{run_tag}`",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 机位：5070",
        "",
        "## 产物",
        "",
    ]
    for k, v in artifacts.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.extend(["", "## 轨 A 成员 Val Acc", ""])
    for m in MEMBERS_A:
        three = _member_three_dir(OUT_A, m, "challenge_mi_3s_59ch", run_tag)
        mu, sd = _read_summary_acc(three)
        lines.append(f"- {m}: {mu} ± {sd}" if mu is not None else f"- {m}: (missing)")
    e1f = OUT_A / "e1f_a59" / f"e1f_full_{run_tag}.json"
    if e1f.is_file():
        obj = json.loads(e1f.read_text(encoding="utf-8"))
        lines.append(
            f"- **E1f-A59**: {obj.get('val_acc_mean')} ± {obj.get('val_acc_std')}"
        )
    lines.extend(["", "## 轨 B FT", ""])
    for m in MEMBERS_B:
        three = _member_three_dir(OUT_B, m, "challenge_mi_3s_8ch", run_tag, arm="ft")
        mu, sd = _read_summary_acc(three)
        lines.append(f"- {m}: {mu} ± {sd}" if mu is not None else f"- {m}: (missing)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp34 full pipeline")
    ap.add_argument("--run-tag", default="", help="默认 full_时间戳")
    ap.add_argument("--only", choices=["all", "A", "B", "C"], default="all")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-track-c", action="store_true")
    ap.add_argument(
        "--do-b-scratch",
        action="store_true",
        help="额外跑 B8-scratch 消融（默认只跑 OpenBMI-FT）",
    )
    ap.add_argument("--resume", action="store_true", help="已有 6 折则跳过该成员")
    ap.add_argument("--max-folds", type=int, default=0, help="0=满 6 折；调试可用 1")
    ap.add_argument("--track-c-subjects", nargs="*", default=["syj0828", "fnz0828"])
    args = ap.parse_args()

    run_tag = args.run_tag.strip() or ("full_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    log = LOG_DIR / f"exp34_{run_tag}.log"
    py = _py()
    _log(log, f"=== Exp34 FULL start run_tag={run_tag} py={py} only={args.only} ===")

    artifacts: dict[str, str] = {"run_tag": run_tag, "log": str(log)}
    try:
        if not args.skip_preprocess and args.only in ("all", "A", "B"):
            stage_preprocess(py, log)

        if args.only in ("all", "A"):
            stage_train_a(py, log, run_tag, resume=args.resume, max_folds=args.max_folds)
            csv_a = stage_e1f_a(py, log, run_tag)
            artifacts["submission_A"] = str(csv_a)
            artifacts["e1f_A"] = str(OUT_A / "e1f_a59" / f"e1f_full_{run_tag}.json")

        if args.only in ("all", "B"):
            stage_train_b(
                py, log, run_tag, resume=args.resume, max_folds=args.max_folds, scratch=False
            )
            csv_b = stage_e1f_b(py, log, run_tag, arm="ft")
            artifacts["submission_B_ft"] = str(csv_b)
            if args.do_b_scratch:
                stage_train_b(
                    py, log, run_tag, resume=args.resume, max_folds=args.max_folds, scratch=True
                )
                csv_bs = stage_e1f_b(py, log, run_tag, arm="scratch")
                artifacts["submission_B_scratch"] = str(csv_bs)

        if args.only in ("all", "C") and not args.skip_track_c:
            stage_track_c(py, log, list(args.track_c_subjects))
            artifacts["track_C"] = ",".join(args.track_c_subjects)

        status = write_status_md(run_tag, artifacts)
        artifacts["status_md"] = str(status)
        _log(log, "=== Exp34 FULL DONE ===")
        _log(log, json.dumps(artifacts, ensure_ascii=False, indent=2))
        print("\nArtifacts:")
        for k, v in artifacts.items():
            print(f"  {k}: {v}")
        return 0
    except Exception as exc:
        _log(log, f"FAILED: {type(exc).__name__}: {exc}")
        write_status_md(run_tag, {**artifacts, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
