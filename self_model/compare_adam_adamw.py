"""Compare Adam vs AdamW on the same hop100 Acc_paper protocol.

Examples:
  python compare_adam_adamw.py --data-dir D:/path/to/bci2a_2s_hop100
  python compare_adam_adamw.py --max-folds 1 --max-epochs 5 --patience 3 --skip-three
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_shallow_hop100_accpaper.py"
DEFAULT_DATA = HERE.parent / "code" / "preprocess_lab" / "out" / "bci2a_2s_hop100"


def _f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pick(summary, *keys, default=None):
    if summary is None:
        return default
    cur = summary
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_one(optimizer: str, out_root: Path, args: argparse.Namespace) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN),
        "--optimizer",
        optimizer,
        "--data-dir",
        str(args.data_dir),
        "--prefix",
        args.prefix,
        "--out",
        str(out_root),
    ]
    if args.skip_three:
        cmd.append("--skip-three")
    if args.max_folds > 0:
        cmd.extend(["--max-folds", str(args.max_folds)])
    if args.max_epochs > 0:
        cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.patience > 0:
        cmd.extend(["--patience", str(args.patience)])

    log_path = out_root / "compare_launch.log"
    print(f"\n=== RUN optimizer={optimizer} ===\n{' '.join(cmd)}\n")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(HERE),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"optimizer={optimizer} failed (code={proc.returncode}). See {log_path}"
        )
    meta = out_root / "meta.json"
    if not meta.is_file():
        raise SystemExit(f"missing meta.json after run: {meta}")
    return meta


def stage_rows(name: str, adam_meta: dict, adamw_meta: dict):
    a = adam_meta.get(name)
    w = adamw_meta.get(name)
    if a is None and w is None:
        return [f"### {name}", "", "- (skipped)", ""]
    lines = [
        f"### {name}",
        "",
        "| metric | Adam | AdamW | delta (AdamW-Adam) |",
        "|---|---:|---:|---:|",
    ]
    keys = [
        ("Val Acc_paper", ("val_acc_paper_mean",)),
        ("Test Acc_paper", ("test_acc_paper_mean",)),
        ("Test BalAcc_maj", ("test_balacc_maj_mean",)),
        ("Test win BalAcc", ("test_window_balacc_mean",)),
        ("Test win Sens", ("test_window_sensitivity_mean",)),
        ("Test win Spec", ("test_window_specificity_mean",)),
        ("Test win F1", ("test_window_f1_mean",)),
        ("Test trial Spec", ("test_trial_specificity_mean",)),
        ("Test F1-macro maj", ("test_f1_macro_maj_mean",)),
        ("Test win F1-macro", ("test_window_f1_macro_mean",)),
    ]
    for label, path in keys:
        va = _pick(a, *path)
        vw = _pick(w, *path)
        if va is None and vw is None:
            continue
        try:
            delta = float(vw) - float(va) if va is not None and vw is not None else None
        except (TypeError, ValueError):
            delta = None
        d = "-" if delta is None else f"{delta:+.4f}"
        lines.append(f"| {label} | {_f(va)} | {_f(vw)} | {d} |")
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Adam vs AdamW (same protocol)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--prefix", default="bci2a")
    p.add_argument("--skip-three", action="store_true")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    need = [
        f"{args.prefix}_X.npy",
        f"{args.prefix}_y_task.npy",
        f"{args.prefix}_subjects.npy",
        f"{args.prefix}_trial_id.npy",
    ]
    if not args.skip_three:
        need.append(f"{args.prefix}_y_three.npy")
    missing = [n for n in need if not (args.data_dir / n).is_file()]
    if missing:
        raise SystemExit(
            f"Data not found under {args.data_dir}\n"
            f"Missing: {missing}\n"
            f"Prepare bci2a_2s_hop100 first, then re-run."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.out or (HERE / "out" / f"optimizer_compare_{stamp}")
    root.mkdir(parents=True, exist_ok=True)

    adam_meta_path = run_one("adam", root / "adam", args)
    adamw_meta_path = run_one("adamw", root / "adamw", args)
    adam_meta = json.loads(adam_meta_path.read_text(encoding="utf-8"))
    adamw_meta = json.loads(adamw_meta_path.read_text(encoding="utf-8"))

    def score(meta, stage):
        s = meta.get(stage)
        return None if not s else float(s["test_acc_paper_mean"])

    task_a, task_w = score(adam_meta, "task"), score(adamw_meta, "task")
    three_a, three_w = score(adam_meta, "three"), score(adamw_meta, "three")

    def verdict(a, w, name):
        if a is None or w is None:
            return f"{name}: incomplete"
        if abs(a - w) < 1e-4:
            return f"{name}: tie (Acc_paper)"
        winner = "AdamW" if w > a else "Adam"
        return f"{name}: {winner} better (Test Acc_paper)"

    md = [
        f"# Optimizer compare: Adam vs AdamW ({stamp})",
        "",
        "- Protocol: hop100 Acc_paper, same seed/hparams; only optimizer changes",
        f"- data: `{args.data_dir}`",
        f"- skip_three: `{args.skip_three}` | max_folds: `{args.max_folds}` | "
        f"max_epochs: `{args.max_epochs}` | patience: `{args.patience}`",
        f"- adam run: `{adam_meta_path.parent}`",
        f"- adamw run: `{adamw_meta_path.parent}`",
        "",
        "## Verdict",
        "",
        f"- {verdict(task_a, task_w, 'Task')}",
        f"- {verdict(three_a, three_w, 'Three')}",
        "",
        *stage_rows("task", adam_meta, adamw_meta),
        *stage_rows("three", adam_meta, adamw_meta),
        "## Notes",
        "",
        "- Primary metric: **Test Acc_paper**",
        "- Also compare BalAcc / Sens / Spec / F1 when present",
        "- Full compare: omit smoke flags (5-fold, both stages)",
        "",
    ]
    md_path = root / f"{stamp}_adam_vs_adamw.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    (root / f"{stamp}_adam_vs_adamw.json").write_text(
        json.dumps(
            {
                "stamp": stamp,
                "data_dir": str(args.data_dir),
                "adam": adam_meta,
                "adamw": adamw_meta,
                "verdict_task": verdict(task_a, task_w, "Task"),
                "verdict_three": verdict(three_a, three_w, "Three"),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nCompare done.\nMD: {md_path}\nroot: {root}")


if __name__ == "__main__":
    main()
