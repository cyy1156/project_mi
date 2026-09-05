"""E1f-B8：用法同 A59，换 OUT 与成员目录解析。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
sys.path.insert(0, str(_STEP))

from e1f_core import fit_e1f, fuse_with_config  # noqa: E402
from metrics_three import three_class_report  # noqa: E402
from shared_hparams import OUT_ROOT_TAG  # noqa: E402

MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")


def _find_latest(model_name: str, arm: str) -> Path | None:
    root = (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / f"{model_name}_challenge_mi_3s_8ch_{arm}"
        / "challenge_mi_3s_8ch"
    )
    if not root.is_dir():
        return None
    runs = sorted(root.glob("run_*/three"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def fit_one_fold(member_dirs: dict[str, Path], fold: int) -> dict:
    names, probs, y_ref = [], [], None
    for name in MEMBER_KEYS:
        if name not in member_dirs:
            continue
        fd = member_dirs[name] / f"fold{fold}"
        if not (fd / "val_prob.npy").is_file():
            continue
        p = np.load(fd / "val_prob.npy")
        y = np.load(fd / "val_y.npy")
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise RuntimeError(f"fold{fold} y mismatch {name}")
        names.append(name)
        probs.append(p)
    if len(names) < 2:
        raise RuntimeError(f"fold{fold} members={names}")
    cfg = fit_e1f(names, probs, y_ref)
    fused = fuse_with_config(probs, cfg)
    return {
        "fold": fold,
        "config": cfg.to_dict(),
        "val_metrics": three_class_report(y_ref, fused.argmax(1)),
        "members": names,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ft", choices=["ft", "scratch"])
    ap.add_argument("--auto-latest", action="store_true")
    ap.add_argument("--member-runs", nargs="*", default=[])
    ap.add_argument("--max-folds", type=int, default=6)
    args = ap.parse_args()

    if args.auto_latest:
        member_dirs = {n: p for n in MEMBER_KEYS if (p := _find_latest(n, args.arm))}
    else:
        member_dirs = {}
        for it in args.member_runs:
            k, v = it.split("=", 1)
            member_dirs[k.strip()] = Path(v.strip())
    if len(member_dirs) < 2:
        raise SystemExit("成员不足")

    folds = []
    for fold in range(args.max_folds):
        if not all((d / f"fold{fold}" / "val_prob.npy").is_file() for d in member_dirs.values()):
            break
        rec = fit_one_fold(member_dirs, fold)
        folds.append(rec)
        print(f"fold{fold} acc={rec['val_metrics']['acc']:.4f}")

    accs = [float(f["val_metrics"]["acc"]) for f in folds]
    summary = {
        "experiment": 34,
        "track": "B8",
        "arm": args.arm,
        "member_dirs": {k: str(v) for k, v in member_dirs.items()},
        "n_folds": len(folds),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": folds,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    out = (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / "e1f_b8"
        / f"e1f_{args.arm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
