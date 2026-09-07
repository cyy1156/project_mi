"""从各成员 run 的 fold*/val_prob.npy 拟合 E1f-A59，并汇总 Val Acc。

用法：
  python fit_e1f_a59.py --member-runs shallow=PATH/three shallow_b=PATH/three eegnet=... conformer=...
  python fit_e1f_a59.py --auto-latest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from e1f_core import fit_e1f, fuse_with_config  # noqa: E402
from metrics_three import three_class_report  # noqa: E402
from shared_hparams import OUT_ROOT_TAG  # noqa: E402

MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")


def _parse_member_runs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"期望 name=path，收到 {it!r}")
        name, path = it.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def _find_latest_three(model_name: str) -> Path | None:
    root = (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / f"{model_name}_challenge_mi_3s_59ch"
        / "challenge_mi_3s_59ch"
    )
    if not root.is_dir():
        return None
    runs = sorted(root.glob("run_*/three"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def fit_one_fold(member_dirs: dict[str, Path], fold: int) -> dict:
    names: list[str] = []
    probs: list[np.ndarray] = []
    y_ref: np.ndarray | None = None
    idx_ref: np.ndarray | None = None
    for name in MEMBER_KEYS:
        if name not in member_dirs:
            continue
        fd = member_dirs[name] / f"fold{fold}"
        pp = fd / "val_prob.npy"
        yp = fd / "val_y.npy"
        ip = fd / "val_idx.npy"
        if not pp.is_file():
            continue
        p = np.load(pp)
        y = np.load(yp)
        idx = np.load(ip) if ip.is_file() else None
        if y_ref is None:
            y_ref = y
            idx_ref = idx
        else:
            if not np.array_equal(y, y_ref):
                raise RuntimeError(f"fold{fold} {name} val_y 与基准不一致")
            if idx is not None and idx_ref is not None and not np.array_equal(idx, idx_ref):
                raise RuntimeError(f"fold{fold} {name} val_idx 与基准不一致")
        names.append(name)
        probs.append(p)
    if len(names) < 2:
        raise RuntimeError(f"fold{fold} 可用成员不足: {names}")
    assert y_ref is not None
    cfg = fit_e1f(names, probs, y_ref)
    fused = fuse_with_config(probs, cfg)
    report = three_class_report(y_ref, fused.argmax(axis=1))
    return {
        "fold": fold,
        "config": cfg.to_dict(),
        "val_metrics": report,
        "n_members": len(names),
        "members": names,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-runs", nargs="*", default=[])
    ap.add_argument("--auto-latest", action="store_true")
    ap.add_argument("--max-folds", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.auto_latest:
        member_dirs = {}
        for name in MEMBER_KEYS:
            p = _find_latest_three(name)
            if p is not None:
                member_dirs[name] = p
                print("auto", name, p)
        if len(member_dirs) < 2:
            raise SystemExit("auto-latest 找到的成员不足")
    else:
        member_dirs = _parse_member_runs(args.member_runs)
        if len(member_dirs) < 2:
            raise SystemExit("请提供至少 2 个 --member-runs name=path")

    folds = []
    for fold in range(args.max_folds):
        # 需要所有成员都有该 fold
        if not all((d / f"fold{fold}" / "val_prob.npy").is_file() for d in member_dirs.values()):
            if fold == 0:
                raise SystemExit("fold0 缺失 val_prob；请先训完成员")
            break
        rec = fit_one_fold(member_dirs, fold)
        folds.append(rec)
        print(
            f"fold{fold}: val_acc={rec['val_metrics']['acc']:.4f} "
            f"T={rec['config']['temperatures']} w={rec['config']['weights']}"
        )

    accs = [float(f["val_metrics"]["acc"]) for f in folds]
    summary = {
        "experiment": 34,
        "track": "A59",
        "device": "5070",
        "member_dirs": {k: str(v) for k, v in member_dirs.items()},
        "n_folds": len(folds),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": folds,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    out = args.out or (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / "e1f_a59"
        / f"e1f_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("wrote", out)
    print(
        f"E1f-A59 Val Acc mean±std = {summary['val_acc_mean']:.4f}±{summary['val_acc_std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
