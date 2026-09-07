# -*- coding: utf-8 -*-
"""Exp36 Day1 轻量重训：B1/B2/B3（A59）+ B4（B8-ft 多种子）。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
_B8 = _STEP.parent / "5070_challenge_mi_8ch_ft_accpaper"
_PARENT = _STEP.parent
for p in (_PARENT, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp36_config import exp36_out  # noqa: E402
from data_paths import resolve_data  # noqa: E402


def _load_pkg_module(dir_path: Path, mod_name: str) -> ModuleType:
    path = dir_path / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(f"exp36_{dir_path.name}_{mod_name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    # ensure package dir on path for sibling imports inside module
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))
    spec.loader.exec_module(mod)
    return mod


def _builder(dir_path: Path, name: str):
    if name in ("shallow", "shallow_b"):
        mod = _load_pkg_module(dir_path, "baseline_shallow")
    elif name == "eegnet":
        mod = _load_pkg_module(dir_path, "baseline_eegnet")
    elif name == "conformer":
        mod = _load_pkg_module(dir_path, "baseline_conformer")
    else:
        raise KeyError(name)
    return mod.build_model


def _run_job(
    *,
    pkg: Path,
    track: str,
    arm: str,
    member: str,
    folder_name: str,
    seed: int,
    run_tag: str,
    max_folds: int,
    hp_kwargs: dict,
) -> dict:
    hp_mod = _load_pkg_module(pkg, "shared_hparams")
    tr_mod = _load_pkg_module(pkg, "task_runner")
    loso_mod = _load_pkg_module(pkg, "loso")

    base = hp_mod.SHARED
    if member == "conformer":
        if hasattr(hp_mod, "hp_for_conformer"):
            base = hp_mod.hp_for_conformer(base)
        elif hasattr(hp_mod, "hp_conformer"):
            base = hp_mod.hp_conformer(base)
    d = asdict(base)
    d["seed"] = int(seed)
    d.update(hp_kwargs)
    if track == "b8":
        d["init_from_openbmi"] = True
    hp = hp_mod.SharedTrainHP(**{k: d[k] for k in d if k in hp_mod.SharedTrainHP.__dataclass_fields__})

    build = _builder(pkg, member)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("device", torch.cuda.get_device_name(0), flush=True)

    data_dir, prefix = resolve_data(hp.data_tag)
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)

    root = exp36_out() / ("A59" if track == "a59" else "B8ft") / arm / folder_name
    out_dir = root / f"run_{run_tag}" / "three"
    if (out_dir / "summary.json").is_file() and max_folds == 0:
        print("SKIP existing", out_dir, flush=True)
        return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": 36,
        "arm": arm,
        "track": track,
        "member": member,
        "folder_name": folder_name,
        "seed": seed,
        "hp": asdict(hp),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    folds = []
    t0 = time.time()
    for info in loso_mod.iter_loso6(subjects):
        if max_folds > 0 and info["fold"] >= max_folds:
            break
        folds.append(
            tr_mod.train_one_fold(
                fold_info=info,
                X=X,
                y=y,
                device=device,
                hp=hp,
                out_dir=out_dir,
                model_name=member,
                build_model=build,
            )
        )
    accs = [float(r["best_val_acc"]) for r in folds]
    summary = {
        **meta,
        "n_folds": len(folds),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": folds,
        "elapsed_sec": time.time() - t0,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{arm}/{track}] {folder_name} "
        f"{summary['val_acc_mean']:.4f}±{summary['val_acc_std']:.4f} ({summary['elapsed_sec']:.0f}s)",
        flush=True,
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B1,B2,B3,B4")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()
    tag = args.run_tag.strip() or f"exp36_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}

    plan: list[tuple] = []
    if "B1" in wanted:
        for seed in (43, 44):
            plan.append((_A59, "a59", "B1", "conformer", f"conformer_s{seed}", seed, {}))
    if "B2" in wanted:
        for member in ("shallow", "eegnet"):
            plan.append((_A59, "a59", "B2", member, f"{member}_s43", 43, {}))
    if "B3" in wanted:
        plan.append((_A59, "a59", "B3", "conformer", "conformer_pat40_s42", 42, {"patience": 40}))
        plan.append((_A59, "a59", "B3", "conformer", "conformer_drop025_s42", 42, {"drop_prob": 0.25}))
    if "B4" in wanted:
        for member in ("shallow", "shallow_b", "eegnet", "conformer"):
            plan.append((_B8, "b8", "B4", member, f"{member}_s43", 43, {}))
        plan.append((_B8, "b8", "B4", "conformer", "conformer_s44", 44, {}))

    results = []
    for pkg, track, arm, member, folder, seed, kw in plan:
        print(f"=== {arm} {folder} ===", flush=True)
        results.append(
            _run_job(
                pkg=pkg,
                track=track,
                arm=arm,
                member=member,
                folder_name=folder,
                seed=seed,
                run_tag=tag,
                max_folds=args.max_folds,
                hp_kwargs=kw,
            )
        )

    out = exp36_out() / "day1_train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": tag,
        "jobs": [
            {
                "arm": r.get("arm"),
                "folder_name": r.get("folder_name"),
                "member": r.get("member"),
                "seed": r.get("seed"),
                "val_acc_mean": r.get("val_acc_mean"),
                "val_acc_std": r.get("val_acc_std"),
                "out_dir": r.get("out_dir"),
                "elapsed_sec": r.get("elapsed_sec"),
            }
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
