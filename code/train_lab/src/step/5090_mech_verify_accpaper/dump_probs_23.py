"""方案 24 · W 腿：重放 O1s/O2s/O3s ckpt，导出 val/test 窗级 softmax。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEP = HERE.parent
if str(STEP) not in sys.path:
    sys.path.append(str(STEP))

from arms_registry import ARMS  # noqa: E402
from shared_hparams import SHARED  # noqa: E402
from train_23_kfold import (  # noqa: E402
    WinDS23,
    _collate_win,
    _load_data,
    _pred_key,
    build_model_for_arm,
    eval_indices,
)

# 复用 3s 包 prob dump 格式
_PKG3 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
if str(_PKG3) not in sys.path:
    sys.path.insert(0, str(_PKG3))
from prob_dump import dump_rows_to_csv  # noqa: E402

from _paths import PRE  # noqa: E402

if str(PRE) not in sys.path:
    sys.path.insert(0, str(PRE))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402


@torch.no_grad()
def _dump_fold(
    *,
    arm_id: str,
    run_dir: Path,
    fold: int,
    fold_info: dict,
    device: torch.device,
    hp,
) -> None:
    arm = ARMS[arm_id]
    x_full, y, subjects, trial_id, t0_sec, _meta = _load_data(hp)
    ckpt_path = run_dir / f"fold{fold}" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model_for_arm(arm, hp, n_outputs=3).to(device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    pk = _pred_key(arm)
    ds_kw = dict(geom_id=arm.geom_id, oracle=arm.oracle, t0_sec=t0_sec)
    rows: list[dict] = []
    for split, key in (("val", "val"), ("test", "test")):
        idx = np.where(fold_info["masks"][key])[0]
        idx = eval_indices(idx)
        ds = WinDS23(x_full, y, trial_id, subjects, idx, **ds_kw)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=hp.batch_eval,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_win,
        )
        off = 0
        indices = idx
        for x_in, xf, yy, tid, subj, t0 in loader:
            x_in = x_in.to(device, non_blocking=True)
            xf = xf.to(device, non_blocking=True)
            t0 = t0.to(device, non_blocking=True)
            out = model(
                x_in,
                x_full=xf if arm.use_predictor else None,
                t0_sec=t0,
                train_mode=False,
            )
            probs = F.softmax(out[pk], dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            bs = probs.shape[0]
            for i in range(bs):
                gi = int(indices[off + i])
                p = probs[i]
                rows.append(
                    {
                        "subject": str(subjects[gi]),
                        "fold": fold,
                        "split": split,
                        "trial_id": int(trial_id[gi]),
                        "t0_sec": float(t0_sec[gi]),
                        "pred": int(preds[i]),
                        "y": int(yy[i].item()),
                        "p_max": float(p.max()),
                        "p0": float(p[0]),
                        "p1": float(p[1]),
                        "p2": float(p[2]),
                    }
                )
            off += bs
    out_csv = run_dir / f"fold{fold}" / "prob_dump_three.csv"
    dump_rows_to_csv(out_csv, rows)
    print(f"{arm_id} fold{fold} → {out_csv} ({len(rows)} rows)")


def dump_arm_run(*, arm_id: str, run_dir: Path, hp=None) -> None:
    hp = hp or SHARED
    run_dir = run_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _x, _y, subjects, _tid, _t0, _ = _load_data(hp)
    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        _dump_fold(
            arm_id=arm_id,
            run_dir=run_dir,
            fold=int(info["fold"]),
            fold_info=info,
            device=device,
            hp=hp,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme24 W · dump probs for mech-verify arm")
    p.add_argument("--arm", required=True, choices=("O1s_m", "O2s_m", "O3s_m"))
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    dump_arm_run(arm_id=args.arm, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
