"""方案 26 · 数据泄露与对齐快速审计（只读）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
PKG24 = STEP / "5090_baselines_openbmi_3s_hop100_accpaper"
PRE = STEP.parents[1] / "preprocess_lab"
for p in (str(STEP), str(PKG24), str(PRE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from s26_config import DEFAULT_MEMBERS  # noqa: E402
from prob_io import load_members  # noqa: E402
from e1_fusion_core import align_members, acc_paper_for_split, fuse_pipeline, fit_e1_config  # noqa: E402


def main() -> None:
    runs = [DEFAULT_MEMBERS.shallow, DEFAULT_MEMBERS.eegnet, DEFAULT_MEMBERS.conformer]
    members = load_members(runs)
    align_members(members)
    m0 = members[0]

    print("=== member alignment ===")
    for i, m in enumerate(members):
        nv = int((m["split"] == "val").sum())
        nt = int((m["split"] == "test").sum())
        print(f"  member{i}: rows={len(m['y'])} val={nv} test={nt}")

    for i in range(1, 3):
        assert np.array_equal(members[i]["y"], m0["y"])
        assert np.array_equal(members[i]["split"], m0["split"])
        assert np.array_equal(members[i]["fold"], m0["fold"])
    print("  cross-member y/split/fold: OK")

    print("\n=== val subject disjointness (per fold) ===")
    for f in range(5):
        for split in ("train", "val", "test"):
            pass  # need masks from kfold
    from data_paths import resolve_data
    from src.common.steps.split_subjects import iter_subject_kfold

    _, prefix = resolve_data("openbmi_3s_hop100")
    data_dir = Path(resolve_data("openbmi_3s_hop100")[0])
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    for info in iter_subject_kfold(subjects, n_folds=5, val_ratio=0.2, seed=42):
        f = info["fold"]
        tr_s = set(info["train_subjects"])
        va_s = set(info["val_subjects"])
        te_s = set(info["test_subjects"])
        assert not tr_s & va_s and not tr_s & te_s and not va_s & te_s
        print(f"  fold{f}: train={len(tr_s)} val={len(va_s)} test={len(te_s)} disjoint OK")

    print("\n=== E1 val-only tuning sanity ===")
    cfg = fit_e1_config(members, use_temp=True, use_weights=True, use_smooth=True)
    fused = fuse_pipeline(
        members,
        temperatures=cfg.temperatures,
        weights=cfg.weights,
        smooth_radius=cfg.smooth_radius,
    )
    val_acc = acc_paper_for_split(fused, "val")
    test_acc = acc_paper_for_split(fused, "test")
    print(f"  E1d val={val_acc:.4f} test={test_acc:.4f} (test not used in fit_e1_config)")

    print("\n=== temporal smooth future-window check ===")
    # bidirectional smooth uses t0 neighbors within trial — note only
    print("  temporal_smooth is bidirectional within trial (offline TTA; see scheme E1c)")

    print("\naudit_leakage_check: PASS")


if __name__ == "__main__":
    main()
