"""方案 24 · 臂 E：三骨干窗级概率平均 → Acc_paper。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prob_dump import load_prob_dump
from trial_metrics import aggregate_windows_to_trials


def _align_fuse(members: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    base = members[0]
    n = len(base["y"])
    for m in members[1:]:
        assert len(m["y"]) == n
        assert np.array_equal(m["y"], base["y"])
        assert np.array_equal(m["trial_id"], base["trial_id"])
        assert np.array_equal(m["subject"], base["subject"])
    probs = np.mean([m["probs"] for m in members], axis=0)
    pred = probs.argmax(axis=1).astype(np.int64)
    pmax = probs.max(axis=1).astype(np.float32)
    out = dict(base)
    out["probs"] = probs
    out["pred"] = pred
    out["p_max"] = pmax
    return out


def acc_paper_for_split(data: dict[str, np.ndarray], split: str) -> float:
    m = data["split"] == split
    trial = aggregate_windows_to_trials(
        data["y"][m],
        data["pred"][m],
        data["subject"][m],
        data["trial_id"][m],
        n_classes=3,
    )
    return float(trial["metrics"]["acc_paper"])


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme24 E · prob fusion replay")
    p.add_argument("--shallow-run", type=Path, required=True, help=".../three")
    p.add_argument("--eegnet-run", type=Path, required=True)
    p.add_argument("--conformer-run", type=Path, required=True)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    def load_run(run_dir: Path) -> dict[str, np.ndarray]:
        dumps = sorted(run_dir.glob("fold*/prob_dump_three.csv"))
        if not dumps:
            dumps = sorted(run_dir.glob("fold*/prob_dump_three_*.csv"))
        if not dumps:
            raise FileNotFoundError(f"no dumps in {run_dir}")
        from prob_dump import merge_prob_dumps

        return merge_prob_dumps(dumps)

    members = [
        load_run(args.shallow_run.resolve()),
        load_run(args.eegnet_run.resolve()),
        load_run(args.conformer_run.resolve()),
    ]
    fused = _align_fuse(members)
    val_acc = acc_paper_for_split(fused, "val")
    test_acc = acc_paper_for_split(fused, "test")
    shallow_test = acc_paper_for_split(members[0], "test")

    out = {
        "shallow_run": str(args.shallow_run),
        "eegnet_run": str(args.eegnet_run),
        "conformer_run": str(args.conformer_run),
        "val_acc_paper_fused": val_acc,
        "test_acc_paper_fused": test_acc,
        "test_acc_paper_shallow_only": shallow_test,
        "delta_test_pp_vs_shallow": (test_acc - shallow_test) * 100.0,
    }
    print(json.dumps(out, indent=2))
    out_path = args.out or (args.shallow_run / "replay_e_fusion.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
