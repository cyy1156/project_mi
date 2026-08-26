"""方案 26 · E2 fold0 门控：候选成员融合增量判定。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from e1_fusion_core import acc_paper_for_split, fit_e1_config, fuse_pipeline  # noqa: E402
from prob_io import load_run_three  # noqa: E402
from s26_config import DEFAULT_MEMBERS, E2_FUSION_PP  # noqa: E402


def _filter_fold_split(data: dict, *, fold: int, split: str) -> dict:
    m = (data["fold"] == fold) & (data["split"] == split)
    out = {}
    for k, v in data.items():
        out[k] = v[m]
    return out


def gate_fusion_delta(
    base_runs: list[Path],
    candidate_run: Path,
    *,
    gate_fold: int = 0,
    split: str = "val",
) -> dict:
    base_members = [load_run_three(p) for p in base_runs]
    cand = load_run_three(candidate_run)
    base_fold = [_filter_fold_split(m, fold=gate_fold, split=split) for m in base_members]
    cand_fold = _filter_fold_split(cand, fold=gate_fold, split=split)

    cfg3 = fit_e1_config(base_fold, use_temp=True, use_weights=True, use_smooth=True)
    fused3 = fuse_pipeline(
        base_fold,
        temperatures=cfg3.temperatures,
        weights=cfg3.weights,
        smooth_radius=cfg3.smooth_radius,
    )
    acc3 = acc_paper_for_split(fused3, split)

    pool4 = base_fold + [cand_fold]
    cfg4 = fit_e1_config(pool4, use_temp=True, use_weights=True, use_smooth=True)
    fused4 = fuse_pipeline(
        pool4,
        temperatures=cfg4.temperatures,
        weights=cfg4.weights,
        smooth_radius=cfg4.smooth_radius,
    )
    acc4 = acc_paper_for_split(fused4, split)
    delta_pp = (acc4 - acc3) * 100.0
    verdict = "adopt" if delta_pp >= E2_FUSION_PP else ("report" if delta_pp >= 0.1 else "negative")
    return {
        "gate_fold": gate_fold,
        "split": split,
        "base_runs": [str(p) for p in base_runs],
        "candidate_run": str(candidate_run),
        "acc_paper_3member": acc3,
        "acc_paper_4member": acc4,
        "delta_pp": delta_pp,
        "adopt_ge_pp": E2_FUSION_PP,
        "config_3member": cfg3.to_dict(),
        "config_4member": cfg4.to_dict(),
        "verdict": verdict,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme26 E2 fold0 fusion gate")
    p.add_argument("--candidate-run", type=Path, required=True, help="E2a/E2b .../three run dir")
    p.add_argument("--arm", default="E2", choices=("E2a", "E2b", "E2"))
    p.add_argument(
        "--four-member-base",
        action="store_true",
        help="基池含 T-shallow（与 E1e 四成员一致；方案 §4b 推荐）",
    )
    p.add_argument("--gate-fold", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    m = DEFAULT_MEMBERS
    if args.four_member_base:
        base = [m.shallow, m.t_shallow, m.eegnet, m.conformer]
    else:
        base = [m.shallow, m.eegnet, m.conformer]
        print(
            "WARN: gate uses 3-member base; for §4b use --four-member-base after E1e",
            file=sys.stderr,
        )

    out = gate_fusion_delta(base, args.candidate_run, gate_fold=args.gate_fold)
    out["arm"] = args.arm
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    path = args.out or (HERE / f"e2_gate_{args.arm.lower()}_fold{args.gate_fold}.json")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
